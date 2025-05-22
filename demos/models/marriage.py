import time
import orca
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from templates import estimated_models, modelmanager as mm


def update_married_households_random(persons, marriage_list, get_new_households, graveyard):
    """
    Update the marriage status of individuals and create new households

    Args:
        persons (DataFrameWrapper): DataFrameWrapper of the persons table
        households (DataFrameWrapper): DataFrameWrapper of the households table
        marriage_list (pd.Series): Pandas Series of the married individuals
    Returns:
        None
    """
    married_reindexed = marriage_list.reindex(persons.local.index).fillna(0)
    
    # TODO: What is this checking for?
    if (married_reindexed == 1).sum() <= 10 or (married_reindexed == 2).sum() <= 10:
        return

    # Make match between male and female candidates
    male_index = persons["person_sex"] == "male"
    female_index = persons["person_sex"] == "female"
    n_candidates_newmarried_male   = ((married_reindexed == 2) &   male_index).sum()
    n_candidates_newmarried_female = ((married_reindexed == 2) & female_index).sum()
    n_candidates_cohab_male   = ((married_reindexed == 1) &   male_index).sum()
    n_candidates_cohab_female = ((married_reindexed == 1) & female_index).sum()
    
    # Determine the number of couples to be formed
    n_weddings = min(n_candidates_newmarried_male, n_candidates_newmarried_female)
    n_newcohabs = min(n_candidates_cohab_male, n_candidates_cohab_female)

    # TODO: Check if this logic is correct
    # I believe this could be removed
    if n_weddings == 0 and n_newcohabs == 0:
        return
    
    ## Selecting individuals for marriage and cohabitation
    female_newmarried = persons.local.loc[(married_reindexed == 2) & female_index][["age", "household_id", "earning", "relate"]].sort_index(axis=0).sample(n_weddings, random_state=orca.get_injectable("year") + 100).copy()
    male_newmarried   = persons.local.loc[(married_reindexed == 2) &   male_index][["age", "household_id", "earning", "relate"]].sort_index(axis=0).sample(n_weddings, random_state=orca.get_injectable("year") + 110).copy()
    female_newcohab = persons.local.loc[(married_reindexed == 1) & female_index][["age", "household_id", "earning", "relate"]]  .sort_index(axis=0).sample(n_newcohabs, random_state=orca.get_injectable("year") + 120).copy()
    male_newcohab   = persons.local.loc[(married_reindexed == 1) &   male_index][["age", "household_id", "earning", "relate"]]  .sort_index(axis=0).sample(n_newcohabs, random_state=orca.get_injectable("year") + 130).copy()
    
    ## Modifying auxiliary dataframe to compute new relation and household_id
    ### Pairs are selected by age
    female_newmarried.sort_values("age", inplace=True)
    male_newmarried.sort_values("age", inplace=True)
    newmarried = pd.concat([male_newmarried, female_newmarried], axis=0)         # NOTE: This order is important, relate = 0 is assigned to male
    newmarried["hh_group"] = np.arange(len(newmarried)) % (len(newmarried) // 2) # [0, 1, 2, ..., n_weddings -1, 0, 1, ..., n_weddings - 1]
    
    # TODO: This part is for comparison to other experiments
    np.random.seed(orca.get_injectable("year") + 140)
    newmarried["rnd"] = np.random.random(len(newmarried))
    
    newmarried.sort_values(by=["hh_group", "earning", "rnd"], ascending=[True, False, True], inplace=True)
    newmarried["new_relate"] = np.arange(len(newmarried)) % 2                    # [0, 1, 0, 1, ...]
    newmarried["did_marry"] = True

    female_newcohab.sort_values("age", inplace=True)
    male_newcohab.sort_values("age", inplace=True)
    newcohab = pd.concat([male_newcohab, female_newcohab], axis=0) # NOTE: This order is important, relate = 0 is assigned to female
    newcohab["hh_group"] = (np.arange(len(newcohab)) % (len(newcohab) // 2)) + newmarried["hh_group"].max() + 1

    np.random.seed(orca.get_injectable("year") + 150)
    newcohab["rnd"] = np.random.random(len(newcohab))

    newcohab.sort_values(by=["hh_group", "earning", "rnd"], ascending=[True, False, True], inplace=True)
    newcohab["new_relate"] = (np.arange(len(newcohab)) % 2) * 13 # [0, 13, 0, 13, ...]
    newcohab["did_marry"] = False

    all_df = pd.concat([newmarried, newcohab])
    len_all_df = len(all_df)
    
    ### Matching each person with their partner
    #### Computing an index that selects the partner
    #### This evaluates to [1, 0, 3, 2, 5, 4 ...]
    #### [0, 1, 2, 3, ...] + [1, 0, 1, 0, ...] + [0, -1, 0, -1, ...]
    swap_selector = np.arange(len_all_df) + (1 - (np.arange(len_all_df) % 2)) + ((-1) * (np.arange(len_all_df) % 2))
    swap_index = all_df.index[swap_selector]

    all_df["first"] = 1 - (np.arange(len_all_df) % 2)
    all_df["partner_id"] = swap_index.values
    all_df["partner_house_id"] = all_df.loc[swap_index]["household_id"].values # NOTE: The `.values` part is very important
    all_df["partner_relate"] = all_df.loc[swap_index]["relate"].values

    #### NOTE: This prevents members of the same family to be partners. There must be a better way
    all_df = all_df[~(all_df["household_id"] == all_df["partner_house_id"])]

    ## Update household information
    ### We are going to store the new household id of each person in the `new_hh_id` column of the auxiliary df
    head_index = all_df.relate == 0
    first_index = all_df["first"] == 1 # NOTE: `first` is a method of pd.DataFrame so we need to use this notation
    all_df["new_hh_id"] = np.nan

    ### If first is head of household, the new partner moves in
    first_and_head_index = head_index & first_index
    first_and_head_partners_index = all_df.loc[first_and_head_index, "partner_id"].values
    all_df.loc[first_and_head_index, "new_hh_id"] = all_df.loc[first_and_head_index, "household_id"].values
    all_df.loc[first_and_head_partners_index, "new_hh_id"] = all_df.loc[first_and_head_index, "household_id"].values

    ### If first is not head, but new partner is, first moves in
    first_and_not_head_index = ~head_index & first_index & (all_df.partner_relate == 0)
    first_and_not_head_partners_index = all_df.loc[first_and_not_head_index, "partner_id"].values
    all_df.loc[first_and_not_head_index, "new_hh_id"] = all_df.loc[first_and_not_head_partners_index, "household_id"].values
    all_df.loc[first_and_not_head_partners_index, "new_hh_id"] = all_df.loc[first_and_not_head_index, "new_hh_id"].values

    ### If neither is head, form a new household
    neither_head_index = (all_df.relate != 0) & (all_df.partner_relate != 0)
    neither_head_not_first_index = all_df.loc[first_index & neither_head_index].partner_id.values
    new_hh_ids = get_new_households((first_index & neither_head_index).sum(), persons, graveyard)

    #### Set new households for heads and not heads
    all_df.loc[first_index & neither_head_index, "new_hh_id"] = new_hh_ids
    all_df.loc[neither_head_not_first_index, "new_hh_id"] = new_hh_ids

    ### This is an important sanity check
    assert all_df.new_hh_id.isnull().sum() == 0, "Some people were not assigned a household"

    # Finally update values in persons table
    persons.local.loc[all_df.index, "household_id"] = all_df["new_hh_id"]
    persons.local.loc[all_df.index, "relate"] = all_df["new_relate"]
    persons.local.loc[all_df[all_df.did_marry].index, "MAR"] = 1

    ## Decide who is household head in the households where the head left
    head_left_index = (all_df.relate == 0) & (all_df.household_id != all_df.new_hh_id)
    head_left_households = all_df[head_left_index]["household_id"].unique()
    
    ### This gets the person in each household with no head
    ### The head(1) is not the same as first()!
    new_household_heads = persons.local[persons["household_id"].isin(head_left_households)] \
        .sort_values(["household_id", "earning"], ascending=False) \
        .groupby("household_id") \
        .head(1) \
        .index
    persons.local.loc[new_household_heads, "relate"] = 0


def update_divorce(persons, divorce_list, get_new_households, graveyard):
    """
    Updating stats for divorced households

    Args:
        persons (DataFrameWrapper): DataFrameWrapper of the persons table
        households (DataFrameWrapper): DataFrameWrapper of the households table
        divorce_list (pd.Series): pandas Series of the divorced households

    Returns:
        None
    """
    divorced_household_ids = divorce_list[divorce_list.astype(bool)].index
    person_in_divorced_household_index = persons["household_id"].isin(divorced_household_ids)
    head_and_spose_index = ((persons["relate"] == 0) | (persons["relate"] == 1)) & (persons["MAR"] == 1)

    people_divorcing_groupby = persons.local.loc[person_in_divorced_household_index & head_and_spose_index].sort_index().groupby("household_id")
    assert (people_divorcing_groupby.size() != 2).sum() == 0, "Some divorcing households have more than 2 people eligible for divorce"

    person_leaving_ids = people_divorcing_groupby.sample(n=1, random_state=orca.get_injectable("year") + 250).index
    person_leaving_index = persons.local.index.isin(person_leaving_ids)
    person_staying_index = person_in_divorced_household_index & head_and_spose_index & ~person_leaving_index

    # Update columns
    ## People leaving get a new household id
    persons.local.loc[person_leaving_index, "household_id"] = get_new_households(person_leaving_index.sum(), persons, graveyard)
    persons.local.loc[person_leaving_index, "relate"] = 0
    persons.local.loc[person_leaving_index, "MAR"] = 3
    persons.local.loc[person_leaving_index, "member_id"] = 1 # TODO: Needed?

    ## Updates for people staying
    persons.local.loc[person_staying_index, "relate"] = 0
    persons.local.loc[person_staying_index, "MAR"] = 3

    ## Update member_id column. TODO: What is this for?
    staying_household_index = person_in_divorced_household_index & ~person_leaving_index
    staying_member_id_filter = persons["member_id"] != 1
    persons.local.loc[staying_household_index & staying_member_id_filter, "member_id"] = persons.local\
                                                .loc[staying_household_index & staying_member_id_filter, "member_id"] - 1
    persons.local.loc[staying_household_index & ~staying_member_id_filter, "member_id"] = persons.local\
                                                .loc[staying_household_index & ~staying_member_id_filter, "relate"]