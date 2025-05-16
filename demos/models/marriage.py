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



def update_married_households(persons, households, marriage_list):
    """
    Update the marriage status of individuals and create new households

    Args:
        persons (DataFrameWrapper): DataFrameWrapper of the persons table
        households (DataFrameWrapper): DataFrameWrapper of the households table
        marriage_list (pd.Series): Pandas Series of the married individuals

    Returns:
        None
    """
    # print("Updating persons and households...")
    p_df = persons.local
    household_cols = households.local_columns
    household_df = households.local
    persons_cols = persons.local_columns
    persons_local_cols = persons.local_columns
    hh_df = households.to_frame(columns=["lcm_county_id"])
    hh_df.reset_index(inplace=True)
    p_df["new_mar"] = marriage_list
    p_df["new_mar"].fillna(0, inplace=True)
    relevant = p_df[p_df["new_mar"] > 0].copy()
    # print("New marriages:", (relevant["new_mar"] ==2).sum())
    # print("New cohabs:", (relevant["new_mar"] ==1).sum())
    # Ensure an even number of people get married
    if relevant[relevant["new_mar"] == 1].shape[0] % 2 != 0:
        sampled = p_df[p_df["new_mar"] == 1].sample(1)
        sampled.new_mar = 0
        p_df.update(sampled)
        relevant = p_df[p_df["new_mar"] > 0].copy()

    if relevant[relevant["new_mar"] == 2].shape[0] % 2 != 0:
        sampled = p_df[p_df["new_mar"] == 2].sample(1)
        sampled.new_mar = 0
        p_df.update(sampled)
        relevant = p_df[p_df["new_mar"] > 0].copy()

    relevant.sort_values("new_mar", inplace=True)

    relevant = (
        relevant.reset_index().merge(hh_df, on=["household_id"]).set_index("person_id")
    )
    p_df = p_df.reset_index().merge(hh_df, on=["household_id"]).set_index("person_id")

    # print("Pair people.")
    min_mar = relevant[relevant["new_mar"] == 2]["person_sex"].value_counts().min()
    min_cohab = relevant[relevant["new_mar"] == 1]["person_sex"].value_counts().min()

    female_mar = relevant[
        (relevant["new_mar"] == 2) & (relevant["person_sex"] == "female")
    ].sample(min_mar)
    male_mar = relevant[
        (relevant["new_mar"] == 2) & (relevant["person_sex"] == "male")
    ].sample(min_mar)
    female_coh = relevant[
        (relevant["new_mar"] == 1) & (relevant["person_sex"] == "female")
    ].sample(min_cohab)
    male_coh = relevant[
        (relevant["new_mar"] == 1) & (relevant["person_sex"] == "male")
    ].sample(min_cohab)

    def brute_force_matching(male, female):
        """Function to run brute force marriage matching.

        TODO: Improve the matchmaking process
        TODO: Account for same-sex marriages

        Args:
            male (DataFrame): DataFrame of the male side
            female (DataFrame): DataFrame of the female side

        Returns:
            DataFrame: DataFrame of newly formed households
        """
        ordered_households = pd.DataFrame()
        male_mar = male.sample(frac=1)
        female_mar = female.sample(frac=1)
        for index in np.arange(female_mar.shape[0]):
            dist = cdist(
                female_mar.iloc[index][["age", "earning"]]
                .to_numpy()
                .reshape((1, 2))
                .astype(float),
                male_mar[["age", "earning"]].to_numpy(),
                "euclidean",
            )
            arg = dist.argmin()
            household_new = pd.DataFrame([female_mar.iloc[index], male_mar.iloc[arg]])
            # household_new["household_group"] = index + 1
            # household_new["new_household_id"] = -99
            # household_new["stay"] = -99
            male_mar = male_mar.drop(male_mar.iloc[arg].name)
            ordered_households = pd.concat([ordered_households, household_new])

        return ordered_households

    cohabitate = brute_force_matching(male_coh, female_coh)
    cohabitate.index.name = "person_id"

    married = brute_force_matching(male_mar, female_mar)
    married.index.name = "person_id"

    def relate(size, marriage=True):
        result = np.zeros(size)
        if marriage:
            result[1::2] = result[1::2] + 1
        else:
            result[1::2] = result[1::2] + 13
        return result

    def swap(arr):
        result = np.empty_like(arr)
        result[::2] = arr[1::2]
        result[1::2] = arr[::2]
        return result

    def first(size):
        result = np.zeros(size)
        result[::2] = result[::2] + 1
        return result

    married["household_group"] = np.repeat(np.arange(len(married.index) / 2), 2)
    cohabitate["household_group"] = np.repeat(np.arange(len(cohabitate.index) / 2), 2)

    married = married.sort_values(
        by=["household_group", "earning"], ascending=[True, False]
    )
    cohabitate = cohabitate.sort_values(
        by=["household_group", "earning"], ascending=[True, False]
    )

    cohabitate["household_group"] = (
        cohabitate["household_group"] + married["household_group"].max() + 1
    )

    married["new_relate"] = relate(married.shape[0])
    cohabitate["new_relate"] = relate(cohabitate.shape[0], False)

    final = pd.concat([married, cohabitate])

    final["first"] = first(final.shape[0])
    final["partner"] = swap(final.index)
    final["partner_house"] = swap(final["household_id"])
    final["partner_relate"] = swap(final["relate"])

    final["new_household_id"] = -99
    final["stay"] = -99

    final = final[~(final["household_id"] == final["partner_house"])]

    # Pair up the people and classify what type of marriage it is
    # TODO speed up this code by a lot
    # relevant.sort_values("new_mar", inplace=True)
    # married = final[final["new_mar"]==1].copy()
    # cohabitation = final[final["new_mar"]==2].copy()
    # Stay documentation
    # 0 - leaves household
    # 1 - stays
    # 2 - this persons household becomes a root household (a root household absorbs the leaf household)
    # 4 - this persons household becomes a leaf household
    # 3 - this person leaves their household and creates a new household with partner
    # Marriage
    CONDITION_1 = (
        (final["first"] == 1) & (final["relate"] == 0) & (final["partner_relate"] == 0)
    )
    final.loc[final[CONDITION_1].index, "stay"] = 1
    final.loc[final[CONDITION_1]["partner"].values, "stay"] = 0

    CONDITION_2 = (
        (final["first"] == 1) & (final["relate"] == 0) & (final["partner_relate"] != 0)
    )
    final.loc[final[CONDITION_2].index, "stay"] = 1
    final.loc[final[CONDITION_2]["partner"].values, "stay"] = 0

    CONDITION_3 = (
        (final["first"] == 1) & (final["relate"] != 0) & (final["partner_relate"] == 0)
    )
    final.loc[final[CONDITION_3].index, "stay"] = 0
    final.loc[final[CONDITION_3]["partner"].values, "stay"] = 1

    CONDITION_4 = (
        (final["first"] == 1) & (final["relate"] != 0) & (final["partner_relate"] != 0)
    )
    final.loc[final[CONDITION_4].index, "stay"] = 3
    final.loc[final[CONDITION_4]["partner"].values, "stay"] = 3

    new_household_ids = np.arange(final[CONDITION_4].index.shape[0])
    new_household_ids_max = new_household_ids.max() + 1
    final.loc[final[CONDITION_4].index, "new_household_id"] = new_household_ids
    final.loc[
        final[CONDITION_4]["partner"].values, "new_household_id"
    ] = new_household_ids

    # print("Finished Pairing")
    # print("Updating households and persons table")
    # print(final.household_id.unique().shape[0])
    metadata = orca.get_table("metadata").to_frame()
    max_hh_id = metadata.loc["max_hh_id", "value"]
    current_max_id = max(max_hh_id, household_df.index.max())

    final["hh_new_id"] = np.where(
        final["stay"].isin([1]),
        final["household_id"],
        np.where(
            final["stay"].isin([0]),
            final["partner_house"],
            final["new_household_id"] + current_max_id + 1,
        ),
    )

    # final["new_relate"] = relate(final.shape[0])
    ## NEED TO SEPARATE MARRIED FROM COHABITATE

    # Households where everyone left
    # household_matched = (p_df[p_df["household_id"].isin(final["household_id"].unique())].groupby("household_id").size() == final.groupby("household_id").size())
    # removed_hh_values = household_matched[household_matched==True].index.values

    # Households where head left
    household_ids_reorganized = final[(final["stay"] == 0) & (final["relate"] == 0)][
        "household_id"
    ].unique()

    p_df.loc[final.index, "household_id"] = final["hh_new_id"]
    p_df.loc[final.index, "relate"] = final["new_relate"]
    # print("HH SHAPE 1:", p_df["household_id"].unique().shape[0])

    households_restructuring = p_df.loc[
        p_df["household_id"].isin(household_ids_reorganized)
    ]

    households_restructuring = households_restructuring.sort_values(
        by=["household_id", "earning"], ascending=False
    )
    households_restructuring.loc[
        households_restructuring.groupby(["household_id"]).head(1).index, "relate"
    ] = 0

    household_df = household_df.loc[household_df.index.isin(p_df["household_id"])]

    # print("HH SHAPE 1:", p_df["household_id"].unique().shape[0])

    # leaf_hh = final.loc[final["stay"]==4, ["household_id", "partner_house"]]["household_id"].to_list()
    # root_hh = final.loc[final["stay"]==2, ["household_id", "partner_house"]]["household_id"].to_list()
    # new_hh = final.loc[final["stay"]==3, "hh_new_id"].to_list()

    # household_mapping_dict = {leaf_hh[i]: root_hh[i] for i in range(len(root_hh))}

    # household_df = household_df.reset_index()

    # class MyDict(dict):
    #     def __missing__(self, key):
    #         return key

    # recodes = MyDict(household_mapping_dict)

    # household_df["household_id"] = household_df["household_id"].map(recodes)
    # p_df["household_id"] = p_df["household_id"].map(recodes)

    p_df = p_df.sort_values("relate")

    p_df["person"] = 1
    p_df["is_head"] = np.where(p_df["relate"] == 0, 1, 0)
    p_df["race_head"] = p_df["is_head"] * p_df["race_id"]
    p_df["age_head"] = p_df["is_head"] * p_df["age"]
    p_df["hispanic_head"] = p_df["is_head"] * p_df["hispanic"]
    p_df["child"] = np.where(p_df["relate"].isin([2, 3, 4, 14]), 1, 0)
    p_df["senior"] = np.where(p_df["age"] >= 65, 1, 0)
    p_df["age_gt55"] = np.where(p_df["age"] >= 55, 1, 0)

    p_df = p_df.sort_values(by=["household_id", "relate"])
    household_agg = p_df.groupby("household_id").agg(
        income=("earning", "sum"),
        race_of_head=("race_id", "first"),
        age_of_head=("age", "first"),
        size=("person", "sum"),
        workers=("worker", "sum"),
        hispanic_head=("hispanic_head", "sum"),
        # lcm_county_id=("lcm_county_id", "first"),
        persons_age_gt55=("age_gt55", "sum"),
        seniors=("senior", "sum"),
        children=("child", "sum"),
        persons=("person", "sum"),
    )

    # household_agg["lcm_county_id"] = household_agg["lcm_county_id"]
    household_agg["gt55"] = np.where(household_agg["persons_age_gt55"] > 0, 1, 0)
    household_agg["gt2"] = np.where(household_agg["persons"] > 2, 1, 0)
    # household_agg["sf_detached"] = "unknown"
    # household_agg["serialno"] = "unknown"
    # household_agg["cars"] = np.random.ran
    # dom_integers(0, 2, size=household_agg.shape[0])
    household_agg["hh_workers"] = np.where(
        household_agg["workers"] == 0,
        "none",
        np.where(household_agg["workers"] == 1, "one", "two or more"),
    )
    household_agg["hh_age_of_head"] = np.where(
        household_agg["age_of_head"] < 35,
        "lt35",
        np.where(household_agg["age_of_head"] < 65, "gt35-lt65", "gt65"),
    )
    household_agg["hh_race_of_head"] = np.where(
        household_agg["race_of_head"] == 1,
        "white",
        np.where(
            household_agg["race_of_head"] == 2,
            "black",
            np.where(household_agg["race_of_head"].isin([6, 7]), "asian", "other"),
        ),
    )
    household_agg["hispanic_head"] = np.where(
        household_agg["hispanic_head"] == 1, "yes", "no"
    )
    household_agg["hh_size"] = np.where(
        household_agg["size"] == 1,
        "one",
        np.where(
            household_agg["size"] == 2,
            "two",
            np.where(household_agg["size"] == 3, "three", "four or more"),
        ),
    )
    household_agg["hh_children"] = np.where(household_agg["children"] >= 1, "yes", "no")
    household_agg["hh_seniors"] = np.where(household_agg["seniors"] >= 1, "yes", "no")
    household_agg["hh_income"] = np.where(
        household_agg["income"] < 30000,
        "lt30",
        np.where(
            household_agg["income"] < 60,
            "gt30-lt60",
            np.where(
                household_agg["income"] < 100,
                "gt60-lt100",
                np.where(household_agg["income"] < 150, "gt100-lt150", "gt150"),
            ),
        ),
    )

    # agg_households = household_df.groupby("household_id").agg(serialno = ("serialno", "first"), # change to min once you change the serial number for all
    #                                         cars = ("cars", "sum"),
    #                                         # income = ("income", "sum"),
    #                                         # workers = ("workers", "sum"),
    #                                         tenure = ("tenure", "first"),
    #                                         recent_mover = ("recent_mover", "first"),
    #                                         sf_detached = ("sf_detached", "first"),
    #                                         lcm_county_id = ("lcm_county_id", "first"),
    #                                         block_id=("block_id", "first")) # we need hhtype here

    # agg_households["hh_cars"] = np.where(agg_households["cars"] == 0, "none",
    #                                         np.where(agg_households["cars"] == 1, "one", "two or more"))

    # household_df = household_df.drop_duplicates(subset="household_id")

    # household_df = household_df.set_index("household_id")
    # household_df.update(agg_households)
    household_df.update(household_agg)

    final["MAR"] = np.where(final["new_mar"] == 2, 1, final["MAR"])
    p_df.update(final["MAR"])

    # print("HH SHAPE 2:", p_df["household_id"].unique().shape[0])

    new_hh = household_agg.loc[
        ~household_agg.index.isin(household_df.index.unique())
    ].copy()
    new_hh["serialno"] = "-1"
    new_hh["cars"] = np.random.choice([0, 1, 2], size=new_hh.shape[0])
    new_hh["hispanic_status_of_head"] = "-1"
    new_hh["tenure"] = "-1"
    new_hh["recent_mover"] = "-1"
    new_hh["sf_detached"] = "-1"
    new_hh["hh_cars"] = np.where(
        new_hh["cars"] == 0, "none", np.where(new_hh["cars"] == 1, "one", "two or more")
    )
    new_hh["tenure_mover"] = "-1"
    new_hh["block_id"] = "-1"
    new_hh["hh_type"] = "-1"
    household_df = pd.concat([household_df, new_hh])

    # p_df.update(relevant["household_id"])

    #
    # household_df = household_df.set_index("household_id")
    # new_households = household_agg.loc[household_agg.index.isin(new_hh)].copy()
    # new_households["serialno"] = "-1"
    # new_households["cars"] = np.random.choice([0, 1, 2], size=new_households.shape[0])
    # new_households["hispanic_status_of_head"] = -1
    # new_households["tenure"] = -1
    # new_households["recent_mover"] = "-1"
    # new_households["sf_detached"] = "-1"
    # new_households["hh_cars"] = np.where(new_households["cars"] == 0, "none",
    #                                      np.where(new_households["cars"] == 1, "one", "two or more"))
    # new_households["tenure_mover"] = "-1"
    # new_households["block_id"] = "-1"
    # new_households["hh_type"] = -1
    # household_df = pd.concat([household_df, new_households])

    # print('Time to run marriage', sp.duration)
    orca.add_table("households", household_df[household_cols])
    orca.add_table("persons", p_df[persons_cols])
    # orca.add_injectable(
    #     "max_hh_id", max(orca.get_injectable("max_hh_id"), household_df.index.max())
    # )

    # print("households size", household_df.shape[0])

    metadata = orca.get_table("metadata").to_frame()
    max_hh_id = metadata.loc["max_hh_id", "value"]
    max_p_id = metadata.loc["max_p_id", "value"]
    if household_df.index.max() > max_hh_id:
        metadata.loc["max_hh_id", "value"] = household_df.index.max()
    if p_df.index.max() > max_p_id:
        metadata.loc["max_p_id", "value"] = p_df.index.max()
    orca.add_table("metadata", metadata)

    married_table = orca.get_table("marriage_table").to_frame()
    if married_table.empty:
        married_table = pd.DataFrame(
            [[(marriage_list == 1).sum(), (marriage_list == 2).sum()]],
            columns=["married", "cohabitated"],
        )
    else:
        married_table = married_table.append(
            {
                "married": (marriage_list == 1).sum(),
                "cohabitated": (marriage_list == 2).sum(),
            },
            ignore_index=True,
        )
    orca.add_table("marriage_table", married_table)


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