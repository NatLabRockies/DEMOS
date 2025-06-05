import time
import orca
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from templates import estimated_models, modelmanager as mm


@orca.step("household_divorce")
def household_divorce(persons, households):
    """
    Running the household divorce tble

    Args:
        persons (DataFrameWrapper): DataFrameWrapper of the persons table
        households (DataFrameWrapper): DataFrameWrapper of the households table

    Returns:
        None
    """
    t0 = time.time()
    households_df = orca.get_table("households").local
    households_df["divorced"] = -99
    orca.add_table("households", households_df)
    persons_df = orca.get_table("persons").local
    t1 = time.time()
    total = t1-t0
    # print("Pulling data:", total)
    t0 = time.time()
    ELIGIBLE_HOUSEHOLDS = list(
        persons_df[(persons_df["relate"].isin([0, 1])) & (persons_df["MAR"] == 1)][
            "household_id"
        ]
        .unique()
        .astype(int)
    )
    sizes = (
        persons_df[
            persons_df["household_id"].isin(ELIGIBLE_HOUSEHOLDS)
            & (persons_df["relate"].isin([0, 1]))
        ]
        .groupby("household_id")
        .size()
    )
    ELIGIBLE_HOUSEHOLDS = sizes[(sizes == 2)].index.to_list()
    t1 = time.time()
    total = t1-t0
    # print("Eligibility time:", total)
    # print("eligible households are", len(ELIGIBLE_HOUSEHOLDS))
    t0 = time.time()
    divorce_model = mm.get_step("divorce")
    t1 = time.time()
    total = t1-t0
    # print("retrieving model:", total)
    list_ids = str(ELIGIBLE_HOUSEHOLDS)
    divorce_model.filters = "index in " + list_ids
    divorce_model.out_filters = "index in " + list_ids
    divorce_model.run()
    t1 = time.time()
    total = t1-t0
    # print("Running model:", total)
    t0 = time.time()
    divorce_list = divorce_model.choices.astype(int)
    t1 = time.time()
    total = t1-t0
    # print("Converting to int:", total)
    update_divorce(persons, households, divorce_list)


def update_married_households_random(persons, households, marriage_list):
    """
    Update the marriage status of individuals and create new households

    YE: ***

    Args:
        persons (DataFrameWrapper): DataFrameWrapper of the persons table
        households (DataFrameWrapper): DataFrameWrapper of the households table
        marriage_list (pd.Series): Pandas Series of the married individuals
    Returns:
        None
    """
    # print("Updating persons and households...")
    p_df = orca.get_table('persons').local
    household_cols = households.local_columns
    household_df = orca.get_table('households').local
    persons_cols = persons.local_columns
    persons_local_cols = persons.local_columns
    hh_df = orca.get_table('households').to_frame(columns=["lcm_county_id"])
    hh_df.reset_index(inplace=True)
    # print("Indices duplicated:",p_df.index.duplicated().sum())
    p_df["new_mar"] = marriage_list
    p_df["new_mar"].fillna(0, inplace=True)
    relevant = p_df[p_df["new_mar"] > 0].copy()
    # print("New married persons:", (relevant["new_mar"] ==2).sum())
    # print("New cohabitating persons:", (relevant["new_mar"] ==1).sum())
    # print("Still Single:",(relevant["new_mar"] ==0).sum())
    # breakpoint()
    if ((relevant["new_mar"] ==1).sum() <= 10) or ((relevant["new_mar"] ==2).sum() <= 10):
        return None
    # breakpoint()
    # Ensure an even number of people get married
    # if relevant[relevant["new_mar"]==1].shape[0] % 2 != 0:
    #     sampled = p_df[p_df["new_mar"]==1].sample(1)
    #     sampled.new_mar = 0
    #     p_df.update(sampled)
    #     relevant = p_df[p_df["new_mar"] > 0].copy()

    # if relevant[relevant["new_mar"]==2].shape[0] % 2 != 0:
    #     sampled = p_df[p_df["new_mar"]==2].sample(1)
    #     sampled.new_mar = 0
    #     p_df.update(sampled)
    #     relevant = p_df[p_df["new_mar"] > 0].copy()

    relevant.sort_values("new_mar", inplace=True)

    relevant = relevant.reset_index().merge(hh_df, on=["household_id"]).set_index("person_id")
    p_df = p_df.reset_index().merge(hh_df, on=["household_id"]).set_index("person_id")


    def swap(arr):
        result = np.empty_like(arr)
        result[::2] = arr[1::2]
        result[1::2] = arr[::2]
        return result

    def first(size):
        result = np.zeros(size)
        result[::2] = result[::2] + 1
        return result

    def relate(size, marriage=True):
        result = np.zeros(size)
        if marriage:
            result[1::2] = result[1::2] + 1
        else:
            result[1::2] = result[1::2] + 13
        return result
    
    min_mar_male = relevant[(relevant["new_mar"] == 2) & (relevant["person_sex"] == "male")].shape[0]
    min_mar_female = relevant[(relevant["new_mar"] == 2) & (relevant["person_sex"] == "female")].shape[0]

    min_cohab_male = relevant[(relevant["new_mar"] == 1) & (relevant["person_sex"] == "male")].shape[0]
    min_cohab_female = relevant[(relevant["new_mar"] == 1) & (relevant["person_sex"] == "female")].shape[0]
    
    min_mar = int(min(min_mar_male, min_mar_female))
    min_cohab = int(min(min_cohab_male, min_cohab_female))
    
    # print("Number of marriages:", min_mar)
    # print("Number of cohabitations:", min_cohab)
    if (min_mar == 0) or (min_mar == 0):
        return None

    # breakpoint()
    female_mar = relevant[(relevant["new_mar"] == 2) & (relevant["person_sex"] == "female")].sort_index(axis=0).sample(min_mar)
    male_mar = relevant[(relevant["new_mar"] == 2) & (relevant["person_sex"] == "male")]    .sort_index(axis=0).sample(min_mar)
    female_coh = relevant[(relevant["new_mar"] == 1) & (relevant["person_sex"] == "female")].sort_index(axis=0).sample(min_cohab)
    male_coh = relevant[(relevant["new_mar"] == 1) & (relevant["person_sex"] == "male")]    .sort_index(axis=0).sample(min_cohab)
    
    # print("Printing family sizes:")
    # print(female_mar.shape[0])
    # print(male_mar.shape[0])
    # print(female_coh.shape[0])
    # print(male_coh.shape[0])
    female_mar = female_mar.sort_values("age")
    male_mar = male_mar.sort_values("age")
    female_coh = female_coh.sort_values("age")
    male_coh = male_coh.sort_values("age")
    female_mar["number"] = np.arange(female_mar.shape[0])
    male_mar["number"] = np.arange(male_mar.shape[0])
    female_coh["number"] = np.arange(female_coh.shape[0])
    male_coh["number"] = np.arange(male_coh.shape[0])
    married = pd.concat([male_mar, female_mar])
    cohabitate = pd.concat([male_coh, female_coh])

    married["rnd"] = np.random.random(len(married))
    cohabitate["rnd"] = np.random.random(len(cohabitate))

    married = married.sort_values(by=["number"])
    cohabitate = cohabitate.sort_values(by=["number"])

    married["household_group"] = np.repeat(np.arange(len(married.index) / 2), 2)
    cohabitate["household_group"] = np.repeat(np.arange(len(cohabitate.index) / 2), 2)

    married = married.sort_values(by=["household_group", "earning", "rnd"], ascending=[True, False, True])
    cohabitate = cohabitate.sort_values(by=["household_group", "earning", "rnd"], ascending=[True, False, True])

    cohabitate["household_group"] = (cohabitate["household_group"] + married["household_group"].max() + 1)

    married["new_relate"] = relate(married.shape[0])
    cohabitate["new_relate"] = relate(cohabitate.shape[0], False)
    final = pd.concat([married, cohabitate])

    final["first"] = first(final.shape[0])
    final["partner"] = swap(final.index)
    final["partner_house"] = swap(final["household_id"])
    final["partner_relate"] = swap(final["relate"])

    final["new_household_id"] = -99
    final["stay"] = -99
    final = final[~(final["household_id"] == final["partner_house"])].copy()

    # print("Pair people.")
    # Pair up the people and classify what type of marriage it is
    # TODO speed up this code by a lot
    # relevant.sort_values("new_mar", inplace=True)
    
    
    
    # Stay documentation
    # 0 - leaves household
    # 1 - stays
    # 2 - this persons household becomes a root household (a root household absorbs the leaf household)
    # 4 - this persons household becomes a leaf household
    # 3 - this person leaves their household and creates a new household with partner
    # Marriage
    CONDITION_1 = ((final["first"] == 1) & (final["relate"] == 0) & (final["partner_relate"] == 0))
    final.loc[final[CONDITION_1].index, "stay"] = 1
    final.loc[final[CONDITION_1]["partner"].values, "stay"] = 0

    CONDITION_2 = ((final["first"] == 1) & (final["relate"] == 0) & (final["partner_relate"] != 0))
    final.loc[final[CONDITION_2].index, "stay"] = 1
    final.loc[final[CONDITION_2]["partner"].values, "stay"] = 0

    CONDITION_3 = ((final["first"] == 1) & (final["relate"] != 0) & (final["partner_relate"] == 0))
    final.loc[final[CONDITION_3].index, "stay"] = 0
    final.loc[final[CONDITION_3]["partner"].values, "stay"] = 1

    CONDITION_4 = ((final["first"] == 1) & (final["relate"] != 0) & (final["partner_relate"] != 0))
    final.loc[final[CONDITION_4].index, "stay"] = 3
    final.loc[final[CONDITION_4]["partner"].values, "stay"] = 3

    new_household_ids = np.arange(final[CONDITION_4].index.shape[0])
    new_household_ids_max = new_household_ids.max() + 1
    final.loc[final[CONDITION_4].index, "new_household_id"] = new_household_ids
    final.loc[final[CONDITION_4]["partner"].values, "new_household_id"] = new_household_ids

    # print('Finished Pairing')
    # print("Updating households and persons table")
    # print(final.household_id.unique().shape[0])
    metadata = orca.get_table("metadata").to_frame()
    max_hh_id = metadata.loc["max_hh_id", "value"]
    current_max_id = max(max_hh_id, household_df.index.max())
    final["hh_new_id"] = np.where(final["stay"].isin([1]), final["household_id"], np.where(final["stay"].isin([0]),final["partner_house"],final["new_household_id"] + current_max_id + 1))

    # final["new_relate"] = relate(final.shape[0])
    ## NEED TO SEPARATE MARRIED FROM COHABITATE

    # Households where everyone left
    # household_matched = (p_df[p_df["household_id"].isin(final["household_id"].unique())].groupby("household_id").size() == final.groupby("household_id").size())
    # removed_hh_values = household_matched[household_matched==True].index.values

    # Households where head left
    household_ids_reorganized = final[(final["stay"] == 0) & (final["relate"] == 0)]["household_id"].unique()

    p_df.loc[final.index, "household_id"] = final["hh_new_id"]
    p_df.loc[final.index, "relate"] = final["new_relate"]
    # print("HH SHAPE 1:", p_df["household_id"].unique().shape[0])

    households_restructuring = p_df.loc[p_df["household_id"].isin(household_ids_reorganized)]

    households_restructuring = households_restructuring.sort_values(by=["household_id", "earning"], ascending=False)
    p_df.loc[households_restructuring.groupby(["household_id"]).head(1).index, "relate"] = 0

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
    household_agg = p_df.groupby("household_id").agg(income=("earning", "sum"),race_of_head=("race_id", "first"),age_of_head=("age", "first"),size=("person", "sum"),workers=("worker", "sum"),hispanic_head=("hispanic_head", "sum"),persons_age_gt55=("age_gt55", "sum"),seniors=("senior", "sum"),children=("child", "sum"),persons=("person", "sum"),)

    # household_agg["lcm_county_id"] = household_agg["lcm_county_id"]
    household_agg["gt55"] = np.where(household_agg["persons_age_gt55"] > 0, 1, 0)
    household_agg["gt2"] = np.where(household_agg["persons"] > 2, 1, 0)
    household_agg["hh_workers"] = np.where(household_agg["workers"] == 0,"none",np.where(household_agg["workers"] == 1, "one", "two or more"),)
    household_agg["hh_age_of_head"] = np.where(household_agg["age_of_head"] < 35,"lt35",np.where(household_agg["age_of_head"] < 65, "gt35-lt65", "gt65"),)
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

    # household_df = household_df.drop_duplicates(subset="household_id")

    # household_df = household_df.set_index("household_id")
    # household_df.update(agg_households)
    household_df.update(household_agg)
    # household_df.loc[household_agg.index, persons_local_cols] = household_agg.loc[household_agg, persons_local_cols].to_numpy()

    final["MAR"] = np.where(final["new_mar"] == 2, 1, final["MAR"])
    # p_df.update(final["MAR"])
    p_df["NEW_MAR"] = final["MAR"]
    p_df["MAR"] = np.where(p_df["NEW_MAR"].isna(),p_df["MAR"], p_df["NEW_MAR"])

    new_hh = household_agg.loc[~household_agg.index.isin(household_df.index.unique())].copy()
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
    # breakpoint()

    # print("HH Size from Persons: ", p_df["household_id"].unique().shape[0])
    # print("HH Size from Household: ", household_df.index.unique().shape[0])
    # print("HH in HH_DF not in P_DF:", len(sorted(set(household_df.index.unique()) - set(p_df["household_id"].unique()))))
    # print("HH in P_DF not in HH_DF:", len(sorted(set(p_df["household_id"].unique()) - set(household_df.index.unique()))))
    # print("HHs with NA persons:", household_df["persons"].isna().sum())
    # print("HH duplicates: ", household_df.index.has_duplicates)
    # # print("Counties: ", households["lcm_county_id"].unique())
    # print("Persons Size: ", p_df.index.unique().shape[0])
    # print("Persons Duplicated: ", p_df.index.has_duplicates)

    # if len(sorted(set(household_df.index.unique()) - set(p_df["household_id"].unique()))) > 0:
    #     breakpoint()
    # if len(sorted(set(p_df["household_id"].unique()) - set(household_df.index.unique()))) > 0:
    #     breakpoint()

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
        new_married_table = pd.DataFrame(
                [[(marriage_list == 1).sum(), (marriage_list == 2).sum()]],
                columns=["married", "cohabitated"]
            )
        married_table = pd.concat([married_table, new_married_table],
                                  ignore_index=True)

    orca.add_table("marriage_table", married_table)


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


def update_divorce(divorce_list):
    """
    Updating stats for divorced households

    YE: ***

    Args:
        persons (DataFrameWrapper): DataFrameWrapper of the persons table
        households (DataFrameWrapper): DataFrameWrapper of the households table
        divorce_list (pd.Series): pandas Series of the divorced households

    Returns:
        None
    """
    # print("Updating household stats...")
    # breakpoint()
    households_local_cols = orca.get_table("households").local.columns

    persons_local_cols = orca.get_table("persons").local.columns

    households_df = orca.get_table("households").local

    persons_df = orca.get_table("persons").local

    # households_df.loc[divorce_list.index,"divorced"] = divorce_list

    # divorce_households = households_df[households_df["divorced"] == 1].copy()
    # DIVORCED_HOUSEHOLDS_ID = divorce_households.index.to_list()

    sizes = persons_df[persons_df["household_id"].isin(divorce_list.index) & (persons_df["relate"].isin([0, 1]))].groupby("household_id").size()

    # print("Sizes not 2: ", sizes[sizes!=2].shape[0])

    # print("divorced households: ", len(DIVORCED_HOUSEHOLDS_ID))
    # print("")
    persons_divorce = persons_df[
        persons_df["household_id"].isin(divorce_list[divorce_list.astype(bool)].index)
    ].copy().sort_index()
    # print("divorced persons: ", persons_divorce.shape[0])
    # print("Min hh size:", persons_divorce.groupby("household_id").size().min())
    # print("Max hh size:", persons_divorce.groupby("household_id").size().max())

    divorced_parents = persons_divorce[
        (persons_divorce["relate"].isin([0, 1])) & (persons_divorce["MAR"] == 1)
    ].copy()

    # print("Min parents size:", divorced_parents.groupby("household_id").size().min())
    # print("Max parents size:", divorced_parents.groupby("household_id").size().max())
    leaving_house = divorced_parents.groupby("household_id").sample(n=1)

    staying_house = persons_divorce[~(persons_divorce.index.isin(leaving_house.index))].copy()

    metadata = orca.get_table("metadata").to_frame()
    max_hh_id = metadata.loc["max_hh_id", "value"]
    # give the people leaving a new household id, update their marriage status, and other variables
    leaving_house["relate"] = 0
    leaving_house["MAR"] = 3
    leaving_house["member_id"] = 1
    leaving_house["household_id"] = (
        np.arange(leaving_house.shape[0]) + max_hh_id + 1
    )

    # modify necessary variables for members staying in household
    staying_house["relate"] = np.where(
        staying_house["relate"].isin([1, 0]), 0, staying_house["relate"]
    )
    staying_house["member_id"] = np.where(
        staying_house["member_id"] != 1,
        staying_house["member_id"] - 1,
        staying_house["relate"],
    )
    staying_house["MAR"] = np.where(
        (staying_house["MAR"] == 1) & (staying_house["relate"].isin([0, 1])), 3, staying_house["MAR"]
    )

    # initiate new households with individuals leaving house
    # TODO: DISCUSS ALL THESE INITIALIZATION MEASURES
    staying_households = staying_house.copy()
    staying_households["person"] = 1
    staying_households["is_head"] = np.where(staying_households["relate"] == 0, 1, 0)
    staying_households["race_head"] = (
        staying_households["is_head"] * staying_households["race_id"]
    )
    staying_households["age_head"] = (
        staying_households["is_head"] * staying_households["age"]
    )
    staying_households["hispanic_head"] = (
        staying_households["is_head"] * staying_households["hispanic"]
    )
    staying_households["child"] = np.where(
        staying_households["relate"].isin([2, 3, 4, 14]), 1, 0
    )
    staying_households["senior"] = np.where(staying_households["age"] >= 65, 1, 0)
    staying_households["age_gt55"] = np.where(staying_households["age"] >= 55, 1, 0)

    staying_households = staying_households.sort_values(by=["household_id", "relate"])
    staying_household_agg = staying_households.groupby("household_id").agg(
        income=("earning", "sum"),
        race_of_head=("race_id", "first"),
        age_of_head=("age", "first"),
        size=("person", "sum"),
        workers=("worker", "sum"),
        hispanic_head=("hispanic_head", "sum"),
        persons_age_gt55=("age_gt55", "sum"),
        seniors=("senior", "sum"),
        children=("child", "sum"),
        persons=("person", "sum"),
    )

    # household_agg["lcm_county_id"] = household_agg["lcm_county_id"]
    staying_household_agg["gt55"] = np.where(
        staying_household_agg["persons_age_gt55"] > 0, 1, 0
    )
    staying_household_agg["gt2"] = np.where(staying_household_agg["persons"] > 2, 1, 0)
    # staying_household_agg["sf_detached"] = "unknown"
    # staying_household_agg["serialno"] = "unknown"
    # staying_household_agg["cars"] = households_df[households_df.index.isin(staying_house["household_id"].unique())]["cars"]

    staying_household_agg["hh_workers"] = np.where(
        staying_household_agg["workers"] == 0,
        "none",
        np.where(staying_household_agg["workers"] == 1, "one", "two or more"),
    )
    staying_household_agg["hh_age_of_head"] = np.where(
        staying_household_agg["age_of_head"] < 35,
        "lt35",
        np.where(staying_household_agg["age_of_head"] < 65, "gt35-lt65", "gt65"),
    )
    staying_household_agg["hh_race_of_head"] = np.where(
        staying_household_agg["race_of_head"] == 1,
        "white",
        np.where(
            staying_household_agg["race_of_head"] == 2,
            "black",
            np.where(
                staying_household_agg["race_of_head"].isin([6, 7]), "asian", "other"
            ),
        ),
    )
    staying_household_agg["hispanic_head"] = np.where(
        staying_household_agg["hispanic_head"] == 1, "yes", "no"
    )
    staying_household_agg["hh_size"] = np.where(
        staying_household_agg["size"] == 1,
        "one",
        np.where(
            staying_household_agg["size"] == 2,
            "two",
            np.where(staying_household_agg["size"] == 3, "three", "four or more"),
        ),
    )
    staying_household_agg["hh_children"] = np.where(
        staying_household_agg["children"] >= 1, "yes", "no"
    )
    staying_household_agg["hh_seniors"] = np.where(
        staying_household_agg["seniors"] >= 1, "yes", "no"
    )
    staying_household_agg["hh_income"] = np.where(
        staying_household_agg["income"] < 30000,
        "lt30",
        np.where(
            staying_household_agg["income"] < 60,
            "gt30-lt60",
            np.where(
                staying_household_agg["income"] < 100,
                "gt60-lt100",
                np.where(staying_household_agg["income"] < 150, "gt100-lt150", "gt150"),
            ),
        ),
    )

    # staying_household_agg["hh_type"] = 1
    # staying_household_agg["household_type"] = 1
    # staying_household_agg["serialno"] = -1
    # staying_household_agg["birth"] = -99
    # staying_household_agg["divorced"] = -99
    # staying_household_agg.set_index(staying_household_agg["household_id"], inplace=True)
    staying_household_agg.index.name = "household_id"

    # initiate new households with individuals leaving house
    # TODO: DISCUSS ALL THESE INITIALIZATION MEASURES
    new_households = leaving_house.copy()
    new_households["person"] = 1
    new_households["is_head"] = np.where(new_households["relate"] == 0, 1, 0)
    new_households["race_head"] = new_households["is_head"] * new_households["race_id"]
    new_households["age_head"] = new_households["is_head"] * new_households["age"]
    new_households["hispanic_head"] = (
        new_households["is_head"] * new_households["hispanic"]
    )
    new_households["child"] = np.where(
        new_households["relate"].isin([2, 3, 4, 14]), 1, 0
    )
    new_households["senior"] = np.where(new_households["age"] >= 65, 1, 0)
    new_households["age_gt55"] = np.where(new_households["age"] >= 55, 1, 0)

    new_households = new_households.sort_values(by=["household_id", "relate"])
    household_agg = new_households.groupby("household_id").agg(
        income=("earning", "sum"),
        race_of_head=("race_head", "sum"),
        age_of_head=("age_head", "sum"),
        size=("person", "sum"),
        workers=("worker", "sum"),
        hispanic_head=("hispanic_head", "sum"),
        persons_age_gt55=("age_gt55", "sum"),
        seniors=("senior", "sum"),
        children=("child", "sum"),
        persons=("person", "sum"),
    )

    # household_agg["lcm_county_id"] = household_agg["lcm_county_id"]
    household_agg["gt55"] = np.where(household_agg["persons_age_gt55"] > 0, 1, 0)
    household_agg["gt2"] = np.where(household_agg["persons"] > 2, 1, 0)
    household_agg["sf_detached"] = "unknown"
    household_agg["serialno"] = "unknown"
    household_agg["tenure"] = "unknown"
    household_agg["tenure_mover"] = "unknown"
    household_agg["recent_mover"] = "unknown"
    household_agg["cars"] = np.random.choice([0, 1], size=household_agg.shape[0])

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
    household_agg["hispanic_status_of_head"] = np.where(
        household_agg["hispanic_head"] == "yes", 1, 0
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

    household_agg["hh_cars"] = np.where(
        household_agg["cars"] == 0,
        "none",
        np.where(household_agg["cars"] == 1, "one", "two or more"),
    )
    household_agg["block_id"] = "-1"
    household_agg["lcm_county_id"] = "-1"
    household_agg["hh_type"] = 1
    household_agg["household_type"] = 1
    household_agg["serialno"] = "-1"
    household_agg["birth"] = -99
    household_agg["divorced"] = -99
    # household_agg.set_index(household_agg["household_id"], inplace=True)
    # household_agg.index.name = "household_id"

    households_df.update(staying_household_agg)

    # print(staying_house["household_id"].unique().shape[0] + leaving_house["household_id"].unique().shape[0])
    hh_ids_p_table = np.hstack((staying_house["household_id"].unique(), leaving_house["household_id"].unique()))
    df_p = persons_df.combine_first(staying_house[persons_local_cols])
    df_p = df_p.combine_first(leaving_house[persons_local_cols])
    hh_ids_hh_table = np.hstack((households_df.index, household_agg.index))
    # if  df_p["household_id"].unique().shape[0] != np.unique(hh_ids_hh_table).shape[0]:
    #     breakpoint()
    # merge all in one persons and households table
    new_households = pd.concat([households_df[households_local_cols], household_agg[households_local_cols]])
    persons_df.update(staying_house[persons_local_cols])
    persons_df.update(leaving_house[persons_local_cols])

    # persons_df
    # persons_df.loc[staying_house.index, persons_local_cols] = staying_house.loc[staying_house.index, persons_local_cols].to_numpy()
    # persons_df.loc[leaving_house.index, persons_local_cols] = leaving_house.loc[leaving_house.index, persons_local_cols].to_numpy()

    # if  persons_df["household_id"].unique().shape[0] != new_households.index.unique().shape[0]:
    #     breakpoint()
    orca.add_table("households", new_households[households_local_cols])
    orca.add_table("persons", persons_df[persons_local_cols])
    # orca.add_injectable(
    #     "max_hh_id", max(orca.get_injectable("max_hh_id"), new_households.index.max())
    # )
    
    metadata = orca.get_table("metadata").to_frame()
    max_hh_id = metadata.loc["max_hh_id", "value"]
    max_p_id = metadata.loc["max_p_id", "value"]
    if new_households.index.max() > max_hh_id:
        metadata.loc["max_hh_id", "value"] = new_households.index.max()
    if persons_df.index.max() > max_p_id:
        metadata.loc["max_p_id", "value"] = persons_df.index.max()
    orca.add_table("metadata", metadata)

    # print("Updating divorce metrics...")
    divorce_table = orca.get_table("divorce_table").to_frame()
    if divorce_table.empty:
        divorce_table = pd.DataFrame([divorce_list.sum()], columns=["divorced"])
    else:
        new_divorce = pd.DataFrame([divorce_list.sum()], columns=["divorced"])
        divorce_table = pd.concat([divorce_table, new_divorce], ignore_index=True)
    orca.add_table("divorce_table", divorce_table)