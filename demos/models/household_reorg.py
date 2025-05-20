import orca
import numpy as np
import pandas as pd
from templates import estimated_models, modelmanager as mm
from .marriage import update_married_households_random, update_married_households, update_divorce

@orca.step("households_reorg")
def households_reorg(persons, households, year):
    """ YE: *** """
    #
    marriage = mm.get_step("marriage")
    # MARRIAGE MODEL
    persons_df = persons.to_frame(marriage.variable_names + ["relate", "household_id", "MAR", "age"])
    # get persons cohabitating and heads of their households
    COHABS_PERSONS = persons_df["relate"] == 13
    cohab_persons_df = persons_df.loc[COHABS_PERSONS].copy()
    COHABS_HOUSEHOLDS = cohab_persons_df["household_id"].unique()
    COHABS_HEADS = (persons_df["household_id"].isin(COHABS_HOUSEHOLDS)) & (persons_df["relate"] == 0)
    cohab_heads_df = persons_df.loc[COHABS_HEADS].copy()
    all_cohabs_df = pd.concat([cohab_heads_df, cohab_persons_df])
    all_cohabs_df["cohab"] = 1
    # Get Single People
    SINGLE_COND = (persons_df["MAR"] != 1) & (persons_df["age"] >= 15)
    single_df = persons_df.loc[SINGLE_COND].copy()
    data = single_df.join(all_cohabs_df[["cohab"]], how="left")
    data = data.loc[data["cohab"] != 1].copy()
    data.drop(columns="cohab", inplace=True)
    ###############################################################
    print("Running marriage model...")
    # breakpoint()
    marriage_list = marriage.run(data.sort_index(axis=0).copy())
    # print("Number of marriages and cohabitations:")
    # print(marriage_list.value_counts())
    random_match = orca.get_injectable("random_match")
    ## ------------------------------------
    
    # DIVORCE MODEL
    households_df = orca.get_table("households").local
    households_df["divorced"] = -99
    orca.add_table("households", households_df)
    persons_df = orca.get_table("persons").local
    ELIGIBLE_HOUSEHOLDS = list(persons_df[(persons_df["relate"].isin([0, 1])) & (persons_df["MAR"] == 1)]["household_id"].unique().astype(int))
    sizes = (persons_df[persons_df["household_id"].isin(ELIGIBLE_HOUSEHOLDS)& (persons_df["relate"].isin([0, 1]))].groupby("household_id").size())
    # print("Sizes value counts: ", sizes.value_counts())
    ELIGIBLE_HOUSEHOLDS = sizes[(sizes == 2)].index.to_list()
    # print("Size of Eligible Households: ", len(ELIGIBLE_HOUSEHOLDS))
    # print("Eligible households for divorce are", len(ELIGIBLE_HOUSEHOLDS))
    divorce_model = mm.get_step("divorce")
    list_ids = str(ELIGIBLE_HOUSEHOLDS)
    divorce_model.filters = "index in " + list_ids
    divorce_model.out_filters = "index in " + list_ids

    print("Running divorce model...")
    divorce_model.run()
    divorce_list = divorce_model.choices.astype(int)
    # if divorce_list.shape[0] !=  len(ELIGIBLE_HOUSEHOLDS):
    #     breakpoint()
    # print("Number of divorces:")
    # print(divorce_list.value_counts())
    # # breakpoint()
    # predicted_num = (2*divorce_list.sum() + (persons_df[persons_df["age"]>=15]["MAR"]==3).sum())
    # predicted_share = predicted_num / persons_df.shape[0]

    # observed_marrital = orca.get_table("observed_marrital_data").to_frame()
    # target = observed_marrital[(observed_marrital["year"]==year) & (observed_marrital["MAR"]==3)]["count"]
    # target_share = target.sum() / persons_df.shape[0]

    # error = np.sqrt(np.mean((predicted_share - target_share)**2))
    # print(error)
    # # print("here")
    # while error >= 0.02:
    #     # print("here")
    #     divorce_model.fitted_parameters[0] += np.log(target.sum()/predicted_num)
    #     # breakpoint()
    #     divorce_model.run()
    #     divorce_list = divorce_model.choices.astype(int)
    #     # print(fatality_list.sum())
    #     predicted_num = (2*divorce_list.sum() + (persons_df[persons_df["age"]>=15]["MAR"]==3).sum())
    #     # print(predicted_num.sum())
    #     predicted_share = predicted_num / persons_df.shape[0]
    #     error = np.sqrt(np.mean((predicted_share - target_share)**2))
    #     print(error)
        
    #########################################
    
    # COHABITATION_TO_X Model
    hh_df = households.to_frame(columns=["lcm_county_id"])
    hh_df.reset_index(inplace=True)

    persons_df = persons.local
    ELIGIBLE_HOUSEHOLDS = (
        persons_df[(persons_df["relate"] == 13) & \
                   (persons_df["MAR"]!=1) & \
                   ((persons_df["age"]>=15))]["household_id"].unique().astype(int)
    )
    cohabitation = mm.get_step("cohabitation")
    data = households.to_frame(cohabitation.variable_names).loc[ELIGIBLE_HOUSEHOLDS]
    # Run Model
    print("Running cohabitation model...")
    cohabitate_x_list = cohabitation.run(data)
    # print("Cohabitation outcomes:")
    # print(cohabitate_x_list.value_counts())

    np.random.seed(year)
    
    ######### UPDATING
    print("Restructuring households:")
    print("Cohabitations..")
    update_cohabitating_households(persons, households, cohabitate_x_list)
    print_household_stats()
    
    print("Marriages..")
    update_married_households_random(persons, households, marriage_list)
    print_household_stats()
    fix_erroneous_households(persons, households)
    print_household_stats()
    
    print("Divorces..")
    update_divorce(divorce_list)
    print_household_stats()
    
    marrital = orca.get_table("marrital").to_frame()
    persons_df = orca.get_table("persons").local
    persons_local_columns = orca.get_injectable("persons_local_cols")
    persons_df["member_id"] = persons_df.groupby("household_id")["relate"].rank(method="first", ascending=True).astype(int)
    orca.add_table("persons", persons_df[persons_local_columns])
    if marrital.empty:
        persons_stats = persons_df[persons_df["age"]>=15]["MAR"].value_counts().reset_index()
        marrital = pd.DataFrame(persons_stats)
        marrital["year"] = year
    else:
        persons_stats = persons_df[persons_df["age"]>=15]["MAR"].value_counts().reset_index()
        new_marrital = pd.DataFrame(persons_stats)
        new_marrital["year"] = year
        marrital = pd.concat([marrital, new_marrital])
    orca.add_table("marrital", marrital)


@orca.step("print_household_stats")
def print_household_stats():
    """Function to print the number of households from both the households and pers

    Args:
        persons (DataFrame): Pandas DataFrame of the persons table
        households (DataFrame): Pandas DataFrame of the households table
    """
    print("Households size from persons table: ", orca.get_table("persons").local["household_id"].unique().shape[0])
    print("Households size from households table: ", orca.get_table("households").local.index.unique().shape[0])
    print("Persons Size: ", orca.get_table("persons").local.index.unique().shape[0])
    print("Missing hh:", len(set(orca.get_table("persons").local["household_id"].unique()) -\
        set(orca.get_table("households").local.index.unique())))
    persons_df = orca.get_table("persons").local
    persons_df["relate_0"] = np.where(persons_df["relate"]==0, 1, 0)
    persons_df["relate_1"] = np.where(persons_df["relate"]==1, 1, 0)
    persons_df["relate_13"] = np.where(persons_df["relate"]==13, 1, 0)
    persons_df_sum = persons_df.groupby("household_id").agg(relate_1 = ("relate_1", sum), relate_13 = ("relate_13", sum),
    relate_0 = ("relate_0", sum))
    print("Households with multiple 0:", ((persons_df_sum["relate_0"])>1).sum())
    print("Households with multiple 1:", ((persons_df_sum["relate_1"])>1).sum())
    print("Households with multiple 13:", ((persons_df_sum["relate_13"])>1).sum())
    print("Households with 1 and 13:", ((persons_df_sum["relate_1"] * persons_df_sum["relate_13"])>0).sum())

@orca.step("household_stats")
def household_stats(persons, households):
    """Function to print the number of households from both the households and pers

    Args:
        persons (DataFrame): Pandas DataFrame of the persons table
        households (DataFrame): Pandas DataFrame of the households table
    """
    print("Households size from persons table: ", orca.get_table("persons").local["household_id"].unique().shape[0])
    print("Households size from households table: ", orca.get_table("households").local.index.unique().shape[0])
    print("Households in households table not in persons table:", len(sorted(set(orca.get_table("households").local.index.unique()) - set(orca.get_table("persons").local["household_id"].unique()))))
    print("Households in persons table not in households table:", len(sorted(set(orca.get_table("persons").local["household_id"].unique()) - set(orca.get_table("households").local.index.unique()))))
    print("Households with NA persons:", orca.get_table("households").local["persons"].isna().sum())
    print("Duplicated households: ", orca.get_table("households").local.index.has_duplicates)
    # print("Counties: ", households["lcm_county_id"].unique())
    print("Persons Size: ", orca.get_table("persons").local.index.unique().shape[0])
    print("Duplicated persons: ", orca.get_table("persons").local.index.has_duplicates)

    persons_df = orca.get_table("persons").local
    persons_df["relate_0"] = np.where(persons_df["relate"]==0, 1, 0)
    persons_df["relate_1"] = np.where(persons_df["relate"]==1, 1, 0)
    persons_df["relate_13"] = np.where(persons_df["relate"]==13, 1, 0)
    persons_df_sum = persons_df.groupby("household_id").agg(relate_1 = ("relate_1", sum), relate_13 = ("relate_13", sum),
    relate_0 = ("relate_0", sum))
    print("Households with multiple 0: ", ((persons_df_sum["relate_0"])>1).sum())
    print("Households with multiple 1: ", ((persons_df_sum["relate_1"])>1).sum())
    print("Households with multiple 13: ", ((persons_df_sum["relate_13"])>1).sum())
    print("Households with 1 and 13: ", ((persons_df_sum["relate_1"] * persons_df_sum["relate_13"])>0).sum())



def update_cohabitating_households(persons, households, cohabitate_list):
    """
    Updating households and persons after cohabitation model.

    YE: ***

    Args:
        persons (DataFrameWrapper): DataFrameWrapper of persons table
        households (DataFrameWrapper): DataFrameWrapper of households table
        cohabitate_list (pd.Series): Pandas Series of cohabitation model output

    Returns:
        None
    """
    persons_df = orca.get_table("persons").local
    persons_local_cols = persons_df.columns
    households_df = orca.get_table("households").local
    hh_df = households.to_frame(columns=["lcm_county_id"])
    households_local_cols = households_df.columns
    married_hh = cohabitate_list.index[cohabitate_list == 2].to_list()
    breakup_hh = cohabitate_list.index[cohabitate_list == 1].to_list()

    persons_df.loc[(persons_df["household_id"].isin(married_hh)) & (persons_df["relate"] == 13),"relate",] = 1
    persons_df.loc[(persons_df["household_id"].isin(married_hh)) & (persons_df["relate"].isin([1, 0])),"MAR"] = 1

    persons_df = (persons_df.reset_index().merge(hh_df, on=["household_id"]).set_index("person_id"))

    leaving_person_index = persons_df.index[(persons_df["household_id"].isin(breakup_hh)) & (persons_df["relate"] == 13)]

    leaving_house = persons_df.loc[leaving_person_index].copy()

    leaving_house["relate"] = 0

    persons_df = persons_df.drop(leaving_person_index)

    # Update characteristics for households staying
    persons_df["person"] = 1
    persons_df["is_head"] = np.where(persons_df["relate"] == 0, 1, 0)
    persons_df["race_head"] = persons_df["is_head"] * persons_df["race_id"]
    persons_df["age_head"] = persons_df["is_head"] * persons_df["age"]
    persons_df["hispanic_head"] = persons_df["is_head"] * persons_df["hispanic"]
    persons_df["child"] = np.where(persons_df["relate"].isin([2, 3, 4, 14]), 1, 0)
    persons_df["senior"] = np.where(persons_df["age"] >= 65, 1, 0)
    persons_df["age_gt55"] = np.where(persons_df["age"] >= 55, 1, 0)

    households_new = persons_df.groupby("household_id").agg(income=("earning", "sum"),race_of_head=("race_head", "sum"),age_of_head=("age_head", "sum"), workers=("worker", "sum"),hispanic_status_of_head=("hispanic", "sum"),persons=("person", "sum"),children=("child", "sum"),seniors=("senior", "sum"),gt55=("age_gt55", "sum"),
    )

    households_new["hh_age_of_head"] = np.where(households_new["age_of_head"] < 35,"lt35",np.where(households_new["age_of_head"] < 65, "gt35-lt65", "gt65"),)
    households_new["hispanic_head"] = np.where( households_new["hispanic_status_of_head"] == 1, "yes", "no")
    households_new["hh_children"] = np.where( households_new["children"] >= 1, "yes", "no")
    households_new["hh_seniors"] = np.where(households_new["seniors"] >= 1, "yes", "no")
    households_new["gt2"] = np.where(households_new["persons"] >= 2, 1, 0)
    households_new["gt55"] = np.where(households_new["gt55"] >= 1, 1, 0)
    households_new["hh_income"] = np.where(households_new["income"] < 30000,"lt30",np.where(households_new["income"] < 60,
            "gt30-lt60",
            np.where(
                households_new["income"] < 100,
                "gt60-lt100",
                np.where(households_new["income"] < 150, "gt100-lt150", "gt150"),
            ),
        ),
    )
    households_new["hh_workers"] = np.where(
        households_new["workers"] == 0,
        "none",
        np.where(households_new["workers"] == 1, "one", "two or more"),
    )

    households_new["hh_race_of_head"] = np.where(
        households_new["race_of_head"] == 1,
        "white",
        np.where(
            households_new["race_of_head"] == 2,
            "black",
            np.where(households_new["race_of_head"].isin([6, 7]), "asian", "other"),
        ),
    )

    households_new["hh_size"] = np.where(
        households_new["persons"] == 1,
        "one",
        np.where(
            households_new["persons"] == 2,
            "two",
            np.where(households_new["persons"] == 3, "three", "four or more"),
        ),
    )

    households_df.update(households_new)

    metadata = orca.get_table("metadata").to_frame()
    max_hh_id = metadata.loc["max_hh_id", "value"]
    # Create household characteristics for new households formed
    leaving_house["household_id"] = (
        np.arange(len(breakup_hh))
        + max(max_hh_id, households_df.index.max())
        + 1
    )
    leaving_house["person"] = 1
    leaving_house["is_head"] = np.where(leaving_house["relate"] == 0, 1, 0)
    leaving_house["race_head"] = leaving_house["is_head"] * leaving_house["race_id"]
    leaving_house["age_head"] = leaving_house["is_head"] * leaving_house["age"]
    leaving_house["hispanic_head"] = (
        leaving_house["is_head"] * leaving_house["hispanic"]
    )
    leaving_house["child"] = np.where(leaving_house["relate"].isin([2, 3, 4, 14]), 1, 0)
    leaving_house["senior"] = np.where(leaving_house["age"] >= 65, 1, 0)
    leaving_house["age_gt55"] = np.where(leaving_house["age"] >= 55, 1, 0)

    households_new = leaving_house.groupby("household_id").agg(
        income=("earning", "sum"),
        race_of_head=("race_head", "sum"),
        age_of_head=("age_head", "sum"),
        workers=("worker", "sum"),
        hispanic_status_of_head=("hispanic", "sum"),
        persons=("person", "sum"),
        children=("child", "sum"),
        seniors=("senior", "sum"),
        gt55=("age_gt55", "sum"),
        lcm_county_id=("lcm_county_id", "first"),
    )

    households_new["hh_age_of_head"] = np.where(
        households_new["age_of_head"] < 35,
        "lt35",
        np.where(households_new["age_of_head"] < 65, "gt35-lt65", "gt65"),
    )
    households_new["hispanic_head"] = np.where(
        households_new["hispanic_status_of_head"] == 1, "yes", "no"
    )
    households_new["hh_children"] = np.where(
        households_new["children"] >= 1, "yes", "no"
    )
    households_new["hh_seniors"] = np.where(households_new["seniors"] >= 1, "yes", "no")
    households_new["gt2"] = np.where(households_new["persons"] >= 2, 1, 0)
    households_new["gt55"] = np.where(households_new["gt55"] >= 1, 1, 0)
    households_new["hh_income"] = np.where(
        households_new["income"] < 30000,
        "lt30",
        np.where(
            households_new["income"] < 60,
            "gt30-lt60",
            np.where(
                households_new["income"] < 100,
                "gt60-lt100",
                np.where(households_new["income"] < 150, "gt100-lt150", "gt150"),
            ),
        ),
    )
    households_new["hh_workers"] = np.where(
        households_new["workers"] == 0,
        "none",
        np.where(households_new["workers"] == 1, "one", "two or more"),
    )

    households_new["hh_race_of_head"] = np.where(
        households_new["race_of_head"] == 1,
        "white",
        np.where(
            households_new["race_of_head"] == 2,
            "black",
            np.where(households_new["race_of_head"].isin([6, 7]), "asian", "other"),
        ),
    )

    households_new["hh_size"] = np.where(
        households_new["persons"] == 1,
        "one",
        np.where(
            households_new["persons"] == 2,
            "two",
            np.where(households_new["persons"] == 3, "three", "four or more"),
        ),
    )

    households_new["cars"] = np.random.choice([0, 1], size=households_new.shape[0])
    households_new["hh_cars"] = np.where(
        households_new["cars"] == 0,
        "none",
        np.where(households_new["cars"] == 1, "one", "two or more"),
    )
    households_new["tenure"] = "unknown"
    households_new["recent_mover"] = "unknown"
    households_new["sf_detached"] = "unknown"
    households_new["tenure_mover"] = "unknown"
    households_new["block_id"] = "-1"
    households_new["hh_type"] = "-1"
    households_df = pd.concat([households_df, households_new])

    persons_df = pd.concat([persons_df, leaving_house])

    # print("HH Size from Persons: ", persons_df["household_id"].unique().shape[0])
    # print("HH Size from Household: ", households_df.index.unique().shape[0])
    # print("HH in HH_DF not in P_DF:", len(sorted(set(households_df.index.unique()) - set(persons_df["household_id"].unique()))))
    # print("HH in P_DF not in HH_DF:", len(sorted(set(persons_df["household_id"].unique()) - set(households_df.index.unique()))))
    # print("HHs with NA persons:", households_df["persons"].isna().sum())
    # print("HH duplicates: ", households_df.index.has_duplicates)
    # # print("Counties: ", households["lcm_county_id"].unique())
    # print("Persons Size: ", persons_df.index.unique().shape[0])
    # print("Persons Duplicated: ", persons_df.index.has_duplicates)

    # if len(sorted(set(households_df.index.unique()) - set(persons_df["household_id"].unique()))) > 0:
    #     breakpoint()
    # if len(sorted(set(persons_df["household_id"].unique()) - set(households_df.index.unique()))) > 0:
    #     breakpoint()
    
    # add to orca
    orca.add_table("households", households_df[households_local_cols])
    orca.add_table("persons", persons_df[persons_local_cols])
    # orca.add_injectable(
    #     "max_hh_id", max(orca.get_injectable("max_hh_id"), households_df.index.max())
    # )
    metadata = orca.get_table("metadata").to_frame()
    max_hh_id = metadata.loc["max_hh_id", "value"]
    max_p_id = metadata.loc["max_p_id", "value"]
    if households_df.index.max() > max_hh_id:
        metadata.loc["max_hh_id", "value"] = households_df.index.max()
    if persons_df.index.max() > max_p_id:
        metadata.loc["max_p_id", "value"] = persons_df.index.max()
    orca.add_table("metadata", metadata)


def fix_erroneous_households(persons, households):
    """ YE: *** """
    print("Fixing erroneous households")
    p_df = orca.get_table('persons').local
    household_cols = orca.get_table('households').local_columns
    household_df = orca.get_table('households').local
    persons_cols = persons.local_columns
    # print("Hh size: ", household_df.shape)
    # print("Persons size: ", p_df.shape)
    households_to_drop = p_df[p_df['relate'].isin([1, 13])].groupby('household_id')['relate'].nunique().reset_index()
    households_to_drop = households_to_drop[households_to_drop["relate"]==2]["household_id"].to_list()
    # print("Num hh to be dropped: ", len(households_to_drop))
    household_df = household_df.drop(households_to_drop)
    p_df = p_df[~p_df["household_id"].isin(households_to_drop)]

    orca.add_table("households", household_df[household_cols])
    orca.add_table("persons", p_df[persons_cols])

    metadata = orca.get_table("metadata").to_frame()
    max_hh_id = metadata.loc["max_hh_id", "value"]
    max_p_id = metadata.loc["max_p_id", "value"]
    if household_df.index.max() > max_hh_id:
        metadata.loc["max_hh_id", "value"] = household_df.index.max()
    if p_df.index.max() > max_p_id:
        metadata.loc["max_p_id", "value"] = p_df.index.max()
    orca.add_table("metadata", metadata)