import time
import orca
import numpy as np
import pandas as pd
from templates import estimated_models, modelmanager as mm
from templates.utils.models import columns_in_formula
from .marriage import update_married_households_random, update_married_households, update_divorce

@orca.column(table_name="persons", cache=True, cache_scope="step")
def cohabitate(persons):
    unmarried_partner_index = persons["relate"] == 13
    cohab_household_ids = persons["household_id"].loc[unmarried_partner_index].unique()
    return unmarried_partner_index | \
            ((persons["relate"] == 0) & persons["household_id"].isin(cohab_household_ids))


@orca.column(table_name="persons", cache=True, cache_scope="step")
def is_single(persons):
    # TODO: Standarize the variable name MAR
    return (persons["MAR"] != 1) & (persons["age"] >= 15)


@orca.column(table_name="persons")
def is_head(persons):
    return (persons["relate"] == 0).astype(int)


@orca.column(table_name="persons")
def race_head(persons):
    return persons["is_head"] * persons["race_id"]


@orca.column(table_name="persons")
def age_head(persons):
    return persons["is_head"] * persons["age"]


@orca.column(table_name="persons")
def hispanic_head(persons):
    return persons["is_head"] * persons["hispanic"]

@orca.injectable(cache=True, cache_scope="step")
def persons_grouped_household(persons):
    return persons.groupby("household_id")


@orca.column(table_name="households")
def hh_age_of_head(persons_grouped_household):
    agg_df = persons_grouped_household\
        .agg(age_of_head=("age_head", "sum"))
    
    return np.where(agg_df["age_of_head"] < 35, "lt35",
                    np.where(agg_df["age_of_head"] < 65, "gt35-lt65",
                             "gt65"))


@orca.column(table_name="households")
def hh_age_of_head(persons_grouped_household):
    agg_df = persons_grouped_household\
        .agg(hispanic_status_of_head=("hispanic", "sum"))
    
    return (agg_df["hispanic_status_of_head"] == 1).replace({0: "no", 1: "yes"})


@orca.column(table_name="households")
def hh_children(persons_grouped_household):
    agg_df = persons_grouped_household\
        .agg(children=("child", "sum"))
    
    return (agg_df["children"] >= 1).replace({0: "no", 1: "yes"})


@orca.column(table_name="households")
def hh_seniors(persons_grouped_household):
    agg_df = persons_grouped_household\
        .agg(seniors=("senior", "sum"))
    
    return (agg_df["seniors"] >= 1).replace({0: "no", 1: "yes"})


@orca.column(table_name="households")
def gt2(persons_grouped_household):
    agg_df = persons_grouped_household.size()
    return (agg_df >= 2).astype(int)


@orca.column(table_name="households")
def gt55(persons_grouped_household):
    agg_df = persons_grouped_household\
        .agg(gt55=("age_gt55", "sum"))
    return (agg_df["age_gt55"] >= 1).astype(int)


@orca.column(table_name="households")
def hh_income(persons_grouped_household):
    agg_df = persons_grouped_household\
        .agg(income=("earning", "sum"))
    return np.where(agg_df["income"] < 30000,"lt30",
                    np.where(agg_df["income"] < 60, "gt30-lt60",
            np.where(
                agg_df["income"] < 100,
                "gt60-lt100",
                np.where(agg_df["income"] < 150, "gt100-lt150", "gt150"),
            ),
        ),
    )


@orca.column(table_name="households")
def hh_workers(persons_grouped_household):
    agg_df = persons_grouped_household\
        .agg(workers=("worker", "sum"))
    return np.where(agg_df["workers"] == 0, "none",
           np.where(agg_df["workers"] == 1, "one", "two or more"))


@orca.column(table_name="households")
def hh_race_of_head(persons_grouped_household):
    agg_df = persons_grouped_household\
        .agg(race_of_head=("race_head", "sum"))
    return np.where(
        agg_df["race_of_head"] == 1, "white",
        np.where(
            agg_df["race_of_head"] == 2,
            "black",
            np.where(agg_df["race_of_head"].isin([6, 7]), "asian", "other"),
        ),
    )


@orca.column(table_name="households")
def hh_size(persons_grouped_household):
    agg_df = persons_grouped_household.size()
    return np.where(
        agg_df == 1, "one",
        np.where(agg_df == 2, "two",
            np.where(agg_df == 3, "three", "four or more"),
        ),
    )



@orca.step("households_reorg")
def households_reorg(persons, households, year):
    """
    Households reorganization module

    Modifies State Variables:
        - persons.relate
        - persons.MAR
        - persons.household_id

    Args:
        persons (DataFrameWrapper): DataFrameWrapper of the persons table
        households (DataFrameWrapper): DataFrameWrapper of the households table

    Returns:
        None
    """
    # Marriage Model
    single_noncohab_index = ~persons["cohabitate"] & persons["is_single"]

    print("Running marriage model...")
    marriage = mm.get_step("marriage")
    # TODO: The marriage.variable_names part was necessary because the Multinomial logit does not
    #       call .to_frame() inside. This is inconsistent with the binary logit interface
    marriage_list = marriage.run(persons.to_frame(marriage.variable_names).loc[single_noncohab_index])

    # Divorce model
    married_household_sizes = persons.local\
        .loc[(persons["MAR"] == 1) & persons["relate"].isin([0, 1])] \
        .groupby("household_id").size()
    married_households_living_together = married_household_sizes[married_household_sizes == 2].index.tolist()

    # TODO: Rethink how we are passing the fitlers here
    print("Running divorce model...")
    households["divorced"] = -99
    divorce_model = mm.get_step("divorce")
    divorce_model_variables = columns_in_formula(divorce_model.model_expression)
    divorce_model_data = households.to_frame(divorce_model_variables).loc[married_households_living_together]
    divorce_list = divorce_model.run_with_data(divorce_model_data).astype(int)
    
    # Cohabitation to X Model
    print("Running cohabitation model...")
    ELIGIBLE_HOUSEHOLDS = (
        persons.local[(persons["relate"] == 13) & \
                   (persons["MAR"]!=1) & \
                   ((persons["age"]>=15))]["household_id"].unique().astype(int)
    )
    cohabitation = mm.get_step("cohabitation")
    cohabitate_x_list = cohabitation.run(households.to_frame(cohabitation.variable_names).loc[ELIGIBLE_HOUSEHOLDS])
    
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

    Args:
        persons (DataFrameWrapper): DataFrameWrapper of persons table
        households (DataFrameWrapper): DataFrameWrapper of households table
        cohabitate_list (pd.Series): Pandas Series of cohabitation model output

    Returns:
        None
    """
    # persons_df = orca.get_table("persons").local
    # households_df = orca.get_table("households").local

    # Precompute some indices
    married_hh = cohabitate_list.index[cohabitate_list == 2].to_list()
    breakup_hh = cohabitate_list.index[cohabitate_list == 1].to_list()
    newly_married_persons_index = persons["household_id"].isin(married_hh)
    newly_brokeup_persons_index = persons["household_id"].isin(breakup_hh)
    unmarried_partner_index = persons["relate"] == 13
    married_or_reference_index = persons["relate"].isin([0, 1])

    # Perform update for people that got married
    persons.local.loc[newly_married_persons_index & unmarried_partner_index, "relate",] = 1
    persons.local.loc[newly_married_persons_index & married_or_reference_index, "MAR"] = 1

    # Moved this to a lazily computed column
    # persons_df = (persons_df.reset_index().merge(hh_df, on=["household_id"]).set_index("person_id"))

    # Perform update for people that broke up
    leaving_person_index = newly_brokeup_persons_index & unmarried_partner_index
    staying_person_index = newly_brokeup_persons_index & unmarried_partner_index

    ## Person leaving is now head of household
    persons.local.loc[leaving_person_index, "relate"] = 0

    ### Assign new household_id to people leaving
    persons.local.loc[leaving_person_index, "relate"] = (
        np.arange(leaving_person_index.sum()) # = [0, 1, 2 ...] up to the number of people that left
        + households.local.index.max()        # = [max_hh_id, max_household_id + 1, ...]
        + 1
    )

    ### These were (mistakenly?) computed only for people leaving
    ### I suspect they are not needed, yet I'm leaving here for now
    ### They are columns of the households table
    # households_new["tenure"] = "unknown"
    # households_new["recent_mover"] = "unknown"
    # households_new["sf_detached"] = "unknown"
    # households_new["tenure_mover"] = "unknown"
    # households_new["block_id"] = "-1"
    # households_new["hh_type"] = "-1"
    # households_new["cars"] = np.random.choice([0, 1], size=households_new.shape[0])
    # households_new["hh_cars"] = np.where(
    #     households_new["cars"] == 0,
    #     "none",
    #     np.where(households_new["cars"] == 1, "one", "two or more"),
    # )

    ## This logic updated a "metadata" table which hold the max person and household ids
    ## I believe this is error prone so I changed the logic of that and plan to eliminate the metadata
    ## table altogether

    # metadata = orca.get_table("metadata").to_frame()
    # max_hh_id = metadata.loc["max_hh_id", "value"]
    # max_p_id = metadata.loc["max_p_id", "value"]
    # if households_df.index.max() > max_hh_id:
    #     metadata.loc["max_hh_id", "value"] = households_df.index.max()
    # if persons_df.index.max() > max_p_id:
    #     metadata.loc["max_p_id", "value"] = persons_df.index.max()
    # orca.add_table("metadata", metadata)


    # leaving_person_index = persons_df.index[(persons_df["household_id"].isin(breakup_hh)) & (persons_df["relate"] == 13)]
    # leaving_house = persons_df.loc[leaving_person_index].copy()
    # leaving_house["relate"] = 0

    # persons_df = persons_df.drop(leaving_person_index)

    # Update characteristics for households staying
    # persons_df["person"] = 1                                                          # TODO: Unnecessary?
    # persons_df["is_head"] = np.where(persons_df["relate"] == 0, 1, 0)                 # Moved to computed column
    # persons_df["race_head"] = persons_df["is_head"] * persons_df["race_id"]           # Moved to computed column
    # persons_df["age_head"] = persons_df["is_head"] * persons_df["age"]                # Moved to computed column
    # persons_df["hispanic_head"] = persons_df["is_head"] * persons_df["hispanic"]      # Moved to computed column
    # persons_df["child"] = np.where(persons_df["relate"].isin([2, 3, 4, 14]), 1, 0)    # Already defined in aging module
    # persons_df["senior"] = np.where(persons_df["age"] >= 65, 1, 0)                    # Already defined in aging module
    # persons_df["age_gt55"] = np.where(persons_df["age"] >= 55, 1, 0)                  # Already defined in aging module

    # households_new = persons_df.groupby("household_id").agg(income=("earning", "sum"),race_of_head=("race_head", "sum"),age_of_head=("age_head", "sum"), workers=("worker", "sum"),hispanic_status_of_head=("hispanic", "sum"),persons=("person", "sum"),children=("child", "sum"),seniors=("senior", "sum"),gt55=("age_gt55", "sum"),
    # )

    # households_new["hh_age_of_head"] = np.where(households_new["age_of_head"] < 35,"lt35",np.where(households_new["age_of_head"] < 65, "gt35-lt65", "gt65"),)
    # households_new["hispanic_head"] = np.where( households_new["hispanic_status_of_head"] == 1, "yes", "no")
    # households_new["hh_children"] = np.where( households_new["children"] >= 1, "yes", "no")
    # households_new["hh_seniors"] = np.where(households_new["seniors"] >= 1, "yes", "no")
    # households_new["gt2"] = np.where(households_new["persons"] >= 2, 1, 0)
    # households_new["gt55"] = np.where(households_new["gt55"] >= 1, 1, 0)
    # households_new["hh_income"] = np.where(households_new["income"] < 30000,"lt30",np.where(households_new["income"] < 60,
    #         "gt30-lt60",
    #         np.where(
    #             households_new["income"] < 100,
    #             "gt60-lt100",
    #             np.where(households_new["income"] < 150, "gt100-lt150", "gt150"),
    #         ),
    #     ),
    # )
    # households_new["hh_workers"] = np.where(
    #     households_new["workers"] == 0,
    #     "none",
    #     np.where(households_new["workers"] == 1, "one", "two or more"),
    # )

    # households_new["hh_race_of_head"] = np.where(
    #     households_new["race_of_head"] == 1,
    #     "white",
    #     np.where(
    #         households_new["race_of_head"] == 2,
    #         "black",
    #         np.where(households_new["race_of_head"].isin([6, 7]), "asian", "other"),
    #     ),
    # )

    # households_new["hh_size"] = np.where(
    #     households_new["persons"] == 1,
    #     "one",
    #     np.where(
    #         households_new["persons"] == 2,
    #         "two",
    #         np.where(households_new["persons"] == 3, "three", "four or more"),
    #     ),
    # )

    # households_df.update(households_new)

    # metadata = orca.get_table("metadata").to_frame()
    # max_hh_id = metadata.loc["max_hh_id", "value"]
    # # Create household characteristics for new households formed
    # leaving_house["household_id"] = (
    #     np.arange(len(breakup_hh))
    #     + max(max_hh_id, households_df.index.max())
    #     + 1
    # )

    # All of these were addressed by moving them to computed columns
    # leaving_house["person"] = 1
    # leaving_house["is_head"] = np.where(leaving_house["relate"] == 0, 1, 0)
    # leaving_house["race_head"] = leaving_house["is_head"] * leaving_house["race_id"]
    # leaving_house["age_head"] = leaving_house["is_head"] * leaving_house["age"]
    # leaving_house["hispanic_head"] = (
    #     leaving_house["is_head"] * leaving_house["hispanic"]
    # )
    # leaving_house["child"] = np.where(leaving_house["relate"].isin([2, 3, 4, 14]), 1, 0)
    # leaving_house["senior"] = np.where(leaving_house["age"] >= 65, 1, 0)
    # leaving_house["age_gt55"] = np.where(leaving_house["age"] >= 55, 1, 0)

    # households_new = leaving_house.groupby("household_id").agg(
    #     income=("earning", "sum"),
    #     race_of_head=("race_head", "sum"),
    #     age_of_head=("age_head", "sum"),
    #     workers=("worker", "sum"),
    #     hispanic_status_of_head=("hispanic", "sum"),
    #     persons=("person", "sum"),
    #     children=("child", "sum"),
    #     seniors=("senior", "sum"),
    #     gt55=("age_gt55", "sum"),
    #     lcm_county_id=("lcm_county_id", "first"),
    # )

    # households_new["hh_age_of_head"] = np.where(
    #     households_new["age_of_head"] < 35,
    #     "lt35",
    #     np.where(households_new["age_of_head"] < 65, "gt35-lt65", "gt65"),
    # )
    # households_new["hispanic_head"] = np.where(
    #     households_new["hispanic_status_of_head"] == 1, "yes", "no"
    # )
    # households_new["hh_children"] = np.where(
    #     households_new["children"] >= 1, "yes", "no"
    # )
    # households_new["hh_seniors"] = np.where(households_new["seniors"] >= 1, "yes", "no")
    # households_new["gt2"] = np.where(households_new["persons"] >= 2, 1, 0)
    # households_new["gt55"] = np.where(households_new["gt55"] >= 1, 1, 0)
    # households_new["hh_income"] = np.where(
    #     households_new["income"] < 30000,
    #     "lt30",
    #     np.where(
    #         households_new["income"] < 60,
    #         "gt30-lt60",
    #         np.where(
    #             households_new["income"] < 100,
    #             "gt60-lt100",
    #             np.where(households_new["income"] < 150, "gt100-lt150", "gt150"),
    #         ),
    #     ),
    # )
    # households_new["hh_workers"] = np.where(
    #     households_new["workers"] == 0,
    #     "none",
    #     np.where(households_new["workers"] == 1, "one", "two or more"),
    # )

    # households_new["hh_race_of_head"] = np.where(
    #     households_new["race_of_head"] == 1,
    #     "white",
    #     np.where(
    #         households_new["race_of_head"] == 2,
    #         "black",
    #         np.where(households_new["race_of_head"].isin([6, 7]), "asian", "other"),
    #     ),
    # )

    # households_new["hh_size"] = np.where(
    #     households_new["persons"] == 1,
    #     "one",
    #     np.where(
    #         households_new["persons"] == 2,
    #         "two",
    #         np.where(households_new["persons"] == 3, "three", "four or more"),
    #     ),
    # )

    # households_new["cars"] = np.random.choice([0, 1], size=households_new.shape[0])
    # households_new["hh_cars"] = np.where(
    #     households_new["cars"] == 0,
    #     "none",
    #     np.where(households_new["cars"] == 1, "one", "two or more"),
    # )
    # households_new["tenure"] = "unknown"
    # households_new["recent_mover"] = "unknown"
    # households_new["sf_detached"] = "unknown"
    # households_new["tenure_mover"] = "unknown"
    # households_new["block_id"] = "-1"
    # households_new["hh_type"] = "-1"
    # households_df = pd.concat([households_df, households_new])

    # persons_df = pd.concat([persons_df, leaving_house])

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
    # orca.add_table("households", households_df[households_local_cols])
    # orca.add_table("persons", persons_df[persons_local_cols])
    # orca.add_injectable(
    #     "max_hh_id", max(orca.get_injectable("max_hh_id"), households_df.index.max())
    # )


def fix_erroneous_households(persons, households):
    """ YE: *** """
    print("Fixing erroneous households")
    p_df = persons.local
    household_cols = households.local_columns
    household_df = households.local
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