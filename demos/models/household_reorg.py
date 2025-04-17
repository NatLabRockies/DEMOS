import time
import orca
import numpy as np
import pandas as pd
from templates import estimated_models, modelmanager as mm
from templates.utils.models import columns_in_formula
from .marriage import update_married_households_random, update_married_households, update_divorce

@orca.injectable(autocall=False)
def get_new_households(n, persons, graveyard):
    current_max = pd.concat([persons.local, graveyard.local], ignore_index=True).household_id.max()
    return (
        np.arange(n)                         # = [0, 1, 2 ...] up to the number of people
        + current_max   # = [max_hh_id, max_household_id + 1, ...]
        + current_max   # = [max_hh_id, max_household_id + 1, ...]
        + 1
    )

@orca.column(table_name="persons", cache=True, cache_scope="step")
def cohabitate(persons):
    unmarried_partner_index = persons["relate"] == 13
    cohab_household_ids = persons["household_id"].loc[unmarried_partner_index].unique()
    return unmarried_partner_index | \
            ((persons["relate"] == 0) & persons["household_id"].isin(cohab_household_ids))


@orca.column(table_name="persons", cache=True, cache_scope="step")
def is_not_married(persons):
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
def hh_agegroup_of_head(persons_grouped_household):
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
def gt2(persons_grouped_household):
    agg_df = persons_grouped_household.size() > 2
    return agg_df.astype(int)


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
def households_reorg(persons, households, year, get_new_households, graveyard):
    """
    Households reorganization module

    Modifies State Variables:
        - persons.relate
        - persons.MAR
        - persons.household_id
        - persons.member_id

    Args:
        persons (DataFrameWrapper): DataFrameWrapper of the persons table
        households (DataFrameWrapper): DataFrameWrapper of the households table

    Returns:
        None
    """
    # Marriage Model
    single_noncohab_index = ~persons["cohabitate"] & persons["is_not_married"]

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
        persons.local[(persons["relate"] == 13) & persons["is_not_married"]]["household_id"] \
            .unique().astype(int)
    )
    cohabitation = mm.get_step("cohabitation")
    cohabitate_x_list = cohabitation.run(households.to_frame(cohabitation.variable_names).loc[ELIGIBLE_HOUSEHOLDS])
    
    ######### UPDATING
    print("Restructuring households:")
    print("Cohabitations..")
    update_cohabitating_households(persons, cohabitate_x_list, get_new_households, graveyard)
    print_household_stats()
    
    print("Marriages..")
    update_married_households_random(persons, marriage_list, get_new_households, graveyard)
    print_household_stats()
    fix_erroneous_households(persons)
    print_household_stats()
    
    print("Divorces..")
    update_divorce(persons, divorce_list, get_new_households, graveyard)
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


def update_cohabitating_households(persons, cohabitate_list, get_new_households, graveyard):
    """
    Updating households and persons after cohabitation model.

    Args:
        persons (DataFrameWrapper): DataFrameWrapper of persons table
        households (DataFrameWrapper): DataFrameWrapper of households table
        cohabitate_list (pd.Series): Pandas Series of cohabitation model output

    Returns:
        None
    """
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

    # Perform update for people that broke up
    leaving_person_index = newly_brokeup_persons_index & unmarried_partner_index

    ## Person leaving is now head of household
    persons.local.loc[leaving_person_index, "relate"] = 0

    ### Assign new household_id to people leaving
    persons.local.loc[leaving_person_index, "household_id"] = get_new_households(leaving_person_index.sum(), persons, graveyard)


def fix_erroneous_households(persons):
    """
    """
    n_partners_df = persons.local[(persons["relate"] == 1) | (persons["relate"] == 13)] \
        .groupby("household_id")["relate"] \
        .nunique() \
        .reset_index()
    households_to_drop = n_partners_df[n_partners_df["relate"] == 2]["household_id"].to_list()

    # Drop the households
    if len(households_to_drop) > 0:
        persons.local = persons.local[~persons.local["household_id"].isin(households_to_drop)]