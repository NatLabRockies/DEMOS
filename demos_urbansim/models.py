import math
import os
import time
import warnings
from operator import index
from typing import Sized

warnings.filterwarnings("ignore")

import indicators
import numpy as np
import orca
import pandana as pdna
import pandas as pd
import stopwatch
import yaml
from google.cloud import storage
from scipy.spatial.distance import cdist
from scipy.special import softmax
from urbansim.developer import developer

# import demo_models
from urbansim.models import GrowthRateTransition, transition
from urbansim_templates import modelmanager as mm
from urbansim_templates.models import BinaryLogitStep, OLSRegressionStep

# modelmanager.initialize()

print("importing models for region", orca.get_injectable("region_code"))

# -----------------------------------------------------------------------------------------
# PREPROCESSING
# -----------------------------------------------------------------------------------------

@orca.step("work_location")
def work_location(persons):

    # This workaorund is necesary to make the work
    # location choice run in batches, with improves
    # simulation efficiency.
    # breakpoint()
    # At this breakpoint, look at the persons who have no work locations
    persons_df_debug = orca.get_table("persons").local
    num_workers = persons_df_debug[(persons_df_debug["worker"]==1)].shape[0]
    num_workers_no_loc = persons_df_debug[(persons_df_debug["worker"]==1) & (persons_df_debug["work_block_id"]=="-1")].shape[0]
    print("Share of workers with no work loc before wlcm:", num_workers_no_loc/num_workers)
    model = mm.get_step('wlcm')
    model.run(chooser_batch_size = 100000)
    # breakpoint()
    # work_block_id = persons['work_block_id']
    persons_work = orca.get_table("persons").to_frame(columns=["work_block_id"])
    persons_work = persons_work.reset_index()
    persons_df_debug = orca.get_table("persons").local
    num_workers = persons_df_debug[(persons_df_debug["worker"]==1)].shape[0]
    num_workers_no_loc = persons_df_debug[(persons_df_debug["worker"]==1) & (persons_df_debug["work_block_id"]=="-1")].shape[0]
    print("Share of workers with no work loc after wlcm:", num_workers_no_loc/num_workers)
    # breakpoint()
    # At this breakpoint, see what changes, if any?
    orca.add_table('work_locations', persons_work.fillna('-1'))

@orca.step("work_location_stats")
def work_location_stats(persons):
    persons_work = orca.get_table("persons").to_frame(columns=["work_block_id"])
    persons_work = persons_work.reset_index()
    persons_df_debug = orca.get_table("persons").local
    num_workers = persons_df_debug[(persons_df_debug["worker"]==1)].shape[0]
    num_workers_no_loc = persons_df_debug[(persons_df_debug["worker"]==1) & (persons_df_debug["work_block_id"]=="-1")].shape[0]
    num_people_no_loc = persons_df_debug[(persons_df_debug["work_block_id"]=="-1")].shape[0]
    print("Share of workers with no work loc:", num_workers_no_loc/num_workers)
    print("Share of people with no work loc:", num_people_no_loc/persons_work.shape[0])

@orca.step("status_report")
def status_report(year):
    print("------------------------------------------------")
    print("STATUS REPORT")
    for tbl in ["households", "jobs", "residential_units"]:
        df = orca.get_table(tbl)
        df = df.local
        unplaced = df[df["block_id"] == "-1"]
        print("------------------------------------------------")
        print(tbl, "unplaced: ", len(unplaced.index))
        if len(unplaced.index) > 0:
            print("Unique values per variable in unplaced agents:")
            for var in df.columns:
                print("         ", var, ": ", unplaced[var].unique())
    print("------------------------------------------------")

# @orca.step("print_columns")
# def print_columns(persons, households):
#     print(persons.local.columns)
#     print(households.local.columns)

@orca.step()
def build_networks(blocks, block_groups, nodes, edges):
    nodes, edges = nodes.local, edges.local
    print("Number of nodes is %s." % len(nodes))
    print("Number of edges is %s." % len(edges))
    nodes = nodes.set_index("id")
    net = pdna.Network(
        nodes["x"], nodes["y"], edges["from"], edges["to"], edges[["weight"]]
    )

    precompute_distance = 1000
    print("Precomputing network for distance %s." % precompute_distance)
    print("Network precompute starting.")
    net.precompute(precompute_distance)
    print("Network precompute done.")

    blocks = blocks.local
    blocks["node_id"] = net.get_node_ids(blocks["x"], blocks["y"])
    orca.add_column("blocks", "node_id", blocks["node_id"])

    block_groups = block_groups.to_frame(["x", "y"])
    block_groups["node_id"] = net.get_node_ids(block_groups["x"], block_groups["y"])
    orca.add_column("block_groups", "node_id", block_groups["node_id"])

    orca.add_injectable("net", net)


# def read_yaml(path):
#     """A function to read YAML file"""
#     with open(path) as f:
#         config = list(yaml.safe_load_all(f))[0]

#     return config


def simulation_mnl(data, coeffs):
    """Function to run simulation of the MNL model

    Args:
        data (_type_): _description_
        coeffs (_type_): _description_

    Returns:
        Pandas Series: Pandas Series of the outcomes of the simulated model
    """
    utils = np.dot(data, coeffs)
    base_util = np.zeros(utils.shape[0])
    utils = np.column_stack((base_util, utils))
    probabilities = softmax(utils, axis=1)
    s = probabilities.cumsum(axis=1)
    r = np.random.rand(probabilities.shape[0]).reshape((-1, 1))
    choices = (s < r).sum(axis=1)
    return pd.Series(index=data.index, data=choices)


@orca.step("add_temp_variables")
def add_temp_variables():
    """Adds temporary variables to the persons and
    households tables.
    """
    persons = orca.get_table("persons").local
    persons.loc[:, "dead"] = -99
    persons.loc[:, "stop"] = -99
    persons.loc[:, "kid_moves"] = -99

    orca.add_table("persons", persons)



@orca.step("remove_temp_variables")
def remove_temp_variables():
    """Removes temporary variables from the persons
    and households tables
    """
    persons = orca.get_table("persons").local
    persons = persons.drop(columns=["dead", "stop", "kid_moves"])
    orca.add_table("persons", persons)


# -----------------------------------------------------------------------------------------
# DEMOS
# -----------------------------------------------------------------------------------------


@orca.step("income_stats")
def income_stats(persons, households):
    """Function to print the number of households from both the households and pers

    Args:
        persons (DataFrame): Pandas DataFrame of the persons table
        households (DataFrame): Pandas DataFrame of the households table
    """

    persons_df = orca.get_table("persons").local
    households_df = orca.get_table("households").local
    print("Households median Income: ", households_df["income"].median())
    print("Households median income persons table: ", persons_df.groupby("household_id")["earning"].sum().median())

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


@orca.step("fatality_model")
def fatality_model(persons, households, year):
    """Function to run the fatality model at the persons level.
    The function also updates the persons and households tables,
    and saves the mortalities table.

    Args:
        persons (DataFrameWrapper): DataFrameWrapper of persons table
        households (DataFrameWrapper): DataFrameWrapper of households table
    """
    persons_df = orca.get_table("persons").local
    persons_df["dead"] = -99
    orca.add_table("persons", persons_df)
    # print("Persons shape: ", persons_df.shape[0])
    # Running fatality Model
    mortality = mm.get_step("mortality")
    # mortality.run()
    # fatality_list = mortality.choices.astype(int)
    # print(fatality_list.sum(), " fatalities")

    mortality.run()
    fatality_list = mortality.choices.astype(int)
    predicted_share = fatality_list.sum() / persons_df.shape[0]
    observed_fatalities = orca.get_table("observed_fatalities_data").to_frame()
    target = observed_fatalities[observed_fatalities["year"]==year]["count"]
    target_share = target / persons_df.shape[0]

    error = np.sqrt(np.mean((fatality_list.sum() - target)**2))
    while error >= 1000:
        mortality.fitted_parameters[0] += np.log(target.sum()/fatality_list.sum())
        mortality.run()
        fatality_list = mortality.choices.astype(int)
        predicted_share = fatality_list.sum() / persons_df.shape[0]
        error = np.sqrt(np.mean((fatality_list.sum() - target)**2))
    # print("Fatality list count: ", fatality_list.value_counts())
    # print(fatality_list.sum(), " fatalities")
    # print("Fatality list shape: ", fatality_list.shape)
    # Updating the households and persons tables
    households = orca.get_table("households")
    persons = orca.get_table("persons")
    remove_dead_persons(persons, households, fatality_list, year)

    # Update mortalities table
    mortalities = orca.get_table("mortalities").to_frame()
    if mortalities.empty:
        mortalities = pd.DataFrame(
            data={"year": [year], "count": [fatality_list.sum()]}
        )
    else:
        mortalities_new = pd.DataFrame(
            data={"year": [year], "count": [fatality_list.sum()]}
        )

        mortalities = pd.concat([mortalities, mortalities_new], ignore_index=True) 
    orca.add_table("mortalities", mortalities)



@orca.step("update_income")
def update_income(persons, households, year):
    """
    Updating income for persons and households

    Args:
        persons (DataFrameWrapper): DataFrameWrapper of persons table
        households (DataFrameWrapper): DataFrameWrapper of households table
        year (int): simulation year
    """
    # Pulling data, income rates, and county IDs
    persons_df = orca.get_table("persons").local
    households_df = orca.get_table("households").local

    households_local_cols = households_df.columns
    persons_local_cols = persons_df.columns
    # print(persons_local_cols)
    hh_counties = households_df["lcm_county_id"].copy()

    income_rates = orca.get_table("income_rates").to_frame()
    income_rates = income_rates[income_rates["year"] == year]

    persons_df = (persons_df.reset_index().merge(hh_counties.reset_index(), on=["household_id"]).set_index("person_id"))
    persons_df = (persons_df.reset_index().merge(income_rates, on=["lcm_county_id"]).set_index("person_id"))
    persons_df["earning"] = persons_df["earning"] * (1 + persons_df["rate"])

    new_incomes = persons_df.groupby("household_id").agg(income=("earning", "sum"))

    households_df.update(new_incomes)
    households_df["income"] = households_df["income"].astype(int)
    persons_df["member_id"] = persons_df.groupby("household_id")["relate"].rank(method="first", ascending=True).astype(int)
    persons_local_columns = orca.get_injectable("persons_local_cols")
    orca.add_table("persons", persons_df[persons_local_columns])
    orca.add_table("households", households_df[households_local_cols])
    orca.add_table("persons", persons_df[persons_local_cols])
    # Update income stats at the persons level
    income_over_time = orca.get_table("income_over_time").to_frame()
    if income_over_time.empty:
        income_over_time = pd.DataFrame(
            data={"year": [year], "mean_income": [persons_df["earning"].mean()]}
        )
    else:
        new_income_over_time = pd.DataFrame(
                data={"year": [year], "mean_income": [persons_df["earning"].mean()]}
            )
        income_over_time = pd.concat([income_over_time,
                                      new_income_over_time],
                                     ignore_index=True)
    orca.add_table("income_over_time", income_over_time)


# Mortality model returns a list of 0s representing alive and 1 representing dead
# Then adds that list to the persons table and updates persons and households tables accordingly
def remove_dead_persons(persons, households, fatality_list, year):
    """
    This function updates the persons table from the output of the fatality model.
    Takes in the persons and households orca tables.

    Args:
        persons (DataFramWrapper): DataFramWrapper of persons table
        households (DataFramWrapper): DataFramWrapper of households table
        fatality_list (pd.Series): Pandas Series of fatality list
    """
    # pd.set_option('display.max_columns', None)
    # print("Starting to update demos")
    # Read tables and store as DataFrames
    houses = households.local
    households_columns = orca.get_injectable("households_local_cols")

    # Pulling the persons data
    persons_df = persons.local
    persons_columns = orca.get_injectable("persons_local_cols")

    persons_df["dead"] = -99
    persons_df["dead"] = fatality_list
    # print("Fatality outcomes: ", persons_df["dead"].value_counts())
    graveyard = persons_df[persons_df["dead"] == 1].copy()
    # print(persons_df["household_id"].unique().shape[0])
    # print(houses.index.unique().shape[0])
    #################################
    # HOUSEHOLD WHERE EVERYONE DIES #
    #################################
    # Get households where everyone died
    persons_df["member"] = 1
    dead_frac = persons_df.groupby("household_id").agg(
        num_dead=("dead", "sum"), size=("member", "sum")
    )
    dead_households = dead_frac[
        dead_frac["num_dead"] == dead_frac["size"]
    ].index.to_list()

    grave_households = houses[houses.index.isin(dead_households)].copy()
    grave_persons = persons_df[persons_df["household_id"].isin(dead_households)].copy()

    # Drop out of the persons table
    persons_df = persons_df.loc[~persons_df["household_id"].isin(dead_households)]
    # Drop out of the households table
    houses = houses.drop(dead_households)

    ##################################################
    ##### HOUSEHOLDS WHERE PART OF HOUSEHOLD DIES ####
    ##################################################
    dead = persons_df[persons_df["dead"] == 1].copy()
    alive = persons_df[persons_df["dead"] == 0].copy()
    # print("Finished splitting dead and alive")

    #################################
    # Alive heads, Dead partners
    #################################
    # Dead partners, either married or cohabitating
    dead_partners = dead[dead["relate"].isin([1, 13])]  # This will need changed
    # Alive heads
    alive_heads = alive[alive["relate"] == 0]
    alive_heads = alive_heads[["household_id", "MAR"]]
    widow_heads = alive[(alive["household_id"].isin(dead_partners["household_id"])) & (alive["relate"]==0)].copy()
    # widow_heads = widow_heads.set_index("person_id")
    widow_heads["MAR"].values[:] = 3

    # Dead heads, alive partners
    dead_heads = dead[dead["relate"] == 0]
    alive_partner = alive[alive["relate"].isin([1, 13])].copy()
    alive_partner = alive_partner[
        ["household_id", "MAR"]
    ]  ## Pull the relate status and update it
    widow_partners = alive[(alive["household_id"].isin(dead_heads["household_id"])) & (alive["relate"].isin([1, 13]))].copy()
    #alive_partner.reset_index().merge(
    #    dead_heads[["household_id"]], how="inner", on="household_id"
    #)
    #widow_partners = widow_partners.set_index("person_id")
    widow_partners["MAR"].values[:] = 3  # THIS MIGHT NEED VERIFICATION, WHAT DOES MAR MEAN?

    # Merge the two groups of widows
    widows = pd.concat([widow_heads, widow_partners])[["MAR"]]
    # Update the alive database's MAR values using the widows table
    alive_copy = alive.copy()
    alive.loc[widows.index, "MAR"] = 3
    #alive = widows.combine_first(alive)
    alive["MAR"] = alive["MAR"].astype(int)

    # if alive_copy.index.has_duplicates:
    #     breakpoint()
    
    # if alive.index.has_duplicates:
    #     breakpoint()
    # print("Finished updating marital status")

    # Select the households in alive where the heads died
    alive_sort = alive[alive["household_id"].isin(dead_heads["household_id"])].copy()
    alive_sort["relate"] = alive_sort["relate"].astype(int)

    # breakpoint()
    if len(alive_sort.index) > 0:
        alive_sort.sort_values("relate", inplace=True)
        # Restructure all the households where the head died
        alive_sort = alive_sort[["household_id", "relate", "age"]]
        # print("Starting to restructure household")
        # Apply the rez function
        alive_sort = alive_sort.groupby("household_id").apply(rez)

        # Update relationship values and make sure correct datatype is used
        alive.loc[alive_sort.index, "relate"] = alive_sort["relate"]
        alive["relate"] = alive["relate"].astype(int)
        # print("Finished restructuring households")

    alive["is_relate_0"] = (alive["relate"]==0).astype(int)
    alive["is_relate_1"] = (alive["relate"]==1).astype(int)

    alive_agg = alive.groupby("household_id").agg(sum_relate_0 = ("is_relate_0", "sum"), sum_relate_1 = ("is_relate_1", "sum"))
    
    # Dropping households with more than one head or more than one partner
    alive_agg = alive_agg[(alive_agg["sum_relate_1"]<=1) & (alive_agg["sum_relate_0"]<=1)]
    alive_hh = alive_agg.index.tolist()
    alive = alive[alive["household_id"].isin(alive_hh)]

    alive["person"] = 1
    alive["is_head"] = np.where(alive["relate"] == 0, 1, 0)
    alive["race_head"] = alive["is_head"] * alive["race_id"]
    alive["age_head"] = alive["is_head"] * alive["age"]
    alive["hispanic_head"] = alive["is_head"] * alive["hispanic"]
    alive["child"] = np.where(alive["relate"].isin([2, 3, 4, 14]), 1, 0)
    alive["senior"] = np.where(alive["age"] >= 65, 1, 0)
    alive["age_gt55"] = np.where(alive["age"] >= 55, 1, 0)

    households_new = alive.groupby("household_id").agg(
        income=("earning", "sum"),
        race_of_head=("race_head", "sum"),
        age_of_head=("age_head", "sum"),
        workers=("worker", "sum"),
        hispanic_status_of_head=("hispanic", "sum"),
        persons=("person", "sum"),
        children=("child", "sum"),
        seniors=("senior", "sum"),
        gt55=("age_gt55", "sum"),
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
    # breakpoint()

    houses.update(households_new)
    houses = houses.loc[alive_hh]
    # print("Updating age stats table")
    # Get the age over time table populated
    age_over_time = orca.get_table("age_over_time").to_frame()
    if age_over_time.empty:
        age_over_time = pd.DataFrame([alive["person_age"].value_counts()])
    else:
        new_age_over_time = pd.DataFrame([alive["person_age"].value_counts()])
        age_over_time = pd.concat([age_over_time, new_age_over_time], ignore_index=True)
    orca.add_table("age_over_time", age_over_time)

    # print("Update the population stats over time.")
    # Update the population over time stats
    graveyard_table = orca.get_table("pop_over_time").to_frame()
    if graveyard_table.empty:
        dead_people = grave_persons.copy()

    else:
        dead_people = pd.concat([graveyard_table, grave_persons])

    # print("Update the dead households and graveyard.")
    # Load the dead households and graveyard table
    # Update persons table
    orca.add_table("persons", alive[persons_columns])
    orca.add_table("households", houses[households_columns])
    orca.add_table("graveyard", dead_people[persons_columns])
    # orca.add_injectable(
    #     "max_p_id", orca.get_injectable("max_p_id"), alive["household_id"].max()
    # )
    metadata = orca.get_table("metadata").to_frame()
    max_hh_id = metadata.loc["max_hh_id", "value"]
    max_p_id = metadata.loc["max_p_id", "value"]
    persons_df = orca.get_table("persons").local
    households_df = orca.get_table("households").local
    if households_df.index.max() > max_hh_id:
        metadata.loc["max_hh_id", "value"] = households_df.index.max()
    if persons_df.index.max() > max_p_id:
        metadata.loc["max_p_id", "value"] = persons_df.index.max()
    orca.add_table("metadata", metadata)
    # print("DONE updating persons.")


def rez(group):
    """
    Function to change the household head role
    TODO: This needs to become vectorized to make it faster.
    """
    # Update the relate variable for the group
    if group["relate"].iloc[0] == 1:
        group["relate"].iloc[0] = 0
        return group
    if 13 in group["relate"].values:
        group["relate"].replace(13, 0, inplace=True)
        return group

    # Get the maximum age of the household, oldest person becomes head of household
    # Verify this with Juan.
    new_head_idx = group["age"].idxmax()
    # Function to map the relation of new head
    map_func = produce_map_func(group.loc[new_head_idx, "relate"])
    group.loc[new_head_idx, "relate"] = 0
    # breakpoint()
    group.relate = group.relate.map(map_func)
    return group


# TODO refactor this method so it can be cleanly used with any model
# Might need to add some stuff (update more variables) or have a flag to determine which function is calling this
def update_households(alive, dead, old_incomes, subtract=True):
    """
    Function to update the households characteristics, namely income, num of workers, race, and age of head.
    """
    # By default the dead variable represents people dying or leaving a household
    if subtract:
        alive = alive.sort_values("relate")
        dead_aggs = dead.groupby("household_id").agg({"earning": "sum"})

        aggregates = alive.groupby("household_id").agg(
            {"earning": "sum", "worker": "sum", "race_id": "first", "age": "first"}
        )
        # print(list(orca.get_table("households").to_frame().columns))
        # TODO -- REPLACE THESE OPERATIONS BY PANDAS OPERATIONS
        orca.get_table("households").update_col(
            "income", old_incomes.subtract(dead_aggs["earning"], fill_value=0)
        )
        orca.get_table("households").update_col("workers", aggregates["worker"])
        orca.get_table("households").update_col("race_of_head", aggregates["race_id"])
        orca.get_table("households").update_col("age_of_head", aggregates["age"])
    # In a certain case, the dead variable actually represents living people being added -- What is this case?
    else:
        aggregates = dead.groupby("household_id").agg(
            {"earning": "sum", "worker": "sum"}
        )
        house_df = orca.get_table("households").to_frame(columns=["income", "workers"])
        # TODO -- REPLACE THESE OPERATIONS WITH PANDAS OPERATIONS
        orca.get_table("households").update_col(
            "income", house_df["income"].add(aggregates["earning"], fill_value=0)
        )
        orca.get_table("households").update_col(
            "workers", house_df["workers"].add(aggregates["worker"], fill_value=0)
        )

    # return households


# Function that takes the head's previous role and returns a function
# that maps roles to new roles based on restructuring
def produce_map_func(old_role):
    """
    Function that uses the relationship mapping in the
    provided table and returns a function that maps
    new household roles.
    """
    # old role is the previous number of the person who has now been promoted to head of the household
    sold_role = str(old_role)
 
    def inner(role):
        rel_map = orca.get_table("rel_map").to_frame()
        if role == 0:
            new_role = 0
        else:
            new_role = rel_map.loc[role, sold_role]
        return new_role

    # Returns function that takes a persons old role and gives them a new one based on how the household is restructured
    return inner


@orca.step("update_age")
def update_age(persons, households):
    """
    This function updates the age of the persons table and
    updates the age of the household head in the household table.

    Args:
        persons (DataFrameWrapper): DataFrameWrapper of the persons table
        households (DataFrameWrapper): DataFrameWrapper of the households table

    Returns:
        None
    """
    # print("Updating age of individuals...")
    persons_df = persons.local
    persons_df["age"] += 1
    households_df = households.to_frame(columns=["age_of_head", "hh_age_of_head"])
    households_df["age_of_head"] += 1
    households_df["hh_age_of_head"] = np.where(
        households_df["age_of_head"] < 35,
        "lt35",
        np.where(households_df["age_of_head"] < 65, "gt35-lt65", "gt65"),
    )
    persons_df["child"] = np.where(persons_df["relate"].isin([2, 3, 4, 14]), 1, 0)
    persons_df["person"] = 1
    persons_df["senior"] = np.where(persons_df["age"] >= 65, 1, 0)
    households_stats = persons_df.groupby(["household_id"]).agg(
        children=("child", "sum"), seniors=("senior", "sum"), size=("person", "sum")
    )
    households_stats["hh_children"] = np.where(
        households_stats["children"] > 0, "yes", "no"
    )
    households_stats["gt55"] = np.where(households_stats["seniors"] > 0, 1, 0)
    households_stats["hh_seniors"] = np.where(
        households_stats["seniors"] > 0, "yes", "no"
    )
    # Update age of household head in household table
    # print("Updating household and persons tables...")
    orca.get_table("households").update_col("age_of_head", households_df["age_of_head"])
    orca.get_table("households").update_col(
        "hh_age_of_head", households_df["hh_age_of_head"]
    )
    orca.get_table("households").update_col(
        "hh_children", households_stats["hh_children"]
    )
    orca.get_table("households").update_col("gt55", households_stats["gt55"])
    orca.get_table("households").update_col(
        "hh_seniors", households_stats["hh_seniors"]
    )
    orca.get_table("persons").update_col("age", persons_df["age"])


def update_education_status(persons, student_list, year):
    """
    Function to update the student status in persons table based
    on the

    Args:
        persons (DataFrameWrapper): DataFrameWrapper of the persons table
        student_list (pd.Series): Pandas Series containing the output of
        the education model

    Returns:
        None
    """
    # Pull Data
    persons_df = persons.to_frame(
        columns=["age", "household_id", "edu", "student", "stop"]
    )
    persons_df["stop"] = student_list
    persons_df["stop"].fillna(2, inplace=True)

    # Update education level for individuals staying in school
    weights = persons_df["edu"].value_counts(normalize=True)

    persons_df.loc[persons_df["age"] == 3, "edu"] = 2
    persons_df.loc[persons_df["age"].isin([4, 5]), "edu"] = 4

    dropping_out = persons_df.loc[persons_df["stop"] == 1].copy()
    staying_school = persons_df.loc[persons_df["stop"] == 0].copy()

    dropping_out.loc[:, "student"] = 0
    staying_school.loc[:, "student"] = 1

    # high school and high school graduates proportions
    hs_p = persons_df[persons_df["edu"].isin([15, 16])]["edu"].value_counts(
        normalize=True
    )
    hs_grad_p = persons_df[persons_df["edu"].isin([16, 17])]["edu"].value_counts(
        normalize=True
    )
    # Students all the way to grade 10
    staying_school.loc[:, "edu"] = np.where(
        staying_school["edu"].between(4, 13, inclusive="both"),
        staying_school["edu"] + 1,
        staying_school["edu"],
    )
    # Students in grade 11 move to either 15 or 16 based on weights
    staying_school.loc[:, "edu"] = np.where(
        staying_school["edu"] == 14,
        np.random.choice([15, 16], p=[hs_p[15], hs_p[16]]),
        staying_school["edu"],
    )
    # Students in grade 12 either get hs degree or GED
    staying_school.loc[:, "edu"] = np.where(
        staying_school["edu"] == 15,
        np.random.choice([16, 17], p=[hs_grad_p[16], hs_grad_p[17]]),
        staying_school["edu"],
    )
    # Students with GED or HS Degree move to college
    staying_school.loc[:, "edu"] = np.where(
        staying_school["edu"].isin([16, 17]), 18, staying_school["edu"]
    )
    # Students with one year of college move to the next
    staying_school.loc[:, "edu"] = np.where(
        staying_school["edu"] == 18, 19, staying_school["edu"]
    )
    # Others to be added here.

    # Update education levels
    persons_df.update(staying_school)
    persons_df.update(dropping_out)

    orca.get_table("persons").update_col("edu", persons_df["edu"])
    orca.get_table("persons").update_col("student", persons_df["student"])

    # compute mean age of students
    # print("Updating students metrics...")
    students = persons_df[persons_df["student"] == 1]
    edu_over_time = orca.get_table("edu_over_time").to_frame()
    # student_population = orca.get_table("student_population").to_frame()
    # if student_population.empty:
    #     student_population = pd.DataFrame(
    #         data={"year": [year], "count": [students.shape[0]]}
    #     )
    # else:
    #     student_population_new = pd.DataFrame(
    #         data={"year": [year], "count": [students.shape[0]]}
    #     )
    #     students = pd.concat([student_population, student_population_new])
    # if edu_over_time.empty:
    #     edu_over_time = pd.DataFrame(
    #         data={"year": [year], "mean_age_of_students": [students["age"].mean()]}
    #     )
    # else:
    #     edu_over_time = edu_over_time.append(
    #         pd.DataFrame(
    #             {"year": [year], "mean_age_of_students": [students["age"].mean()]}
    #         ),
    #         ignore_index=True,
    #     )

    # orca.add_table("edu_over_time", edu_over_time)
    # orca.add_table("student_population", student_population)


@orca.step("education_model")
def education_model(persons, year):
    """
    Run the education model and update the persons table

    Args:
        persons (DataFrameWrapper): DataFrameWrapper of the persons table

    Returns:
        None
    """
    # Add temporary variable
    persons_df = persons.local
    persons_df["stop"] = -99
    orca.add_table("persons", persons_df)

    # Run the education model
    # print("Running the education model...")
    edu_model = mm.get_step("education")
    edu_model.run()
    student_list = edu_model.choices.astype(int)

    # Update student status
    # print("Updating student status...")
    update_education_status(persons, student_list, year)
    

@orca.step("laborforce_model")
def laborforce_model(persons, year):
    """
    Run the education model and update the persons table

    Args:
        persons (DataFrameWrapper): DataFrameWrapper of the persons table

    Returns:
        None
    """
    # Add temporary variable
    persons_df = orca.get_table("persons").local
    persons_df["stay_out"] = -99
    persons_df["leaving_workforce"] = -99
    orca.add_table("persons", persons_df)
    persons_df = orca.get_table("persons").local

    in_workforce_model = mm.get_step("enter_labor_force")
    in_workforce_model.run()
    stay_unemployed_list = in_workforce_model.choices.astype(int)
    predicted_share = stay_unemployed_list.sum() / stay_unemployed_list.shape[0]
    observed_stay_unemployed = orca.get_table("observed_entering_workforce").to_frame()
    target_share = observed_stay_unemployed[observed_stay_unemployed["year"]==year]["share"]
    target = target_share * stay_unemployed_list.shape[0]
    error = np.sqrt(np.mean((predicted_share.sum() - target_share)**2))
    
    while error >= 0.01:
        in_workforce_model.fitted_parameters[0] += np.log(target.sum()/stay_unemployed_list.sum())
        in_workforce_model.run()
        stay_unemployed_list = in_workforce_model.choices.astype(int)
        predicted_share = stay_unemployed_list.sum() / stay_unemployed_list.shape[0]
        error = np.sqrt(np.mean((predicted_share.sum() - target_share)**2))
    
    out_workforce_model = mm.get_step("exit_labor_force")
    out_workforce_model.run()
    exit_workforce_list = out_workforce_model.choices.astype(int)
    predicted_share = exit_workforce_list.sum() / exit_workforce_list.shape[0]
    observed_exit_workforce = orca.get_table("observed_exiting_workforce").to_frame()
    target_share = observed_exit_workforce[observed_exit_workforce["year"]==year]["share"]
    target = target_share * exit_workforce_list.shape[0]

    error = np.sqrt(np.mean((predicted_share.sum() - target_share)**2))
    while error >= 0.01:
        out_workforce_model.fitted_parameters[0] += np.log(target.sum()/exit_workforce_list.sum())
        out_workforce_model.run()
        exit_workforce_list = out_workforce_model.choices.astype(int)
        predicted_share = exit_workforce_list.sum() / exit_workforce_list.shape[0]
        error = np.sqrt(np.mean((predicted_share.sum() - target_share)**2))

    # Update labor status
    update_labor_status(persons, stay_unemployed_list, exit_workforce_list, year)


def sample_income(mean, std):
    return np.random.lognormal(mean, std)

def update_labor_status(persons, stay_unemployed_list, exit_workforce_list, year):
    """
    Function to update the worker status in persons table based
    on the labor participation model

    Args:
        persons (DataFrameWrapper): DataFrameWrapper of the persons table
        student_list (pd.Series): Pandas Series containing the output of
        the education model

    Returns:
        None
    """
    # Pull Data
    persons_df = orca.get_table("persons").local
    persons_cols = orca.get_injectable("persons_local_cols")
    households_df = orca.get_table("households").local
    households_cols = orca.get_injectable("households_local_cols")
    income_summary = orca.get_table("income_dist").local

    #####################################################
    age_intervals = [0, 20, 30, 40, 50, 65, 900]
    education_intervals = [0, 18, 22, 200]
    # Define the labels for age and education groups
    age_labels = ['lte20', '21-29', '30-39', '40-49', '50-64', 'gte65']
    education_labels = ['lte17', '18-21', 'gte22']
    # Create age and education groups with labels
    persons_df['age_group'] = pd.cut(persons_df['age'], bins=age_intervals, labels=age_labels, include_lowest=True)
    persons_df['education_group'] = pd.cut(persons_df['edu'], bins=education_intervals, labels=education_labels, include_lowest=True)
    #####################################################

    # Function to sample income from a normal distribution
    # Sample income for each individual based on their age and education group
    persons_df = persons_df.reset_index().merge(income_summary, on=['age_group', 'education_group'], how='left').set_index("person_id")
    persons_df['new_earning'] = persons_df.apply(lambda row: sample_income(row['mu'], row['sigma']), axis=1)

    persons_df["exit_workforce"] = exit_workforce_list
    persons_df["exit_workforce"].fillna(2, inplace=True)

    persons_df["remain_unemployed"] = stay_unemployed_list
    persons_df["remain_unemployed"].fillna(2, inplace=True)

    # Update education levels
    persons_df["worker"] = np.where(persons_df["exit_workforce"]==1, 0, persons_df["worker"])
    persons_df["worker"] = np.where(persons_df["remain_unemployed"]==0, 1, persons_df["worker"])

    persons_df["work_at_home"] = persons_df["work_at_home"].fillna(0)

    persons_df.loc[persons_df["exit_workforce"]==1, "earning"] = 0
    persons_df["earning"] = np.where(persons_df["remain_unemployed"]==0, persons_df["new_earning"], persons_df["earning"])

    # TODO: Similarly, do something for work from home
    agg_households = persons_df.groupby("household_id").agg(
        sum_workers = ("worker", "sum"),
        income = ("earning", "sum")
    )
    
    agg_households["hh_workers"] = np.where(
        agg_households["sum_workers"] == 0,
        "none",
        np.where(agg_households["sum_workers"] == 1, "one", "two or more"))
          
    # TODO: Make sure that the actual workers don't get restorted due to difference in indexing
    # TODO: Make sure there is a better way to do this
    #orca.get_table("households").update_col("workers", agg_households["workers"])
    #orca.get_table("households").update_col("hh_workers", agg_households["hh_workers"])
    households_df.update(agg_households)

    workers = persons_df[persons_df["worker"] == 1]
    exiting_workforce_df = orca.get_table("exiting_workforce").to_frame()
    entering_workforce_df = orca.get_table("entering_workforce").to_frame()
    if entering_workforce_df.empty:
        entering_workforce_df = pd.DataFrame(
            data={"year": [year], "count": [persons_df[persons_df["remain_unemployed"]==0].shape[0]]}
        )
    else:
        entering_workforce_df_new = pd.DataFrame(
            data={"year": [year], "count": [persons_df[persons_df["remain_unemployed"]==0].shape[0]]}
        )
        entering_workforce_df = pd.concat([entering_workforce_df, entering_workforce_df_new])

    if exiting_workforce_df.empty:
        exiting_workforce_df = pd.DataFrame(
            data={"year": [year], "count": [persons_df[persons_df["exit_workforce"]==1].shape[0]]}
        )
    else:
        exiting_workforce_df_new = pd.DataFrame(
            data={"year": [year], "count": [persons_df[persons_df["exit_workforce"]==1].shape[0]]}
        )
        exiting_workforce_df = pd.concat([exiting_workforce_df, exiting_workforce_df_new])
        
    orca.add_table("entering_workforce", entering_workforce_df)
    orca.add_table("exiting_workforce", exiting_workforce_df)
    orca.add_table("persons", persons_df[persons_cols])
    orca.add_table("households", households_df[households_cols])


@orca.step("birth_model")
def birth_model(persons, households, year):
    """
    Function to run the birth model at the household level.
    The function updates the persons table.

    Args:
        persons (DataFrameWrapper): DataFrameWrapper of the persons table
        households (DataFrameWrapper): DataFrameWrapper of the households table

    Returns:
        None
    """

    households_df = households.local
    households_df["birth"] = -99
    orca.add_table("households", households_df)
    households_df = households.local
    # persons = orca.get_table('persons')
    col_subset = ["sex", "age", "household_id", "relate"]
    persons_df = persons.to_frame(col_subset)
    ELIGIBILITY_COND = (
        (persons_df["sex"] == 2)
        & (persons_df["age"].between(14, 45))
        # & (persons_df["relate"].isin([0, 1, 13]))
    )
    ELIGIBILITY_COND_2 = (
        (persons_df["sex"] == 2)
        & (persons_df["age"] > 45)
    )
    ELIGIBILITY_COND_3 = (
        (persons_df["sex"] == 2)
        & (persons_df["age"] < 14)
    )

    # Subset of eligible households
    ELIGIBLE_HH = persons_df.loc[ELIGIBILITY_COND, "household_id"].unique()
    eligible_hh_df = households_df.loc[ELIGIBLE_HH]

    btable_elig_df = orca.get_table("btable_elig").to_frame()
    if btable_elig_df.empty:
        btable_elig_df = pd.DataFrame.from_dict({
            "year": [str(year)],
            "count":  [ELIGIBLE_HH.shape[0]]
            })
    else:
        btable_elig_df_new = pd.DataFrame.from_dict({
            "year": [str(year)],
            "count":  [ELIGIBLE_HH.shape[0]]
            })
        btable_elig_df = pd.concat([btable_elig_df, btable_elig_df_new], ignore_index=True)
    orca.add_table("btable_elig", btable_elig_df)
    # print("BIRTH ELIGIBILITY POP")
    # print(btable_elig_df)
    # Run model
    # print("Running the birth model...")
    birth = mm.get_step("birth")
    list_ids = str(eligible_hh_df.index.to_list())
    # print(len(eligible_hh_df.index.to_list()))
    birth.filters = "index in " + list_ids
    birth.out_filters = "index in " + list_ids

    birth.run()
    birth_list = birth.choices.astype(int)
    predicted_share = birth_list.sum() / eligible_hh_df.shape[0]
    observed_births = orca.get_table("observed_births_data").to_frame()
    target = observed_births[observed_births["year"]==year]["count"]
    target_share = target / eligible_hh_df.shape[0]

    error = np.sqrt(np.mean((birth_list.sum() - target)**2))
    # print("here")
    while error >= 1000:
        birth.fitted_parameters[0] += np.log(target.sum()/birth_list.sum())
        birth.run()
        birth_list = birth.choices.astype(int)
        predicted_share = birth_list.sum() / eligible_hh_df.shape[0]
        error = np.sqrt(np.mean((birth_list.sum() - target)**2))

    # breakpoint()
    # print("Eligible households >45",
    #       persons_df.loc[ELIGIBILITY_COND_2, "household_id"].unique().shape[0])
    # print("Eligible households <14",
    #       persons_df.loc[ELIGIBILITY_COND_3, "household_id"].unique().shape[0])
    # print(eligible_hh_df.shape[0], " eligible households for birth model")
    # print(birth_list.sum(), " births")
    # print("Updating persons table with newborns...")
    update_birth(persons, households, birth_list)

    # print("Updating birth metrics...")
    btable_df = orca.get_table("btable").to_frame()
    if btable_df.empty:
        btable_df = pd.DataFrame.from_dict({
            "year": [str(year)],
            "count":  [birth_list.sum()]
            })
    else:
        btable_df_new = pd.DataFrame.from_dict({
            "year": [str(year)],
            "count":  [birth_list.sum()]
            })

        btable_df = pd.concat([btable_df, btable_df_new], ignore_index=True)
    orca.add_table("btable", btable_df)


def update_birth(persons, households, birth_list):
    """
    Update the persons tables with newborns and household sizes

    Args:
        persons (DataFrameWrapper): DataFrameWrapper of the persons table
        households (DataFrameWrapper): DataFrameWrapper of the persons table
        birth_list (pd.Series): Pandas Series of the households with newborns

    Returns:
        None
    """
    persons_df = persons.local
    households_df = households.local
    households_columns = households_df.columns

    # Pull max person index from persons table
    highest_index = persons_df.index.max()

    # Check if the pop_over_time is an empty dataframe
    grave = orca.get_table("pop_over_time").to_frame()

    # If not empty, update the highest index with max index of all people
    metadata = orca.get_table("metadata").to_frame()

    max_p_id = metadata.loc["max_p_id", "value"]

    highest_index = max(max_p_id, highest_index)

    if not grave.empty:
        graveyard = orca.get_table("graveyard")
        dead_df = graveyard.to_frame(columns=["member_id", "household_id"])
        highest_dead_index = dead_df.index.max()
        highest_index = max(highest_dead_index, highest_index)

    # Get heads of households
    heads = persons_df[persons_df["relate"] == 0]

    # Get indices of households with babies
    house_indices = list(birth_list[birth_list == 1].index)

    # Initialize babies variables in the persons table.
    babies = pd.DataFrame(house_indices, columns=["household_id"])
    babies.index += highest_index + 1
    babies.index.name = "person_id"
    babies["age"] = 0
    babies["edu"] = 0
    babies["earning"] = 0
    babies["hours"] = 0
    babies["relate"] = 2
    babies["MAR"] = 5
    babies["sex"] = np.random.choice([1, 2])
    babies["student"] = 0

    babies["person_age"] = "19 and under"
    babies["person_sex"] = babies["sex"].map({1: "male", 2: "female"})
    babies["child"] = 1
    babies["senior"] = 0
    babies["dead"] = -99
    babies["person"] = 1
    babies["work_at_home"] = 0
    babies["worker"] = 0
    babies["work_block_id"] = "-1"
    babies["work_zone_id"] = "-1"
    babies["workplace_taz"] = "-1"
    babies["school_block_id"] = "-1"
    babies["school_id"] = "-1"
    babies["school_taz"] = "-1"
    babies["school_zone_id"] = "-1"
    babies["education_group"] = "lte17"
    babies["age_group"] = "lte20"
    household_races = (
        persons_df.groupby("household_id")
        .agg(num_races=("race_id", "nunique"))
        .reset_index()
        .merge(households_df["race_of_head"].reset_index(), on="household_id")
    )
    babies = babies.reset_index().merge(household_races, on="household_id")
    babies["race_id"] = np.where(babies["num_races"] == 1, babies["race_of_head"], 9)
    babies["race"] = babies["race_id"].map(
        {
            1: "white",
            2: "black",
            3: "other",
            4: "other",
            5: "other",
            6: "other",
            7: "other",
            8: "other",
            9: "other",
        }
    )
    babies = (
        babies.reset_index()
        .merge(
            heads[["hispanic", "hispanic.1", "p_hispanic", "household_id"]],
            on="household_id",
        )
        .set_index("person_id")
    )

    # Add counter for member_id to not overlap from dead people for households
    if not grave.empty:
        all_people = pd.concat([grave, persons_df[["member_id", "household_id"]]])
    else:
        all_people = persons_df[["member_id", "household_id"]]
    max_member_id = all_people.groupby("household_id").agg({"member_id": "max"})
    max_member_id += 1
    babies = (
        babies.reset_index()
        .merge(max_member_id, left_on="household_id", right_index=True)
        .set_index("person_id")
    )
    households_babies = households_df.loc[house_indices]
    households_babies["hh_children"] = "yes"
    households_babies["persons"] += 1
    households_babies["gt2"] = np.where(households_babies["persons"] >= 2, 1, 0)
    households_babies["hh_size"] = np.where(
        households_babies["persons"] == 1,
        "one",
        np.where(
            households_babies["persons"] == 2,
            "two",
            np.where(households_babies["persons"] == 3, "three", "four or more"),
        ),
    )

    # Update the households table
    households_df.update(households_babies[households_df.columns])
    # Contactenate the final result
    combined_result = pd.concat([persons_df, babies])
    persons_local_cols = orca.get_injectable("persons_local_cols")
    households_local_cols = orca.get_injectable("households_local_cols")

    orca.add_table("persons", combined_result.loc[:, persons_local_cols])
    orca.add_table("households", households_df.loc[:, households_local_cols])
    metadata = orca.get_table("metadata").to_frame()
    max_hh_id = metadata.loc["max_hh_id", "value"]
    max_p_id = metadata.loc["max_p_id", "value"]
    if households_df.index.max() > max_hh_id:
        metadata.loc["max_hh_id", "value"] = households_df.index.max()
    if combined_result.index.max() > max_p_id:
        metadata.loc["max_p_id", "value"] = combined_result.index.max()
    
    orca.add_table("metadata", metadata)
    # orca.add_injectable("max_p_id", max(highest_index, orca.get_injectable("max_p_id")))


def update_households_after_kids(persons, households, kids_moving):
    """
    Add and update households after kids move out.

    Args:
        persons (DataFrameWrapper): DataFrameWrapper of persons table
        households (DataFrameWrapper): DataFrameWrapper of households table
        kids_moving (pd.Series): Pandas Series of kids moving out of household

    Returns:
        None
    """
    # print("Updating households...")
    persons_df = orca.get_table("persons").local

    persons_local_cols = persons_df.columns

    households_df = orca.get_table("households").local
    households_local_cols = households_df.columns
    hh_id = (
        orca.get_table("households").to_frame(columns=["lcm_county_id"]).reset_index()
    )

    persons_df = (
        persons_df.reset_index()
        .merge(hh_id, on=["household_id"])
        .set_index("person_id")
    )

    persons_df["moveoutkid"] = kids_moving

    highest_index = households_df.index.max()
    metadata = orca.get_table("metadata").to_frame()
    max_hh_id = metadata.loc["max_hh_id", "value"]    
    current_max_household_id = max(max_hh_id, highest_index)

    kids_leaving = persons_df[persons_df["moveoutkid"] == 1]["household_id"].unique()
    single_per_household = (
        persons_df[persons_df["household_id"].isin(kids_leaving)]
        .groupby("household_id")
        .size()
        == 1
    )
    single_per_nonmoving = single_per_household[
        single_per_household == True
    ].index.unique()
    persons_df["moveoutkid"] = np.where(
        persons_df["household_id"].isin(single_per_nonmoving),
        0,
        persons_df["moveoutkid"],
    )

    kids_leaving = persons_df[persons_df["moveoutkid"] == 1]["household_id"].unique()
    entire_household_moving = (
        persons_df[
            persons_df.index.isin(
                persons_df[persons_df["moveoutkid"] == 1].index.unique()
            )
        ]
        .groupby("household_id")
        .size()
        == persons_df[persons_df["household_id"].isin(kids_leaving)]
        .groupby("household_id")
        .size()
    )
    hh_nonmoving = entire_household_moving[
        entire_household_moving == True
    ].index.unique()
    persons_df["moveoutkid"] = np.where(
        persons_df["household_id"].isin(hh_nonmoving), 0, persons_df["moveoutkid"]
    )

    persons_df.loc[persons_df["moveoutkid"] == 1, "household_id"] = (
        np.arange(persons_df["moveoutkid"].sum()) + current_max_household_id + 1
    )

    new_hh = persons_df.loc[persons_df["moveoutkid"] == 1].copy()

    persons_df = persons_df.drop(persons_df[persons_df["moveoutkid"] == 1].index)
    # add to orca
    persons_df["person"] = 1
    persons_df["is_head"] = np.where(persons_df["relate"] == 0, 1, 0)
    persons_df["race_head"] = persons_df["is_head"] * persons_df["race_id"]
    persons_df["age_head"] = persons_df["is_head"] * persons_df["age"]
    persons_df["hispanic_head"] = persons_df["is_head"] * persons_df["hispanic"]
    persons_df["child"] = np.where(persons_df["relate"].isin([2, 3, 4, 7, 9, 14]), 1, 0)
    persons_df["senior"] = np.where(persons_df["age"] >= 65, 1, 0)
    persons_df["age_gt55"] = np.where(persons_df["age"] >= 55, 1, 0)

    persons_df = persons_df.sort_values("relate")

    old_agg_household = persons_df.groupby("household_id").agg(
        income=("earning", "sum"),
        race_of_head=("race_head", "sum"),
        age_of_head=("age_head", "sum"),
        workers=("worker", "sum"),
        hispanic_status_of_head=("hispanic_head", "sum"),
        seniors=("senior", "sum"),
        persons=("person", "sum"),
        age_gt55=("age_gt55", "sum"),
        children=("child", "sum"),
    )
    old_agg_household["hh_age_of_head"] = np.where(
        old_agg_household["age_of_head"] < 35,
        "lt35",
        np.where(old_agg_household["age_of_head"] < 65, "gt35-lt65", "gt65"),
    )
    old_agg_household["hh_race_of_head"] = np.where(
        old_agg_household["race_of_head"] == 1,
        "white",
        np.where(
            old_agg_household["race_of_head"] == 2,
            "black",
            np.where(old_agg_household["race_of_head"].isin([6, 7]), "asian", "other"),
        ),
    )
    old_agg_household["hispanic_head"] = np.where(
        old_agg_household["hispanic_status_of_head"] == 1, "yes", "no"
    )
    old_agg_household["hh_size"] = np.where(
        old_agg_household["persons"] == 1,
        "one",
        np.where(
            old_agg_household["persons"] == 2,
            "two",
            np.where(old_agg_household["persons"] == 3, "three", "four or more"),
        ),
    )
    old_agg_household["hh_children"] = np.where(
        old_agg_household["children"] >= 1, "yes", "no"
    )
    old_agg_household["hh_income"] = np.where(
        old_agg_household["income"] < 30000,
        "lt30",
        np.where(
            old_agg_household["income"] < 60,
            "gt30-lt60",
            np.where(
                old_agg_household["income"] < 100,
                "gt60-lt100",
                np.where(old_agg_household["income"] < 150, "gt100-lt150", "gt150"),
            ),
        ),
    )
    old_agg_household["hh_workers"] = np.where(
        old_agg_household["workers"] == 0,
        "none",
        np.where(old_agg_household["workers"] == 1, "one", "two or more"),
    )
    old_agg_household["hh_seniors"] = np.where(
        old_agg_household["seniors"] >= 1, "yes", "no"
    )
    old_agg_household["gt55"] = np.where(old_agg_household["age_gt55"] > 0, 1, 0)
    old_agg_household["gt2"] = np.where(old_agg_household["persons"] > 2, 1, 0)

    households_df.update(old_agg_household)

    new_hh["person"] = 1
    new_hh["is_head"] = np.where(new_hh["relate"] == 0, 1, 0)
    new_hh["race_head"] = new_hh["is_head"] * new_hh["race_id"]
    new_hh["age_head"] = new_hh["is_head"] * new_hh["age"]
    new_hh["hispanic_head"] = new_hh["is_head"] * new_hh["hispanic"]
    new_hh["child"] = np.where(new_hh["relate"].isin([2, 3, 4, 14]), 1, 0)
    new_hh["senior"] = np.where(new_hh["age"] >= 65, 1, 0)
    new_hh["age_gt55"] = np.where(new_hh["age"] >= 55, 1, 0)
    new_hh["car"] = np.random.choice([0, 1, 2], size=new_hh.shape[0])

    new_hh = new_hh.sort_values("relate")

    agg_households = new_hh.groupby("household_id").agg(
        income=("earning", "sum"),
        race_of_head=("race_head", "sum"),
        age_of_head=("age_head", "sum"),
        workers=("worker", "sum"),
        hispanic_status_of_head=("hispanic_head", "sum"),
        seniors=("senior", "sum"),
        lcm_county_id=("lcm_county_id", "first"),
        persons=("person", "sum"),
        age_gt55=("age_gt55", "sum"),
        cars=("car", "sum"),
        children=("child", "sum"),
    )
    agg_households["serialno"] = "-1"
    agg_households["tenure"] = np.random.choice(
        households_df["tenure"].unique(), size=agg_households.shape[0]
    )  # Needs changed
    agg_households["recent_mover"] = np.random.choice(
        households_df["recent_mover"].unique(), size=agg_households.shape[0]
    )
    agg_households["sf_detached"] = np.random.choice(
        households_df["sf_detached"].unique(), size=agg_households.shape[0]
    )
    agg_households["hh_age_of_head"] = np.where(
        agg_households["age_of_head"] < 35,
        "lt35",
        np.where(agg_households["age_of_head"] < 65, "gt35-lt65", "gt65"),
    )
    agg_households["hh_race_of_head"] = np.where(
        agg_households["race_of_head"] == 1,
        "white",
        np.where(
            agg_households["race_of_head"] == 2,
            "black",
            np.where(agg_households["race_of_head"].isin([6, 7]), "asian", "other"),
        ),
    )
    agg_households["hispanic_head"] = np.where(
        agg_households["hispanic_status_of_head"] == 1, "yes", "no"
    )
    agg_households["hh_size"] = np.where(
        agg_households["persons"] == 1,
        "one",
        np.where(
            agg_households["persons"] == 2,
            "two",
            np.where(agg_households["persons"] == 3, "three", "four or more"),
        ),
    )
    agg_households["hh_cars"] = np.where(
        agg_households["cars"] == 0,
        "none",
        np.where(agg_households["cars"] == 1, "one", "two or more"),
    )
    agg_households["hh_children"] = np.where(
        agg_households["children"] >= 1, "yes", "no"
    )
    agg_households["hh_income"] = np.where(
        agg_households["income"] < 30000,
        "lt30",
        np.where(
            agg_households["income"] < 60,
            "gt30-lt60",
            np.where(
                agg_households["income"] < 100,
                "gt60-lt100",
                np.where(agg_households["income"] < 150, "gt100-lt150", "gt150"),
            ),
        ),
    )
    agg_households["hh_workers"] = np.where(
        agg_households["workers"] == 0,
        "none",
        np.where(agg_households["workers"] == 1, "one", "two or more"),
    )
    agg_households["tenure_mover"] = np.random.choice(
        households_df["tenure_mover"].unique(), size=agg_households.shape[0]
    )
    agg_households["hh_seniors"] = np.where(agg_households["seniors"] >= 1, "yes", "no")
    agg_households["block_id"] = np.random.choice(
        households_df["block_id"].unique(), size=agg_households.shape[0]
    )
    agg_households["gt55"] = np.where(agg_households["age_gt55"] > 0, 1, 0)
    agg_households["gt2"] = np.where(agg_households["persons"] > 2, 1, 0)
    agg_households["hh_type"] = 0  # CHANGE THIS

    households_df["birth"] = -99
    households_df["divorced"] = -99

    agg_households["birth"] = -99
    agg_households["divorced"] = -99

    households_df = pd.concat(
        [households_df[households_local_cols], agg_households[households_local_cols]]
    )
    persons_df = pd.concat([persons_df[persons_local_cols], new_hh[persons_local_cols]])
    # print(households_df["hh_size"].unique())
    # add to orca
    orca.add_table("households", households_df[households_local_cols])
    orca.add_table("persons", persons_df[persons_local_cols])
    # orca.add_injectable(
    #     "max_hh_id", max(households_df.index.max(), orca.get_injectable("max_hh_id"))
    # )

    metadata = orca.get_table("metadata").to_frame()
    max_hh_id = metadata.loc["max_hh_id", "value"]
    max_p_id = metadata.loc["max_p_id", "value"]
    if households_df.index.max() > max_hh_id:
        metadata.loc["max_hh_id", "value"] = households_df.index.max()
    if persons_df.index.max() > max_p_id:
        metadata.loc["max_p_id", "value"] = persons_df.index.max()
    orca.add_table("metadata", metadata)

    # print("Updating kids moving metrics...")
    kids_moving_table = orca.get_table("kids_move_table").to_frame()
    if kids_moving_table.empty:
        kids_moving_table = pd.DataFrame(
            [kids_moving_table.sum()], columns=["kids_moving_out"]
        )
    else:
        new_kids_moving_table = pd.DataFrame(
            {"kids_moving_out": kids_moving_table.sum()}
        )
        kids_moving_table = pd.concat([kids_moving_table, 
                                       new_kids_moving_table],
                                      ignore_index=True)
    orca.add_table("kids_move_table", kids_moving_table)


def extract_students(persons):
    edu_levels = np.arange(3, 16).astype(float)
    STUDENTS_CONDITION = (persons["student"]==1) & (persons["edu"].isin(edu_levels))
    students_df = persons[STUDENTS_CONDITION].copy()
    students_df["STUDENT_SCHOOL_LEVEL"] = np.where(students_df["edu"]<=8, "ELEMENTARY",
                                np.where(students_df["edu"]<=11, "MIDDLE", "HIGH"))
    students_df = students_df.reset_index()
    return students_df

def create_student_groups(students_df):
    students_df = students_df[students_df["DET_DIST_TYPE"]==students_df["STUDENT_SCHOOL_LEVEL"]].reset_index(drop=True)
    student_groups = students_df.groupby(['household_id', 'GEOID10_SD', 'STUDENT_SCHOOL_LEVEL'])['person_id'].apply(list).reset_index(name='students')
    student_groups["DISTRICT_LEVEL"] = student_groups.apply(lambda row: (row['GEOID10_SD'], row['STUDENT_SCHOOL_LEVEL']), axis=1)
    student_groups["size_student_group"] = [len(x) for x in student_groups["students"]]
    student_groups = student_groups.sort_values(by=["GEOID10_SD", "STUDENT_SCHOOL_LEVEL"]).reset_index(drop=True)
    student_groups["CUM_STUDENTS"] = student_groups.groupby(["GEOID10_SD", "STUDENT_SCHOOL_LEVEL"])["size_student_group"].cumsum()
    return student_groups

def assign_schools(student_groups, blocks_districts, schools_df):
    assigned_students_list = []
    for tuple in blocks_districts["DISTRICT_LEVEL"].unique():
        # print("district")
        SCHOOL_DISTRICT = tuple[0]
        SCHOOL_LEVEL = tuple[1]
        schools_pool = schools_df[(schools_df["SCHOOL_LEVEL"]==SCHOOL_LEVEL) &\
                                (schools_df["GEOID10_SD"]==SCHOOL_DISTRICT)].copy()
        student_pool = student_groups[(student_groups["STUDENT_SCHOOL_LEVEL"]==SCHOOL_LEVEL) &\
                        (student_groups["GEOID10_SD"]==SCHOOL_DISTRICT)].copy()
        student_pool = student_pool.sample(frac = 1)
        # Iterate over schools_df
        for idx, row in schools_pool.iterrows():
            # print("school")
            # Get the pool of students for the district and level of the school
            SCHOOL_LEVEL = row["SCHOOL_LEVEL"]
            SCHOOL_DISTRICT = row["GEOID10_SD"]
            
            # Calculate the number of students to assign
            n_students = min(student_pool["size_student_group"].sum(), row['CAP_TOTAL'])
            student_pool["CUM_STUDENTS"] = student_pool["size_student_group"].cumsum()
            student_pool["ASSIGNED"] = np.where(student_pool["CUM_STUDENTS"]<=n_students, 1, 0)
            # Randomly sample students without replacement
            assigned_students = student_pool[student_pool["CUM_STUDENTS"]<=n_students].copy()
            assigned_students["SCHOOL_ID"] = row["SCHOOL_ID"]
            assigned_students_list.append(assigned_students)
            student_pool = student_pool[student_pool["ASSIGNED"]==0].copy()
    return assigned_students_list

def create_results_table(students_df, assigned_students_list, year):
    assigned_students_df = pd.concat(assigned_students_list)[["students", "household_id", "GEOID10_SD", "STUDENT_SCHOOL_LEVEL", "SCHOOL_ID"]]
    assigned_students_df = assigned_students_df.explode("students").rename(columns={"students": "person_id",
                                                                                    "SCHOOL_ID": "school_id",})
    school_assignment_df = students_df[["person_id"]].merge(assigned_students_df[["person_id", "school_id", "GEOID10_SD"]], on="person_id", how='left').fillna("-1")
    school_assignment_df["year"] = year
    return school_assignment_df

@orca.step("mlcm_postprocessing")
def mlcm_postprocessing(persons):
    # breakpoint()
    # SCHOOL
    persons_df = orca.get_table("persons").local
    persons_df = persons_df.reset_index()
    # schools_df = orca.get_table("schools").to_frame()
    # school_assignment_df = orca.get_table("school_locations").to_frame()

    geoid_to_zone = orca.get_table("geoid_to_zone").to_frame()
    geoid_to_zone["work_block_id"] = geoid_to_zone["GEOID10"].copy()
    # geoid_to_zone["school_taz"] = geoid_to_zone["zone_id"].copy()
    geoid_to_zone["workplace_taz"] = geoid_to_zone["zone_id"].copy()

    # schools_df = schools_df.drop_duplicates(subset=["school_id"], keep="first")
    # school_assignment_df = school_assignment_df.merge(schools_df[["school_id", "GEOID10"]], on=["school_id"], how='left')
    # school_assignment_df["GEOID10"] = school_assignment_df["GEOID10"].fillna("-1")
    # school_assignment_df["school_block_id"] = school_assignment_df["GEOID10"].copy().fillna("-1")
    # school_assignment_df = school_assignment_df.merge(geoid_to_zone[["GEOID10", "school_taz"]], on=["GEOID10"], how='left')
    
    # # breakpoint()    
    # persons_df = persons_df.merge(school_assignment_df[["person_id", "school_id", "school_block_id", "school_taz"]], on=["person_id"], suffixes=('', '_replace'), how="left")
    # persons_df["school_taz"] = persons_df["school_taz_replace"].copy().fillna("-1")
    # persons_df["school_id"] = persons_df["school_id_replace"].copy().fillna("-1")
    # persons_df["school_block_id"] = persons_df["school_block_id_replace"].copy().fillna("-1")

    # WORK POSTPROCESSING
    # work_locations = orca.get_table('work_locations').to_frame()
    # work_locations["GEOID10"] = work_locations["work_block_id"].copy().astype("str")
    # work_locations = work_locations.merge(geoid_to_zone[["GEOID10", "workplace_taz"]], on=["GEOID10"], how='left')
    # work_locations["workplace_taz"] = work_locations["workplace_taz"].copy().fillna("-1")

    persons_df = persons_df.merge(geoid_to_zone[["work_block_id", "workplace_taz"]], on=["work_block_id"], suffixes=('', '_replace'), how="left")
    persons_df["workplace_taz"] = persons_df["workplace_taz_replace"].copy().fillna("-1")
    persons_df["work_zone_id"] = persons_df["workplace_taz"].copy()
    persons_df = persons_df.set_index("person_id")

    persons_cols = orca.get_injectable("persons_local_cols")

    orca.add_table("persons", persons_df[persons_cols])


@orca.step("school_location")
def school_location(persons, households, year):
    persons_df = orca.get_table("persons").local
    students_df = extract_students(persons_df)
    schools_df = orca.get_table("schools").to_frame()
    blocks_districts = orca.get_table("blocks_districts").to_frame()
    households_df = orca.get_table("households").local
    households_df = households_df.reset_index()
    # breakpoint()
    # At this breakpoint, figure out how many students don't have school locations
    households_districts = households_df[["household_id", "block_id"]].merge(blocks_districts, left_on="block_id", right_on="GEOID10")
    students_df = students_df.merge(households_districts.reset_index(), on="household_id")
    students_df = students_df[students_df["STUDENT_SCHOOL_LEVEL"]==students_df["DET_DIST_TYPE"]].copy()
    student_groups = create_student_groups(students_df)

    assigned_students_list = assign_schools(student_groups,
                                            blocks_districts,
                                            schools_df)

    school_assignment_df = create_results_table(students_df, assigned_students_list, year)
    # breakpoint()
    # At this breakpoint, figure out how many still don't have an assignment
    orca.add_table("school_locations", school_assignment_df[["person_id", "school_id"]])

@orca.step("kids_moving_model")
def kids_moving_model(persons, households):
    """
    Running the kids moving model and updating household
    stats.

    Args:
        persons (DataFrameWrapper): DataFrameWrapper of the persons table
        households (DataFrameWrapper): DataFrameWrapper of the households table

    Returns:
        None
    """
    persons_df = orca.get_table("persons").local
    persons_df["kid_moves"] = -99
    orca.add_table("persons", persons_df)

    # print("Running the kids moving model...")
    kids_moving_model = mm.get_step("kids_move")
    kids_moving_model.run()
    kids_moving = kids_moving_model.choices.astype(int)

    update_households_after_kids(persons, households, kids_moving)


@orca.step("marriage_model")
def marriage_model(persons, households):
    """Function to run the marriage model, pair individuals, and
    replace marriage status and households in the persons and households
    tables.

    Args:
        persons (DataFrameWrapper): DataFrameWrapper of persons table
        households (DataFrameWrapper): DataFrameWrapper of households table
    """
    household_cols = households.local_columns
    household_df = households.local
    persons_cols = persons.local_columns
    persons_local_cols = persons.local_columns

    marriage_model = orca.get_injectable("marriage_model")
    marriage_coeffs = pd.DataFrame(marriage_model["model_coeffs"])
    marriage_variables = pd.DataFrame(marriage_model["spec_names"])
    model_columns = marriage_variables[0].values.tolist()
    vars = persons_local_cols + model_columns
    persons_df = persons.to_frame(columns=vars)
    # get persons cohabitating and heads of their households
    COHABS_PERSONS = persons_df["relate"] == 13
    cohab_persons_df = persons_df.loc[COHABS_PERSONS].copy()
    COHABS_HOUSEHOLDS = cohab_persons_df["household_id"].unique()
    COHABS_HEADS = (persons_df["household_id"].isin(COHABS_HOUSEHOLDS)) & (persons_df["relate"] == 0)
    cohab_heads_df = persons_df.loc[COHABS_HEADS].copy()
    all_cohabs_df = pd.concat([cohab_heads_df, cohab_persons_df])
    all_cohabs_df["cohab"] = 1
    # Get Single People
    SINGLE_COND = (persons_df["MAR"] != 1) & (persons_df["age"] > 16)
    single_df = persons_df.loc[SINGLE_COND].copy()
    data = single_df.join(all_cohabs_df[["cohab"]], how="left")
    data = data.loc[data["cohab"] != 1].copy()
    data.drop(columns="cohab", inplace=True)
    data = data.loc[:, model_columns].copy()
    ###############################################################
    # print("Running marriage model...")
    marriage_list = simulation_mnl(data, marriage_coeffs)
    random_match = orca.get_injectable("random_match")
    if random_match==True:
        update_married_households_random(persons, households, marriage_list)
    else:
        update_married_households(persons, households, marriage_list)
        

def fix_erroneous_households(persons, households):
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

def update_married_households_random(persons, households, marriage_list):
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
    female_mar = relevant[(relevant["new_mar"] == 2) & (relevant["person_sex"] == "female")].sample(min_mar)
    male_mar = relevant[(relevant["new_mar"] == 2) & (relevant["person_sex"] == "male")].sample(min_mar)
    female_coh = relevant[(relevant["new_mar"] == 1) & (relevant["person_sex"] == "female")].sample(min_cohab)
    male_coh = relevant[(relevant["new_mar"] == 1) & (relevant["person_sex"] == "male")].sample(min_cohab)
    
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
    married = married.sort_values(by=["number"])
    cohabitate = cohabitate.sort_values(by=["number"])

    married["household_group"] = np.repeat(np.arange(len(married.index) / 2), 2)
    cohabitate["household_group"] = np.repeat(np.arange(len(cohabitate.index) / 2), 2)

    married = married.sort_values(by=["household_group", "earning"], ascending=[True, False])
    cohabitate = cohabitate.sort_values(by=["household_group", "earning"], ascending=[True, False])

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
    households_restructuring.loc[households_restructuring.groupby(["household_id"]).head(1).index, "relate"] = 0

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

@orca.step("cohabitation_model")
def cohabitation_model(persons, households):
    """Function to run the cohabitation to X model.
    The function runs the model and updates households.

    Args:
        persons (DataFrameWrapper): DataFrameWrapper of the persons table
        households (DataFrameWrapper): DataFrameWrapper of the households table
    """
    persons_df = persons.local
    hh_df = households.to_frame(columns=["lcm_county_id"])
    hh_df.reset_index(inplace=True)
    hh_local_cols = orca.get_table("households").local_columns
    
    cohabitation_model = orca.get_injectable("cohabitation_model")
    cohabitation_coeffs = pd.DataFrame(cohabitation_model["model_coeffs"])
    cohabitation_variables = pd.DataFrame(cohabitation_model["spec_names"])
    model_columns = cohabitation_variables[0].values.tolist()
    vars = np.unique(np.array(hh_local_cols + model_columns))
    persons_df = persons.local
    ELIGIBLE_HOUSEHOLDS = (
        persons_df[persons_df["relate"] == 13]["household_id"].unique().astype(int)
    )
    # households_df = orca.get_table("households").to_frame(columns=model_columns)
    data = (
        orca.get_table("households")
        .to_frame(columns=model_columns)
        .loc[ELIGIBLE_HOUSEHOLDS, model_columns]
    )
    # Run Model
    # print("Running cohabitation model...")
    # breakpoint()
    cohabitate_x_list = simulation_mnl(data, cohabitation_coeffs)
    update_cohabitating_households(persons, households, cohabitate_x_list)


def update_divorce(divorce_list):
    """
    Updating stats for divorced households

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

    households_df.loc[divorce_list.index,"divorced"] = divorce_list

    divorce_households = households_df[households_df["divorced"] == 1].copy()
    DIVORCED_HOUSEHOLDS_ID = divorce_households.index.to_list()

    sizes = persons_df[persons_df["household_id"].isin(divorce_list.index) & (persons_df["relate"].isin([0, 1]))].groupby("household_id").size()

    # print("Sizes not 2: ", sizes[sizes!=2].shape[0])

    # print("divorced households: ", len(DIVORCED_HOUSEHOLDS_ID))
    # print("")
    persons_divorce = persons_df[
        persons_df["household_id"].isin(divorce_households.index)
    ].copy()
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
        staying_house["MAR"] == 1, 3, staying_house["MAR"]
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


@orca.step("households_reorg")
def households_reorg(persons, households, year):
    #
    # MARRIAGE MODEL
    household_cols = households.local_columns
    household_df = households.local
    persons_cols = persons.local_columns
    persons_local_cols = persons.local_columns

    marriage_model = orca.get_injectable("marriage_model")
    marriage_coeffs = pd.DataFrame(marriage_model["model_coeffs"])
    marriage_variables = pd.DataFrame(marriage_model["spec_names"])
    model_columns = marriage_variables[0].values.tolist()
    vars = persons_local_cols + model_columns
    persons_df = persons.to_frame(columns=vars)
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
    data = data.loc[:, model_columns].copy()
    ###############################################################
    # print("Running marriage model...")
    # breakpoint()
    marriage_list = simulation_mnl(data, marriage_coeffs)
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
    persons_df = persons.local
    hh_df = households.to_frame(columns=["lcm_county_id"])
    hh_df.reset_index(inplace=True)
    hh_local_cols = orca.get_table("households").local_columns
    
    cohabitation_model = orca.get_injectable("cohabitation_model")
    cohabitation_coeffs = pd.DataFrame(cohabitation_model["model_coeffs"])
    cohabitation_variables = pd.DataFrame(cohabitation_model["spec_names"])
    model_columns = cohabitation_variables[0].values.tolist()
    vars = np.unique(np.array(hh_local_cols + model_columns))
    persons_df = persons.local
    ELIGIBLE_HOUSEHOLDS = (
        persons_df[(persons_df["relate"] == 13) & \
                   (persons_df["MAR"]!=1) & \
                   ((persons_df["age"]>=15))]["household_id"].unique().astype(int)
    )
    # households_df = orca.get_table("households").to_frame(columns=model_columns)
    # print(model_columns)
    data = (
        households.to_frame(columns=model_columns)
        .loc[ELIGIBLE_HOUSEHOLDS, model_columns]
    )
    # Run Model
    # print("Running cohabitation model...")
    cohabitate_x_list = simulation_mnl(data, cohabitation_coeffs)
    # print("Cohabitation outcomes:")
    # print(cohabitate_x_list.value_counts())
    
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



@orca.step("print_marr_stats")
def print_marr_stats():
    persons_stats = orca.get_table("persons").local
    persons_stats = persons_stats[persons_stats["age"]>=15]
    print(persons_stats["MAR"].value_counts().sort_values())

# -----------------------------------------------------------------------------------------
# TRANSITION
# -----------------------------------------------------------------------------------------


@orca.step('household_transition')
def household_transition(households, persons, year, metadata):
    # breakpoint()
    # at this breakpoint, look at the persons table
    linked_tables = {'persons': (persons, 'household_id')}
    if ('annual_household_control_totals' in orca.list_tables()) and ('use_database_control_totals' not in orca.list_injectables()):
        control_totals = orca.get_table('annual_household_control_totals').to_frame()
        full_transition(households, control_totals, 'total', year, 'block_id', linked_tables=linked_tables)
    elif ('household_growth_rate' in orca.list_injectables()) and ('use_database_control_totals' not in orca.list_injectables()):
        rate = orca.get_injectable('household_growth_rate')
        simple_transition(households, rate, 'block_id', set_year_built=True, linked_tables=linked_tables)
    elif 'hsize_ct' in orca.list_tables():
        control_totals = orca.get_table('hsize_ct').to_frame()
        full_transition(households, control_totals, 'total_number_of_households', year, 'block_id')
    else:
        control_totals = orca.get_table('hct').to_frame()
        if 'hh_type' in control_totals.columns:
            if control_totals[control_totals.index == year].hh_type.min() == -1:
                control_totals = control_totals[['total_number_of_households']]
        full_transition(households, control_totals, 'total_number_of_households', year, 'block_id', linked_tables=linked_tables)
    households_df = orca.get_table('households').local
    households_df.loc[households_df['block_id'] == "-1", 'lcm_county_id'] = "-1"
    households_df.index.rename('household_id', inplace=True)
    persons_df = orca.get_table('persons').local
    # persons = persons.loc[persons['household_id'].isin(households.index.unique())]
    orca.add_table('households', households_df)
    orca.add_table('persons', persons_df)
    # orca.add_injectable(
    #     'max_hh_id', max(orca.get_injectable("max_hh_id"), households.index.max())
    # )
    metadata_df = orca.get_table('metadata').to_frame()
    max_hh_id = metadata_df.loc['max_hh_id', 'value']
    max_p_id = metadata_df.loc['max_p_id', 'value']
    if households_df.index.max() > max_hh_id:
        metadata_df.loc['max_hh_id', 'value'] = households_df.index.max()
    if persons_df.index.max() > max_p_id:
        metadata_df.loc['max_p_id', 'value'] = persons_df.index.max()
    orca.add_table('metadata', metadata_df)
    # breakpoint()

@orca.step("job_transition")
def job_transition(jobs, year):
    if ("annual_employment_control_totals" in orca.list_tables()) and (
        "use_database_control_totals" not in orca.list_injectables()
    ):
        control_totals = orca.get_table("annual_employment_control_totals").to_frame()
        full_transition(jobs, control_totals, "total", year, "block_id")
    elif ("employment_growth_rate" in orca.list_injectables()) and (
        "use_database_control_totals" not in orca.list_injectables()
    ):
        rate = orca.get_injectable("employment_growth_rate")
        simple_transition(jobs, rate, "block_id", set_year_built=True)
    else:
        control_totals = orca.get_table("ect").to_frame()
        if "agg_sector" in control_totals.columns:
            if control_totals[control_totals.index == year].agg_sector.min() == -1:
                control_totals = control_totals[["total_number_of_jobs"]]
        full_transition(jobs, control_totals, "total_number_of_jobs", year, "block_id")
    jobs = orca.get_table("jobs").local
    jobs.loc[jobs["block_id"] == "-1", "lcm_county_id"] = "-1"
    jobs.index.rename("job_id", inplace=True)
    orca.add_table("jobs", jobs)


@orca.step("supply_transition")
def supply_transition(households, residential_units, vacancy):
    agents = len(households)
    agent_spaces = len(residential_units)
    if "residential_vacancy_rate" in orca.list_injectables():
        target_vacancy = orca.get_injectable("residential_vacancy_rate")
    else:
        target_vacancy = vacancy
    target = developer.Developer.compute_units_to_build(
        agents, agent_spaces, target_vacancy
    )
    if target > 0:
        growth_rate = target * 1.0 / agent_spaces
        print("Growth rate implied by target vacancy rate: %s" % growth_rate)
        simple_transition(
            residential_units, growth_rate, "block_id", set_year_built=True
        )

        units = orca.get_table("residential_units").local
        units.index.rename("unit_id", inplace=True)
        units.loc[units["block_id"] == "-1", "lcm_county_id"] = "-1"
        orca.add_table("residential_units", units)
    else:
        print(
            "No new residential units to construct; current vacancy > target vacancy (%s)."
            % vacancy
        )


def full_transition(
    agents,
    ct,
    totals_column,
    year,
    location_fname,
    linked_tables=None,
    accounting_column=None,
    set_year_built=False,
):
    """
    Run a transition model based on control totals specified in the usual UrbanSim way
    Parameters
    ----------
    agents : DataFrameWrapper
        Table to be transitioned
    agent_controls : DataFrameWrapper
        Table of control totals
    totals_column : str
        String indicating the agent_controls column to use for totals.
    year : int
        The year, which will index into the controls
    location_fname : str
        The field name in the resulting dataframe to set to -1 (to unplace
        new agents)
    linked_tables: dict, optional
        Sets the tables linked to new or removed agents to be updated with
        dict of {'table_name':(DataFrameWrapper, 'link_id')}
    accounting_column : str, optional
        Name of column with accounting totals/quantities to apply toward the
        control. If not provided then row counts will be used for accounting.
    set_year_built: boolean
        Indicates whether to update 'year_built' columns with current
        simulation year
    Returns
    -------
    Nothing
    """
    print("Running full transition")
    if "agg_sector" in ct.columns:
        ct["agg_sector"] = ct["agg_sector"].astype("str")
    add_cols = [col for col in ct.columns if col != totals_column]
    add_cols = [col for col in add_cols if col not in agents.local.columns]
    agnt = agents.to_frame(list(agents.local.columns) + add_cols)
    print("Total agents before transition: {}".format(len(agnt)))
    idx_name = agnt.index.name
    if agents.name == "households":
        agnt = agnt.reset_index()
        hh_sizes = agnt["hh_size"].unique()
        # print(agnt["hh_size"].unique())
        updated = pd.DataFrame()
        added = pd.Index([])
        copied = pd.Index([])
        removed = pd.Index([])
        ct["lcm_county_id"] = ct["lcm_county_id"].astype(str)
        max_hh_id = agnt.index.max()
        for size in hh_sizes:
            # print(size)
            agnt_sub = agnt[agnt["hh_size"] == size].copy()
            # print(agnt_sub.shape[0])
            ct_sub = ct[ct["hh_size"] == size].copy()
            # print(ct_sub.shape[0])
            tran = transition.TabularTotalsTransition(ct_sub, totals_column, accounting_column)
            # print(ct_sub.dtypes)
            updated_sub, added_sub, copied_sub, removed_sub = tran.transition(agnt_sub, year)
            updated_sub.loc[added_sub, location_fname] = "-1"
            # print("Edits shape:")
            # print("==================")
            # print(updated_sub.shape)
            # print(added_sub.shape)
            # print(copied_sub.shape)
            # print(removed_sub.shape)
            # print("==================")
            # max_hh_id = max(agnt.index.max(), updated_sub.index.max())
            # breakpoint()
            if updated.empty:
                updated = updated_sub.copy()
                # print("updated_sub index:", updated_sub.index.name)
                # print("updated_sub has duplicates:", updated.index.has_duplicates)
            else:
                # print("updated_sub index: ", updated_sub.index.name)
                # print("updated before index:", updated.index.name)
                updated = pd.concat([updated, updated_sub])
                # print("updated after index:", updated.index.name)
                # print("Updated Shape after concat:", updated.shape)

            if added.empty:
                added = added_sub.copy()
            else:
                added = added.append(added_sub)
            if copied.empty:
                copied = copied_sub.copy()
            else:
                copied = copied.append(copied_sub)
            if removed.empty:
                removed = removed_sub.copy()
            else:
                # print(removed)
                # print(type(removed_sub))
                removed = removed.append(removed_sub)
        # breakpoint()
        # removed_df = agnt.index.isin(removed)
        # updated = agnt[~removed_df].copy()
        # new_agnts = agnt.loc[copied].copy()

        # updated = pd.concat([updated, new_agnts])
    else:
        tran = transition.TabularTotalsTransition(ct, totals_column, accounting_column)
        updated, added, copied, removed = tran.transition(agnt, year)
        # print(type(updated))
        # print(type(added))
        # print(type(copied))
        # print(type(removed))
    # breakpoint()
    if (len(added) > 0) & (agents.name == "households"):
        metadata = orca.get_table("metadata").to_frame()
        max_hh_id = metadata.loc["max_hh_id", "value"]
        max_p_id = metadata.loc["max_p_id", "value"]
        if updated.loc[added, "household_id"].min() < max_hh_id:
        # if added.min() < max_hh_id:
            # print("HERE")
            # breakpoint()
            persons_df = orca.get_table("persons").local.reset_index()
            unique_hh_ids = updated["household_id"].unique()
            persons_old = persons_df[persons_df["household_id"].isin(unique_hh_ids)]
            updated = updated.sort_values(["household_id"])
            # Get households that are sampled/duplicated
            updated["cum_count"] = updated.groupby("household_id").cumcount()
            # NEW CODE 10/27
            updated = updated.sort_values(by=["cum_count"], ascending=False)
            updated.loc[:,"new_household_id"] = np.arange(updated.shape[0]) + max_hh_id + 1
            updated.loc[:,"new_household_id"] = np.where(updated["cum_count"]>0, updated["new_household_id"], updated["household_id"])
            sampled_persons = updated.merge(persons_df, how="left", left_on="household_id", right_on="household_id")
            sampled_persons = sampled_persons.sort_values(by=["cum_count"], ascending=False)
            sampled_persons.loc[:,"new_person_id"] = np.arange(sampled_persons.shape[0]) + max_p_id + 1
            sampled_persons.loc[:,"person_id"] = np.where(sampled_persons["cum_count"]>0, sampled_persons["new_person_id"], sampled_persons["person_id"])
            sampled_persons.loc[:,"household_id"] = np.where(sampled_persons["cum_count"]>0, sampled_persons["new_household_id"], sampled_persons["household_id"])
            updated.loc[:,"household_id"] = updated.loc[:, "new_household_id"]
            
            # sampled_households = updated[updated["cum_count"]>0]
            # sampled_households.loc[:, location_fname] = "-1"
            # old_households = updated[updated["cum_count"]==0]
            # Sample individuals from such households
            # sampled_households["new_household_id"] = np.arange(sampled_households.shape[0]) + max_hh_id + 1
            # sampled_persons = sampled_households.merge(persons_df, how="left", left_on="household_id", right_on="household_id")
            # # Update the id for households
            # sampled_households["household_id"] = sampled_households["new_household_id"].copy()
            # sampled_persons["person_id"] = np.arange(sampled_persons.shape[0]) + max_p_id + 1
            # sampled_persons["household_id"] = sampled_persons["new_household_id"].copy()
            # persons_df = pd.concat([persons_old, sampled_persons])
            # updated = pd.concat([old_households, sampled_households])
            # print((updated[location_fname]=="-1").sum())
            sampled_persons.set_index("person_id", inplace=True, drop=True)
            updated.set_index("household_id", inplace=True, drop=True)
            persons_local_columns = orca.get_injectable("persons_local_cols")
            # breakpoint()
            # At this breakpoint, figure out what changes from the previous one.
            orca.add_table("persons", sampled_persons.loc[:,persons_local_columns])
        # if added.min() < max_hh_id:
        #     # breakpoint()
        #     # reset "added" row IDs so that new rows do not assign
        #     # IDs of previously removed rows.
        #     new_max = max(agnt.index.max(), updated.index.max())
        #     new_added = np.arange(len(added)) + max_hh_id + 1
        #     updated["new_idx"] = None
        #     print(len(added))
        #     print(len(new_added))
        #     print(len(copied))
        #     print(len(removed))
        #     print(updated.shape)
        #     print(new_agnts.shape)
        #     # breakpoint()

        #     persons = orca.get_table("persons").local
        #     persons_removed = persons["household_id"].isin(removed)
        #     persons = persons[~persons_removed].copy()
        #     # person_times = new_agnts.groupby('household_id').size().to_frame('times').reset_index()
        #     persons_copied = persons[persons["household_id"].isin(copied)].reset_index()
        #     # person_times = person_times.merge(persons_copied, on="household_id")
        #     # persons_copied = person_times.loc[person_times.index.repeat(person_times['times'])]
        #     # breakpoint()

        #     new_agnts["new_idx"] = None
        #     new_agnts["new_idx"] = new_added
        #     # not_added = updated['new_idx'].isnull()
        #     # breakpoint()
        #     new_agnts.index.name = idx_name
        #     # breakpoint()
        #     # new_agnts
        #     persons_multiple = (
        #         new_agnts.groupby(["household_id", "new_idx"]).size().sort_index().to_frame("times").reset_index())
        #     new_persons = persons_copied.merge(
        #         persons_multiple, on="household_id", how="outer"
        #     )
        #     new_persons["household_id"] = new_persons["new_idx"].copy()
        #     # max_p_id = orca.get_table("persons").local.index.max()
        #     new_persons["person_id"] = np.arange(new_persons.shape[0]) + max_p_id + 1
        #     new_persons.set_index("person_id", inplace=True, drop=True)
        #     # breakpoint()
        #     persons = pd.concat([persons, new_persons])
        #     # breakpoint()

        #     updated["new_idx"] = updated.index.values
        #     # updated.loc[not_added, 'new_idx'] = updated.loc[not_added].index.values
        #     updated = pd.concat([updated, new_agnts])
        #     print(updated.shape[0])
        #     updated.set_index("new_idx", inplace=True, drop=True)
        #     updated.index.name = idx_name
        #     added = new_added
        #     # orca.get_table('persons')
        #     persons_local_columns = orca.get_table("persons").local_columns
        #     orca.add_table("persons", persons[persons_local_columns])
    # breakpoint()        
    # print("Updated shape:", updated.shape[0])
    # print("Added shape:", added.shape[0])
    # print("Removed shape:", removed.shape[0])
    # print("Copied shape:", copied.shape[0])
    # updated.loc[added, location_fname] = "-1"
    if agents.name != "households":
        updated.loc[added, location_fname] = "-1"
    if set_year_built:
        updated.loc[added, "year_built"] = year
    updated_links = {}
    if linked_tables:
        for table_name, (table, col) in linked_tables.items():
            print("updating linked table {}".format(table_name))
            updated_links[table_name] = update_linked_table(
                table, col, added, copied, removed
            )
            orca.add_table(table_name, updated_links[table_name])
    print("Total agents after transition: {}".format(len(updated)))
    orca.add_table(agents.name, updated[agents.local_columns])
    return updated, added, copied, removed


def simple_transition(
    tbl, rate, location_fname, linked_tables={}, set_year_built=False
):
    """
    Run a simple growth rate transition model

    Parameters
    ----------
    tbl : DataFrameWrapper
        Table to be transitioned
    rate : float
        Growth rate
    linked_tables : dict, optional
        Sets the tables linked to new or removed agents to be updated with dict of
        {'table_name':(DataFrameWrapper, 'link_id')}
    location_fname : str
        The field name in the resulting dataframe to set to -1 (to unplace
        new agents)
    Returns
    -------
    Nothing
    """
    print("Running simple transition with ", rate * 100, "% rate")
    transition = GrowthRateTransition(rate)
    df_base = tbl.to_frame(tbl.local_columns)
    print("%d agents before transition" % len(df_base.index))
    df, added, copied, removed = transition.transition(df_base, None)
    print("%d agents after transition" % len(df.index))
    if (len(added) > 0) & (tbl.name == "households"):
        metadata = orca.get_table("metadata").to_frame()
        max_hh_id = metadata.loc["max_hh_id", "value"]
        if added.min() < max_hh_id:
            # breakpoint()
            # reset "added" row IDs so that new rows do not assign
            # IDs of previously removed rows.
            new_max = max(df_base.index.max(), df.index.max())
            new_added = np.arange(len(added)) + new_max + 1
            idx_name = df.index.name
            df["new_idx"] = None
            df.loc[added, "new_idx"] = new_added
            not_added = df["new_idx"].isnull()
            # breakpoint()
            df.loc[not_added, "new_idx"] = df.loc[not_added].index.values
            df.set_index("new_idx", inplace=True, drop=True)
            df.index.name = idx_name
            added = new_added

    df.loc[added, location_fname] = "-1"

    if set_year_built:
        df.loc[added, "year_built"] = orca.get_injectable("year")
    updated_links = {}
    for table_name, (table, col) in linked_tables.items():
        updated_links[table_name] = update_linked_table(
            table, col, added, copied, removed
        )
        orca.add_table(table_name, updated_links[table_name])
    orca.add_table(tbl.name, df)


def update_linked_table(tbl, col_name, added, copied, removed):
    """
    Copy and update rows in a table that has a column referencing another
    table that has had rows added via copying.
    Parameters
    ----------
    tbl : DataFrameWrapper
        Table to update with new or removed rows.
    col_name : str
        Name of column in `table` that corresponds to the index values
        in `copied` and `removed`.
    added : pandas.Index
        Indexes of rows that are new in the linked table.
    copied : pandas.Index
        Indexes of rows that were copied to make new rows in linked table.
    removed : pandas.Index
        Indexes of rows that were removed from the linked table.
    Returns
    -------
    updated : pandas.DataFrame
    """
    # max ID should be preserved before rows are removed
    # otherwise new rows could have ID of what was removed.
    max_id = tbl.index.values.max()

    # max ID should be preserved before rows are removed
    # otherwise new rows could have ID of what was removed.
    max_id = tbl.index.values.max()

    # handle removals
    table = tbl.local
    table = table.loc[~table[col_name].isin(set(removed))]
    removed = table.loc[table[col_name].isin(set(removed))]
    if added is None or len(added) == 0:
        return table

    # map new IDs to the IDs from which they were copied
    id_map = pd.concat(
        [pd.Series(copied, name=col_name), pd.Series(added, name="temp_id")], axis=1
    )

    # join to linked table and assign new id
    new_rows = id_map.merge(table, on=col_name)
    new_rows.drop(col_name, axis=1, inplace=True)
    new_rows.rename(columns={"temp_id": col_name}, inplace=True)

    # index the new rows
    starting_index = max_id + 1
    new_rows.index = np.arange(
        starting_index, starting_index + len(new_rows), dtype=np.int)
    new_rows.index.name = table.index.name

    return pd.concat([table, new_rows])


@orca.step('households_relocation_basic')
def households_relocation_basic(households):
    return simple_relocation(households, .005, "block_id")


def simple_relocation(choosers, relocation_rate, fieldname):
    print("Total agents: %d" % len(choosers))
    print("Total currently unplaced: %d" % choosers[fieldname].value_counts().get("-1", 0))
    print("Assigning for relocation...")
    chooser_ids = np.random.choice(choosers.index, size=int(relocation_rate * len(choosers)), replace=False)
    choosers.update_col_from_series(fieldname, pd.Series('-1', index=chooser_ids))
    print("Total currently unplaced: %d" % choosers[fieldname].value_counts().get("-1", 0))

# -----------------------------------------------------------------------------------------
# POSTPROCESSING
# -----------------------------------------------------------------------------------------


@orca.step("generate_outputs")
def generate_outputs(year, base_year, forecast_year, tracts):
    print(
        "Generating outputs for (year {}, forecast year {})...".format(
            year, forecast_year
        )
    )
    if not os.path.exists("runs"):
        os.makedirs("./runs")

    if orca.get_injectable("all_local"):
        return

    if year == base_year:
        indicators.export_indicator_definitions()

    cfg = orca.get_injectable("output_parameters")

    # Layer indicators
    indicators.gen_all_indicators(cfg["output_indicators"], year)

    # Chart indicators
    # if year == forecast_year:
    #     chart_data, geo_small, geo_large = indicators.prepare_chart_data(cfg, year)
    #     # indicators.gen_all_charts(cfg['output_charts'], base_year, forecast_year, chart_data, geo_large)

    # Calibration metrics for pdf report in calibration/microsimulation routine
    if (year == 2018) and (orca.get_injectable("local_simulation") == True):
        indicators.gen_calibration_metrics(tracts)


@orca.step("export_demo_stats")
def export_demo_stats(year, forecast_year):
    """
    Export Demographic Stats tables

    Args:
        year (int): simulation year
        forecast_year (int): final forecast year

    Returns:
        None
    """

    if year == forecast_year:
        export("pop_over_time")
        export("hh_size_over_time")
        export("age_over_time")
        export("edu_over_time")
        export("income_over_time")
        export("kids_move_table")
        export("divorce_table")
        export("marriage_table")
        export("btable")
        export("age_dist_over_time")
        export("pop_size_over_time")
        export("student_population")
        export("mortalities")
        export("btable_elig")
        export("marrital")


def export(table_name):
    """
    Export the tables

    Args:
        table_name (string): Name of the orca table
    """
    
    region_code = orca.get_injectable("region_code")
    output_folder = orca.get_injectable("output_folder")
    df = orca.get_table(table_name).to_frame()
    # scenario_name = orca.get_injectable("scenario_name")
    # if scenario_name is False:
    #     csv_name = table_name + "_" + region_code +".csv"
    # else:
    #     csv_name = table_name + "_" + region_code + "_" + scenario_name + ".csv"
    csv_name = table_name + "_" + region_code +".csv"
    df.to_csv(output_folder+csv_name, index=False)


@orca.step("generate_metrics")
def generate_metrics(year, persons, households):
    """
    Update metrics of persons and households.

    Args:
        year (int): simulation year
        persons (DataFrameWrapper): DataFrameWrapper of the persons table

    Returns:
        None
    """
    persons_df = orca.get_table("persons").local
    households_df = orca.get_table("households").local
    age_over_time = orca.get_table("age_dist_over_time").to_frame()
    pop_over_time = orca.get_table("pop_size_over_time").to_frame()
    hh_over_time = orca.get_table("hh_size_over_time").to_frame()
    students = orca.get_table("student_population").to_frame()
    # age
    if age_over_time.empty:
        age_over_time = (
            persons_df.groupby("sex")["age"]
            .value_counts(
                bins=[
                    0,
                    0.9,
                    4,
                    9,
                    14,
                    19,
                    24,
                    29,
                    34,
                    39,
                    44,
                    49,
                    54,
                    59,
                    64,
                    69,
                    74,
                    79,
                    84,
                    89,
                    94,
                    99,
                    1000,
                ],
                sort=False,
            )
            .reset_index(name="count_" + str(year))
            .T
        )
    else:
        age_over_time_new = (
            persons_df.groupby("sex")["age"]
            .value_counts(
                bins=[
                    0,
                    0.9,
                    4,
                    9,
                    14,
                    19,
                    24,
                    29,
                    34,
                    39,
                    44,
                    49,
                    54,
                    59,
                    64,
                    69,
                    74,
                    79,
                    84,
                    89,
                    94,
                    99,
                    1000,
                ],
                sort=False,
            )
            .reset_index(name="count_" + str(year))
            .T
        )
        age_over_time = pd.concat([age_over_time, age_over_time_new])

    # pop
    if pop_over_time.empty:
        pop_over_time = pd.DataFrame.from_dict({
            "year": [str(year)],
            "count":  [persons_df.index.unique().shape[0]]
            })
    else:
        pop_over_time_new = pd.DataFrame.from_dict({
            "year": [str(year)],
            "count":  [persons_df.index.unique().shape[0]]
            })
        pop_over_time = pd.concat([pop_over_time, pop_over_time_new])

    # hh
    if hh_over_time.empty:
        hh_over_time = households_df.reset_index().groupby(["lcm_county_id","hh_size"]).agg(count = ("household_id", "size")).reset_index()
        hh_over_time["year"] = year
        hh_over_time["year"] = hh_over_time["year"].astype(str)
        hh_over_time["lcm_county_id"] = hh_over_time["lcm_county_id"].astype(str)
    else:
        hh_over_time_new = households_df.reset_index().groupby(["lcm_county_id","hh_size"]).agg(count = ("household_id", "size")).reset_index()
        hh_over_time_new["year"] = year
        hh_over_time_new["year"] = hh_over_time_new["year"].astype(str)
        hh_over_time_new["lcm_county_id"] = hh_over_time_new["lcm_county_id"].astype(str)
        hh_over_time = pd.concat([hh_over_time, hh_over_time_new])

    # students
    if students.empty:
        students = pd.DataFrame.from_dict({
            "year": [str(year)],
            "count":  [persons_df[
                    persons_df["edu"].isin(
                        [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
                    )
                ]["student"].sum()]
            })
    else:
        new_students = pd.DataFrame.from_dict({
            "year": [str(year)],
            "count":  [persons_df[
                    persons_df["edu"].isin(
                        [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
                    )
                ]["student"].sum()]
            })
        students = pd.concat([students, new_students])
        # print(students)
    # marrital = orca.get_table("marrital").to_frame()
    # if marrital.empty:
    #     persons_stats = persons_df[persons_df["age"]>=15]["MAR"].value_counts().reset_index()
    #     marrital = pd.DataFrame(persons_stats)
    #     marrital["year"] = year
    # else:
    #     persons_stats = persons_df[persons_df["age"]>=15]["MAR"].value_counts().reset_index()
    #     new_marrital = pd.DataFrame(persons_stats)
    #     new_marrital["year"] = year
    #     marrital = pd.concat([marrital, new_marrital])
    # print(marrital)
        
    orca.add_table("age_dist_over_time", age_over_time)
    orca.add_table("pop_size_over_time", pop_over_time)
    orca.add_table("student_population", students)
    orca.add_table("hh_size_over_time", hh_over_time)
    # orca.add_table("marrital", marrital)


# -----------------------------------------------------------------------------------------
# STEP DEFINITION
# -----------------------------------------------------------------------------------------

all_local = orca.get_injectable("all_local")
if orca.get_injectable("running_calibration_routine") == False:
    region_code = orca.get_injectable("region_code")

    if not all_local:
        storage_client = storage.Client("swarm-test-1470707908646")
        bucket = storage_client.get_bucket("national_block_v2")

    county_ids = orca.get_table("blocks").county_id.unique()
    rdplcm_segments = ["sf", "mf"]
    hlcm_segments = [
        "own_1p_54less",
        "own_1p_55plus",
        "own_2p_54less",
        "own_2p_55plus",
        "rent_1p_54less",
        "rent_1p_55plus",
        "rent_2p_54less",
        "rent_2p_55plus",
    ]
    elcm_segments = ["0", "1", "2", "3", "4", "5"]

    if orca.get_injectable("calibrated") == True:
        rdplcm_models = []
        hlcm_models = []
        elcm_models = []
        price_models = []
        if orca.get_injectable("multi_level_lcms") == True:
            if orca.get_injectable("segmented_lcms") == True:
                for county_id in county_ids:
                    rdplcm_models += [
                        "rdplcm_%s_blocks_%s_pf" % (county_id, segment)
                        for segment in rdplcm_segments
                    ]
                    hlcm_models += [
                        "hlcm_%s_blocks_%s_pf" % (county_id, segment)
                        for segment in hlcm_segments
                    ]
                    elcm_models += [
                        "elcm_%s_blocks_%s_pf" % (county_id, segment)
                        for segment in elcm_segments
                    ]
                if len(county_ids) > 1:
                    rdplcm_models = [
                        "rdplcm_county_%s_pf" % segment for segment in rdplcm_segments
                    ] + rdplcm_models
                    hlcm_models = [
                        "hlcm_county_%s_pf" % segment for segment in hlcm_segments
                    ] + hlcm_models
                    elcm_models = [
                        "elcm_county_%s_pf" % segment for segment in elcm_segments
                    ] + elcm_models
            else:
                rdplcm_models += [
                    "rdplcm_%s_blocks_pf" % county_id for county_id in county_ids
                ]
                hlcm_models += [
                    "hlcm_%s_blocks_pf" % county_id for county_id in county_ids
                ]
                elcm_models += [
                    "elcm_%s_blocks_pf" % county_id for county_id in county_ids
                ]
                if len(county_ids) > 1:
                    rdplcm_models = ["rdplcm_county_pf"] + rdplcm_models
                    hlcm_models = ["hlcm_county_pf"] + hlcm_models
                    elcm_models = ["elcm_county_pf"] + elcm_models
        else:
            if orca.get_injectable("segmented_lcms") == True:
                rdplcm_models += ["rdplcm_pf_" + segment for segment in rdplcm_segments]
                hlcm_models += ["hlcm_pf_" + segment for segment in hlcm_segments]
                elcm_models += ["elcm_pf_" + segment for segment in elcm_segments]
            else:
                rdplcm_models += ["rdplcm_pf"]
                hlcm_models += ["hlcm_pf"]
                elcm_models += ["elcm_pf"]

        developer_models = ["supply_transition"] + rdplcm_models
        household_models = ["household_transition"] + ["households_relocation_basic"] + hlcm_models
        employment_models = ["job_transition"] + elcm_models
        location_models = rdplcm_models + hlcm_models + elcm_models
        calibrated_folder = orca.get_injectable("calibrated_folder")
        region_type = orca.get_injectable("region_type")
        remote_configs_path = "calibrated_configs/%s/%s" % (
            calibrated_folder,
            region_code,
        )
        if calibrated_folder == "custom":
            remote_configs_path = "calibrated_configs/custom/custom_%s_%s" % (
                region_type,
                region_code,
            )
        local_configs_path = "calibrated_configs"
        if orca.get_injectable("local_simulation") is True:
            local_configs_path = os.path.join(
                local_configs_path, calibrated_folder, region_code
            )
            skim_source = orca.get_injectable("skim_source")
            if os.path.exists(os.path.join("configs", local_configs_path, skim_source)):
                local_configs_path = os.path.join(local_configs_path, skim_source)
        if not os.path.exists("configs/" + local_configs_path):
            os.makedirs("./configs/" + local_configs_path)
        for f in location_models:
            if not all_local:
                print(
                    "Downloading %s config from calibrated_configs/%s"
                    % (f, calibrated_folder)
                )
                blob = bucket.get_blob("%s/%s.yaml" % (remote_configs_path, f))
                blob.download_to_filename(
                    "./configs/%s/%s.yaml" % (local_configs_path, f)
                )
            else:
                if not os.path.exists("./configs/%s/%s.yaml" % (local_configs_path, f)):
                    raise OSError(
                        "No model config found at ./configs/%s/%s.yaml"
                        % (local_configs_path, f)
                    )

        for model in ["value", "rent"]:
            print("Checking if %s configs exist" % model)
            model_name = "repm_residential_%s" % model
            if not all_local:
                blob = bucket.blob("%s/%s.yaml" % (remote_configs_path, model_name))
                if blob.exists():
                    print("Downloading %s" % model_name)
                    blob.download_to_filename(
                        "./configs/%s/%s.yaml" % (local_configs_path, model_name)
                    )
                    price_models += [model_name]
            else:
                if os.path.exists(
                    "./configs/%s/%s.yaml" % (local_configs_path, model_name)
                ):
                    price_models += [model_name]

    else:
        rdplcm_models = ["rdplcm" + segment for segment in rdplcm_segments]
        hlcm_models = ["hlcm" + segment for segment in hlcm_segments]
        elcm_models = ["elcm" + segment for segment in elcm_segments]
        developer_models = ["supply_transition"] + [
            "rdplcm" + str(segment) for segment in range(0, 4)
        ]
        household_models = ["household_transition"] + ["households_relocation_basic"] + ["household_stats"], [
            "hlcm" + str(segment) for segment in range(1, 11)
        ]
        employment_models = ["job_transition"] + [
            "elcm" + str(segment) for segment in range(0, 6)
        ]
        location_models = rdplcm_models + hlcm_models + elcm_models
        price_models = [
            region_code + "_pred_bg_median_rent",
            region_code + "_pred_bg_median_value",
        ]

        if not os.path.exists("configs/estimated_configs"):
            os.makedirs("./configs/estimated_configs")
        for f in location_models:
            if not all_local:
                print("Downloading %s config from estimated_configs" % f)
                blob = bucket.get_blob(
                    "estimated_configs/us/%s/%s.yaml" % (region_code, f)
                )
                blob.download_to_filename("./configs/estimated_configs/%s.yaml" % f)
            else:
                if not os.path.exists("./configs/estimated_configs/%s.yaml" % f):
                    raise OSError(
                        "No model config found at ./configs/estimated_configs/%s.yaml"
                        % f
                    )

    if orca.get_injectable("local_simulation") == True:
        add_variables = ["add_temp_variables"]
        start_of_year_models = ["status_report"]
        demo_models = [
            "update_age",
            "laborforce_model",
            "households_reorg",
            "kids_moving_model",
            "fatality_model",
            "birth_model",
            "education_model",
            "export_demo_stats",
        ]
        pre_processing_steps = price_models + ["build_networks", "generate_outputs", "update_travel_data"]
        rem_variables = ["remove_temp_variables"]
        export_demo_steps = ["export_demo_stats"]
        household_stats = ["household_stats"]
        school_models = ["school_location"]
        end_of_year_models = ["generate_outputs"]
        work_models = ["work_location"]
        mlcm_postprocessing = ["mlcm_postprocessing"]
        update_income = ["update_income"]
        steps_all_years = (
            start_of_year_models
            + demo_models
            + work_models
            # + school_models
            + ["work_location_stats"]
            + price_models
            + ["work_location_stats"]
            + developer_models
            + ["work_location_stats"]
            + household_models
            + ["work_location_stats"]
            + employment_models
            + ["work_location_stats"]
            + end_of_year_models
            + ["income_stats"]
            + mlcm_postprocessing
            + ["work_location_stats"]
            + export_demo_steps
        )
    else:
        start_of_year_models = [
            "status_report",
            "skim_swapper",
            "scheduled_development_events_model",
        ]
        end_of_year_models = ["adjustment_model", "generate_outputs"]
        pre_processing_steps = (
            ["scenario_definition", "scheduled_development_events_model"]
            + price_models
            + ["build_networks", "skim_swapper", "generate_outputs"]
        )
        steps_all_years = (
            start_of_year_models
            + price_models
            + developer_models
            + household_models
            + employment_models
            + end_of_year_models
        )
    orca.add_injectable("sim_steps", steps_all_years)
    orca.add_injectable("pre_processing_steps", pre_processing_steps)
    orca.add_injectable("export_demo_stats", export_demo_steps)
