import orca
import numpy as np
import pandas as pd
from templates import estimated_models, modelmanager as mm
import time
from datasources import log_execution_time

@orca.step("fatality_model")
def fatality_model(persons, households, year):
    """Function to run the fatality model at the persons level.
    The function also updates the persons and households tables,
    and saves the mortalities table.

    Args:
        persons (DataFrameWrapper): DataFrameWrapper of persons table
        households (DataFrameWrapper): DataFrameWrapper of households table
    """
    start_time = time.time()
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
    print("The Fatality Model Calibration:")
    calibrate_time = 0
    while error >= 1000:
        print(f"{calibrate_time} time: {error}")
        mortality.fitted_parameters[0] += np.log(target.sum()/fatality_list.sum())
        mortality.run()
        fatality_list = mortality.choices.astype(int)
        predicted_share = fatality_list.sum() / persons_df.shape[0]
        error = np.sqrt(np.mean((fatality_list.sum() - target)**2))
        calibrate_time += 1
    print(f"{calibrate_time} time: {error}")
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

    log_execution_time(start_time, orca.get_injectable("year"), "mortality")


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
