import orca
import numpy as np
import pandas as pd
from templates.utils.models import columns_in_formula
from templates import estimated_models, modelmanager as mm
import time
from datasources import log_execution_time
from config import DEMOSConfig

@orca.injectable(autocall=False)
def get_new_person_id(n):
    persons = orca.get_table("persons")
    graveyard = orca.get_table("graveyard")
    rebalanced_persons = orca.get_table("rebalanced_persons")

    current_max = max([persons.local.index.max(), graveyard.local.index.max(), rebalanced_persons.local.index.max()])
    return (
        np.arange(n)    # = [0, 1, 2 ...] up to the number of people
        + current_max   # = [max_person_id, max_person_id + 1, ...]
        + 1
    )


@orca.step("birth_model")
def birth_model(persons, households, graveyard, observed_births_data, get_new_person_id, year):
    """
    Function to run the birth model at the household level.
    The function updates the persons table.

    Modifies State Variables:
        - (Adds rows to `persons` table)

    Args:
        persons (DataFrameWrapper): DataFrameWrapper of the persons table
        households (DataFrameWrapper): DataFrameWrapper of the households table

    Returns:
        None
    """
    start_time = time.time()
    birth_list = run_and_calibrate_birth_model(persons, households, observed_births_data, year)

    # Get indices of households with babies
    house_indices = list(birth_list[birth_list == 1].index)

    # Initialize babies variables in the persons table.
    babies = pd.DataFrame(house_indices, columns=["household_id"])
    babies.index = get_new_person_id(len(babies))
    babies.index.name = "person_id"

    # Set default values
    babies["age"] = 0
    babies["edu"] = 0
    babies["earning"] = 0
    babies["relate"] = 2
    babies["MAR"] = 5
    babies["sex"] = np.random.choice([1, 2])
    babies["student"] = 0
    babies["worker"] = 0
    babies["work_at_home"] = 0


    # TODO: Values not used in refactored code
    # babies["hours"] = 0
    # babies["person_age"] = "19 and under"
    # babies["person_sex"] = babies["sex"].map({1: "male", 2: "female"})
    # babies["work_at_home"] = 0
    # babies["work_block_id"] = "-1"
    # babies["work_zone_id"] = "-1"
    # babies["workplace_taz"] = "-1"
    # babies["school_block_id"] = "-1"
    # babies["school_id"] = "-1"
    # babies["school_taz"] = "-1"
    # babies["school_zone_id"] = "-1"

    # TODO: Values now defined by orca columns
    # babies["child"] = 1
    # babies["senior"] = 0
    # babies["dead"] = -99
    # babies["person"] = 1
    # babies["education_group"] = "lte17"
    # babies["age_group"] = "lte20"

    # Set race of babies
    # TODO: There is duplication of information between `race_id` and `race`
    hh_races = (persons.local.groupby("household_id")
                             .agg(num_races=("race_id", "nunique"))
                             .reset_index()
                             .merge(
                                 households.to_frame(["hh_race_of_head", "hh_race_id_of_head", "household_id"])
                                 .reset_index(),
                               on="household_id")).set_index("household_id")
    one_race_hh_filter = (hh_races.loc[babies.household_id]["num_races"] == 1).values
    babies["race_id"] = 9
    babies.loc[one_race_hh_filter, "race_id"] = hh_races.loc[babies.loc[one_race_hh_filter, "household_id"], "hh_race_id_of_head"].values
    babies["race"] = babies["race_id"].map({1: "white", 2: "black"})
    babies["race"].fillna("other", inplace=True)
    
    # Finally add babies to persons table
    persons.local = pd.concat([persons.local, babies])

    log_execution_time(start_time, orca.get_injectable("year"), "birth")


def run_and_calibrate_birth_model(persons, households, observed_births_data, year,):
    ELIGIBILITY_COND = (persons["sex"] == 2) & (persons["age"].between(14, 45))
    ELIGIBLE_HH = persons.local.loc[ELIGIBILITY_COND, "household_id"].unique()

    households["birth"] = -99

    demos_config: DEMOSConfig = orca.get_injectable("demos_config")
    calibration_procedure = demos_config.birth_module_config.calibration_procedure
    if calibration_procedure is not None:
        birth_model = mm.get_step("birth")
        birth_model_variables = columns_in_formula(birth_model.model_expression)
        birth_model_data = households.to_frame(birth_model_variables).loc[ELIGIBLE_HH]
        
        return calibration_procedure.calibrate_model(birth_model, birth_model_data)