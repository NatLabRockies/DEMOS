import orca
import numpy as np
import pandas as pd
from templates.utils.models import columns_in_formula
from templates import estimated_models, modelmanager as mm
import time
from logging_logic import log_execution_time
from config import DEMOSConfig, get_config

STEP_NAME = "birth"
REQUIRED_COLUMNS = []


@orca.injectable(autocall=False)
def get_new_person_id(n):
    """
    Generate new unique person IDs for newborns.

    Parameters
    ----------
    n : int
        Number of new person IDs to generate.

    Returns
    -------
    np.ndarray
        Array of new unique person IDs.
    """
    persons = orca.get_table("persons")
    graveyard = orca.get_table("graveyard")
    rebalanced_persons = orca.get_table("rebalanced_persons")

    current_max = max(
        [
            persons.local.index.max(),
            graveyard.local.index.max(),
            rebalanced_persons.local.index.max(),
        ]
    )
    return (
        np.arange(n)  # = [0, 1, 2 ...] up to the number of people
        + current_max  # = [max_person_id, max_person_id + 1, ...]
        + 1
    )


@orca.step(STEP_NAME)
def birth(persons, households, get_new_person_id):
    """
    Simulate household-level births and add new persons to the population.

    This step applies the birth model to eligible households, determines which have a birth event,
    and adds new babies to the persons table with default and inferred attributes.

    Parameters
    ----------
    persons : orca.Table
        The persons table containing individual-level attributes.
    households : orca.Table
        The households table containing household-level attributes.
    get_new_person_id : callable
        Function to generate new unique person IDs as needed.

    Returns
    -------
    None

    Notes
    -----
    - Adds new rows to the persons table for each birth event.
    - Babies are assigned default values for most attributes.
    - Race is assigned based on household head if all members share the same race; otherwise, "other".
    - Some attributes may be duplicated or missing if not set in input data.
    """
    start_time = time.time()
    birth_list = run_and_calibrate_birth_model(persons, households)

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

    # Set race of babies
    # TODO: There is duplication of information between `race_id` and `race`
    hh_races = (
        persons.local.groupby("household_id")
        .agg(num_races=("race_id", "nunique"))
        .reset_index()
        .merge(
            households.to_frame(
                ["hh_race_of_head", "hh_race_id_of_head", "household_id"]
            ).reset_index(),
            on="household_id",
        )
    ).set_index("household_id")
    one_race_hh_filter = (hh_races.loc[babies.household_id]["num_races"] == 1).values
    babies["race_id"] = 9
    babies.loc[one_race_hh_filter, "race_id"] = hh_races.loc[
        babies.loc[one_race_hh_filter, "household_id"], "hh_race_id_of_head"
    ].values
    babies["race"] = babies["race_id"].map({1: "white", 2: "black"})
    babies["race"].fillna("other", inplace=True)

    # Finally add babies to persons table
    persons.local = pd.concat([persons.local, babies])

    log_execution_time(start_time, orca.get_injectable("year"), "birth")


def run_and_calibrate_birth_model(persons, households):
    ELIGIBILITY_COND = (persons["sex"] == 2) & (persons["age"].between(14, 45))
    ELIGIBLE_HH = persons.local.loc[ELIGIBILITY_COND, "household_id"].unique()

    # Load calibration config
    demos_config: DEMOSConfig = get_config()
    calibration_procedure = demos_config.birth_module_config.calibration_procedure

    # Get model data
    birth_model = mm.get_step("birth")
    birth_model_variables = columns_in_formula(birth_model.model_expression)
    birth_model_data = households.to_frame(birth_model_variables).loc[ELIGIBLE_HH]

    # Calibrate if needed
    if calibration_procedure is not None:
        return calibration_procedure.calibrate_and_run_model(
            birth_model, birth_model_data
        )
    return birth_model.predict(birth_model_data)
