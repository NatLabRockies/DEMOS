import orca
from templates import estimated_models, modelmanager as mm
import time
import numpy as np
from logging_logic import log_execution_time
from templates.utils.models import columns_in_formula
from config import DEMOSConfig, KidsMovingModuleConfig, get_config

STEP_NAME = "kids_moving"
REQUIRED_COLUMNS = [
    "persons.age",
    "persons.relate",
]


@orca.step(STEP_NAME)
def kids_moving(persons, households, get_new_households):
    """
    Simulate children or young adults moving out of their parental households.

    This step applies the `kids_move` estimated model to eligible persons (age >= 16 and specific `relate` codes)
    to determine who moves out. Movers are assigned to new households, and both the persons and households tables
    are updated in place.

    Parameters
    ----------
    persons : orca.Table
        The persons table containing individual-level attributes.
    households : orca.Table
        The households table containing household-level attributes.
    get_new_households : callable
        Function to generate new unique household IDs as needed.

    Returns
    -------
    None

    Notes
    -----
    - Modifies `persons.household_id`, `persons.relate`, and `households.lcm_county_id` in place.
    - Only persons with `relate` codes [2, 3, 4, 7, 9, 14] and age >= 16 are considered.
    - Movers are only reassigned if their departure does not leave the household empty, unless all are moving.
    - Geographic assignment for new households is inherited from the original household.
    """
    start_time = time.time()

    kids_moving = run_and_calibrate_model(persons)
    update_households_after_kids(persons, households, kids_moving, get_new_households)
    log_execution_time(start_time, orca.get_injectable("year"), "kids_moving")


def update_households_after_kids(persons, households, kids_moving, get_new_households):
    """
    Update persons and households tables after kids move out.

    Assigns new household IDs to movers, updates their relationship code, and ensures
    new households inherit the geographic assignment from the original household.

    Parameters
    ----------
    persons : orca.Table
        The persons table.
    households : orca.Table
        The households table.
    kids_moving : pandas.Series
        Boolean Series indicating which persons are moving.
    get_new_households : callable
        Function to generate new unique household IDs.

    Returns
    -------
    None

    Notes
    -----
    - Movers are only reassigned if their departure does not leave the household empty, unless all are moving.
    - The `relate` code for movers is set to 0 (head of household).
    - The geographic assignment column is specified in the module config.
    """
    # Load module config
    demos_config: DEMOSConfig = get_config()
    module_config: KidsMovingModuleConfig = demos_config.kids_moving_module_config

    # Kids moving to a new household conditions
    ## Condition 1: Kids flagged by kids_moving
    ## Condition 2: Households with more than 1 people
    ## Condition 3: Households with some people staying

    household_sizes = persons.local.groupby("household_id").size()
    person_household_size_index = (
        household_sizes.loc[persons["household_id"]] > 1
    ).values

    ## Compute kids moving per household
    kids_moving_per_household = kids_moving.groupby(
        persons.local.loc[kids_moving.index, "household_id"]
    ).sum()
    ### This re-index speeds up querying by a lot
    kids_moving_per_household = kids_moving_per_household.reindex(
        persons["household_id"].unique()
    ).fillna(0)

    ### Household-level filter for condition 3
    household_completely_moving_index = (
        kids_moving_per_household.loc[household_sizes.index] == household_sizes
    )
    person_completely_moving_index = household_completely_moving_index.loc[
        persons["household_id"]
    ]

    ### Combine both household conditions to know which kids we need to move
    eligeble_households_index = (
        person_household_size_index & ~person_completely_moving_index
    ).values

    # Finally combine all filters into one
    kids_moving_index = (
        kids_moving.reindex(persons.local.index).fillna(0).astype(bool)
        & eligeble_households_index
    )

    # Get the old household_id for the moving kids to retrieve the county_id
    old_household_id = persons.local.loc[kids_moving_index, "household_id"].values
    geoid_assignment = households.local.loc[
        old_household_id, module_config.geoid_col
    ].values
    county_assignment = households.local.loc[old_household_id, "lcm_county_id"].values

    new_households = get_new_households(kids_moving_index.sum())
    persons.local.loc[kids_moving_index, "household_id"] = new_households
    persons.local.loc[kids_moving_index, "relate"] = 0
    households.local.loc[new_households, module_config.geoid_col] = geoid_assignment
    households.local.loc[new_households, "lcm_county_id"] = county_assignment


def run_and_calibrate_model(persons):
    # Load module config
    demos_config: DEMOSConfig = get_config()
    module_config: KidsMovingModuleConfig = demos_config.kids_moving_module_config

    child_relate = [
        2,
        3,
        4,
        7,
        9,
        14,
    ]  # This is more `dependent` because `child` is determined by age
    target_share = module_config.calibration_target_share
    max_iter = module_config.max_iter

    # Get model data
    model = mm.get_step("kids_move")
    model_variables = columns_in_formula(model.model_expression)
    model_filters = (persons.relate.isin(child_relate)) & (persons.age >= 16)
    model_data = persons.to_frame(model_variables)[model_filters]
    kids_moving = model.predict(model_data).astype(int)

    # NOTE: This could be much easier if we set the age at 18 because we could use model_filters
    adult_filter = persons.age >= 18
    age_moved = persons.age.loc[kids_moving[kids_moving == 1].index]
    adult_stay = (adult_filter & persons.relate.isin(child_relate)).sum() - (
        age_moved >= 18
    ).sum()
    observed_share = adult_stay / adult_filter.sum()
    error = observed_share - target_share

    print("Calibrating Kids moving model")
    calibrate_iteration = 0
    while (
        abs(error) > module_config.calibration_tolerance
        and calibrate_iteration < max_iter
    ):
        print(f"{calibrate_iteration} iteration error: {error}")
        model.fitted_parameters[0] += np.log(observed_share / target_share)

        kids_moving = model.predict(model_data).astype(int)
        age_moved = persons.age.loc[kids_moving[kids_moving == 1].index]
        adult_stay = (adult_filter & persons.relate.isin(child_relate)).sum() - (
            age_moved >= 18
        ).sum()
        observed_share = adult_stay / adult_filter.sum()
        error = observed_share - target_share

        calibrate_iteration += 1
    print(f"{calibrate_iteration} iteration error: {error}")

    return kids_moving
