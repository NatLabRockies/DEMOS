import orca
import numpy as np
import pandas as pd
from templates.utils.models import columns_in_formula
from templates import estimated_models, modelmanager as mm
import time
from logging_logic import log_execution_time

STEP_NAME = "income_adjustment"


@orca.step(STEP_NAME)
def income_adjustment(persons, households, income_rates, year):
    """
    Update person-level earnings based on county-specific income growth rates.

    This step applies multiplicative adjustments to person earnings using annual
    growth rates that vary by county. It maintains geographic variation in income
    trajectories while updating the entire population simultaneously.

    Parameters
    ----------
    persons : orca.Table
        The persons table containing individual-level attributes including earnings.
    households : orca.Table
        The households table containing household-level attributes including county ID.
    income_rates : orca.Table
        Table with annual income growth rates by county.
    year : int
        The current simulation year.

    Returns
    -------
    None

    Notes
    -----
    - Modifies `persons.earning` in place using multiplicative adjustment.
    - Requires `income_rates` table with columns: year, lcm_county_id, rate.
    - All persons in the same county receive the same proportional adjustment.
    - Rate of 0.05 means 5% growth (earnings multiplied by 1.05).
    """
    start_time = time.time()
    # TODO: CountyID is not being updated by default
    # Update income according to county rate
    persons.local.earning *= (
        1
        + income_rates.local.loc[year]
        .set_index("lcm_county_id")["rate"]
        .loc[households.lcm_county_id.loc[persons.household_id]]
        .values
    )
    log_execution_time(start_time, orca.get_injectable("year"), "income_adjustment")
