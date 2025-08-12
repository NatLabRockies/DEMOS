import orca
import numpy as np
import pandas as pd
from templates.utils.models import columns_in_formula
from templates import estimated_models, modelmanager as mm
import time
from logging_logic import log_execution_time

@orca.step("update_income")
def update_income(persons, households, income_rates, year):
    """
    Updating income for persons and households

    Args:
        persons (DataFrameWrapper): DataFrameWrapper of persons table
        households (DataFrameWrapper): DataFrameWrapper of households table
        year (int): simulation year
    """
    start_time = time.time()
    # Update income according to county rate
    persons.local.earning *= 1 + income_rates.local.loc[year] \
                                       .set_index("lcm_county_id")["rate"] \
                                       .loc[households.lcm_county_id \
                                       .loc[persons.household_id]].values
    log_execution_time(start_time, orca.get_injectable("year"), "income_adjustment")
