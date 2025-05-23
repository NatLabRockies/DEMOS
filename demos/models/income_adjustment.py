import orca
import numpy as np
import pandas as pd
from templates.utils.models import columns_in_formula
from templates import estimated_models, modelmanager as mm

@orca.step("update_income")
def update_income(persons, households, income_rates, year):
    """
    Updating income for persons and households

    Args:
        persons (DataFrameWrapper): DataFrameWrapper of persons table
        households (DataFrameWrapper): DataFrameWrapper of households table
        year (int): simulation year
    """
    # Update income according to county rate
    persons.earning *= 1 + income_rates.local[income_rates["year"] == year] \
                                       .set_index("lcm_county_id")["rate"] \
                                       .loc[households.lcm_county_id \
                                       .loc[persons.household_id]].values
