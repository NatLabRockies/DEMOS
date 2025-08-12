import os
import orca
import warnings
import pandas as pd
import numpy as np
from config import get_config

@orca.injectable("year")
def year():
    default_year = get_config().base_year
    iter_var = orca.get_injectable("iter_var")
    if iter_var is not None:
        return iter_var
    else:
        return default_year
    
# -----------------------------------------------------------------------------------------
# STEP DEFINITION
# -----------------------------------------------------------------------------------------
demo_models = [
    "update_age",
    "laborforce_model",
    "households_reorg",
    "kids_moving_model",
    "fatality_model",
    "birth_model",
    "education_model",
    "household_rebalancing",
    "update_income",
    # "export_demo_stats",
]
steps_all_years = (
    demo_models
)
orca.add_injectable("sim_steps", steps_all_years)
