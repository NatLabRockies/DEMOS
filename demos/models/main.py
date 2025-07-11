import os
import orca
import warnings
import pandas as pd
import numpy as np

# -----------------------------------------------------------------------------------------
# STEP DEFINITION
# -----------------------------------------------------------------------------------------
demo_models = [
    "update_age",
    # "laborforce_model",
    # "households_reorg",
    # "kids_moving_model",
    # "fatality_model",
    # "birth_model",
    # "education_model",
    # "household_rebalancing",
    # "update_income",
    # "export_demo_stats",
]
steps_all_years = (
    demo_models
)
orca.add_injectable("sim_steps", steps_all_years)
