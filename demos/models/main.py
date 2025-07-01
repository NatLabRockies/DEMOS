import os
import orca
import warnings
import indicators
import pandas as pd
import numpy as np

# TODO: This seems to be logging. Integrate all logging in a consistent way.
print("Importing models for region", orca.get_injectable("region_code"))
# TODO: Handle this
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------------------
# STEP DEFINITION
# -----------------------------------------------------------------------------------------

all_local = orca.get_injectable("all_local")
if orca.get_injectable("running_calibration_routine") == False:
    if orca.get_injectable("local_simulation") == True:
        demo_models = [
            # "update_age",
            # "laborforce_model",
            # "households_reorg",
            # "kids_moving_model",
            # "fatality_model",
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
