import orca
import time
import numpy as np
import pandas as pd
from logging_logic import log_execution_time
from config import DEMOSConfig, AgingModuleConfig, get_config

STEP_NAME = "aging"
REQUIRED_COLUMNS = [
    "persons.age"
]

@orca.step(STEP_NAME)
def aging(persons):
    """
    Increases the age of every person in the persons table by 1

    **Required tables:**
        - persons
    
    **Modifies State Variables:**
        - persons.age

    """
    start_time = time.time()
    persons["age"] += 1
    log_execution_time(start_time, orca.get_injectable("year"), STEP_NAME)


@orca.column(table_name="persons")
def child(data="persons.relate"):
    """
    This column returns a binary value equal to 1 if the person has a `relate` value
    of `2`, `3`, `4` or `14`.
    """
    return data.isin([2, 3, 4, 14]).astype(int)


@orca.column(table_name="persons")
def senior(data="persons.age"):
    """
    Returns a binary value for each row in persons table set to 1 if the person
    is above `aging_module_config.senior_age` (defaults to `65`).
    """
    # Load calibration config
    demos_config: DEMOSConfig = get_config()
    aging_config: AgingModuleConfig = demos_config.aging_module_config

    return (data >= aging_config.senior_age).astype(int)

@orca.column(table_name="persons")
def age_group(data="persons.age"):
    age_intervals = [0, 20, 30, 40, 50, 65, 900]
    age_labels = ['lte19', '20-29', '30-39', '40-49', '50-64', 'gte65']
    return pd.cut(data, bins=age_intervals, labels=age_labels, include_lowest=True).astype(str)