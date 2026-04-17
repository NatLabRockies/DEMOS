import orca
import time
import pandas as pd
from logging_logic import log_execution_time
from config import DEMOSConfig, AgingModuleConfig, get_config

STEP_NAME = "aging"
REQUIRED_COLUMNS = ["persons.age"]


@orca.step(STEP_NAME)
def aging(persons):
    """
    Increment the age of every person by one year.

    This step updates the `age` column in the `persons` table, simulating the passage of one year
    for all agents in the population.

    Parameters
    ----------
    persons : orca.Table
        The persons table containing the `age` column.

    Notes
    -----
    - Modifies `persons.age` in place.
    - Should be run once per simulation year.
    """
    start_time = time.time()
    persons["age"] += 1
    log_execution_time(start_time, orca.get_injectable("year"), STEP_NAME)


@orca.column(table_name="persons")
def child(data="persons.relate"):
    """
    Identify children in the persons table.

    Returns a binary indicator (1 or 0) for each person, where 1 indicates the person is a child
    based on the `relate` code (values 2, 3, 4, or 14).

    Parameters
    ----------
    data : pandas.Series
        The `relate` column from the persons table.

    Returns
    -------
    pandas.Series
        Binary indicator for child status.
    """
    return data.isin([2, 3, 4, 14]).astype(int)


@orca.column(table_name="persons")
def senior(data="persons.age"):
    """
    Identify seniors in the persons table.

    Returns a binary indicator (1 or 0) for each person, where 1 indicates the person is a senior,
    defined as having an age greater than or equal to the threshold specified in the aging module config
    (default is 65).

    Parameters
    ----------
    data : pandas.Series
        The `age` column from the persons table.

    Returns
    -------
    pandas.Series
        Binary indicator for senior status.
    """
    # Load calibration config
    demos_config: DEMOSConfig = get_config()
    aging_config: AgingModuleConfig = demos_config.aging_module_config

    return (data >= aging_config.senior_age).astype(int)


@orca.column(table_name="persons")
def age_group(data="persons.age"):
    """
    Assign each person to an age group.

    Categorizes persons into predefined age intervals for use in modeling and reporting.

    Parameters
    ----------
    data : pandas.Series
        The `age` column from the persons table.

    Returns
    -------
    pandas.Series
        Categorical age group labels as strings.
    """
    age_intervals = [0, 20, 30, 40, 50, 65, 900]
    age_labels = ["lte19", "20-29", "30-39", "40-49", "50-64", "gte65"]
    return pd.cut(
        data, bins=age_intervals, labels=age_labels, include_lowest=True
    ).astype(str)

@orca.column("households")
def hh_age_head(persons, households):
    """
    Get the age of the head of each household.

    Identifies the head of household (where `relate` == 0) and returns their age for each household.

    Parameters
    ----------
    persons : orca.Table
        The persons table containing `age`, `relate`, and `household_id` columns.

    Returns
    -------
    pandas.Series
        Age of the head of household, indexed by household_id.
    """
    heads = persons.local[persons["relate"] == 0]
    return heads.set_index("household_id").loc[households.index, "age"]