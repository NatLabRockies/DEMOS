import orca
import numpy as np
import pandas as pd
from templates import estimated_models, modelmanager as mm
import time
from logging_logic import log_execution_time

@orca.step("update_age")
def update_age(persons):
    """
    This function updates the age of the persons table and
    updates the age of the household head in the household table.

    Modifies State Variables:
        - persons.age

    Args:
        persons (DataFrameWrapper): DataFrameWrapper of the persons table
        households (DataFrameWrapper): DataFrameWrapper of the households table

    Returns:
        None
    """
    start_time = time.time()
    persons["age"] += 1
    log_execution_time(start_time, orca.get_injectable("year"), "aging")


@orca.column(table_name="persons", cache=True, cache_scope="iteration")
def child(data="persons.relate"):
    return data.isin([2, 3, 4, 14]).astype(int)


@orca.column(table_name="persons", cache=True, cache_scope="iteration")
def senior(data="persons.age"):
    return (data >= 65).astype(int)


@orca.column(table_name="persons", cache=True, cache_scope="iteration")
def age_gt55(data="persons.age"):
    return (data >= 55).astype(int)


@orca.column(table_name="households", cache=True, cache_scope="iteration")
def hh_children(persons):
    return persons.to_frame(["household_id", "child"]) \
                  .groupby("household_id") \
                  .sum()["child"].replace({0: "no", 1: "yes"})

@orca.column(table_name="households", cache=True, cache_scope="iteration")
def age_gt55(persons):
    return (persons.to_frame(["household_id", "senior"]) \
                  .groupby("household_id") \
                  .sum()["senior"] > 0).astype(int)


@orca.column(table_name="households", cache=True, cache_scope="iteration")
def hh_seniors(data="households.gt55"):
    return data.replace({0: "no", 1: "yes"})


@orca.column(table_name="households", cache=True, cache_scope="iteration")
def hh_age_of_head(data="households.age_of_head"):
    return pd.Series(
            np.where(data < 35,"lt35",
            np.where(data < 65, "gt35-lt65", "gt65")),
        index = data.index
    )