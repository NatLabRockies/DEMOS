import time
import orca
import numpy as np
import pandas as pd
from templates import estimated_models, modelmanager as mm
from datasources import log_execution_time


@orca.step("update_age")
def update_age(persons, households):
    """
    This function updates the age of the persons table and
    updates the age of the household head in the household table.

    Args:
        persons (DataFrameWrapper): DataFrameWrapper of the persons table
        households (DataFrameWrapper): DataFrameWrapper of the households table

    Returns:
        None
    """
    start_time = time.time()

    # print("Updating age of individuals...")
    persons_df = persons.local
    persons_df["age"] += 1
    households_df = households.to_frame(columns=["age_of_head", "hh_age_of_head"])
    households_df["age_of_head"] += 1
    households_df["hh_age_of_head"] = np.where(
        households_df["age_of_head"] < 35,
        "lt35",
        np.where(households_df["age_of_head"] < 65, "gt35-lt65", "gt65"),
    )
    persons_df["child"] = np.where(persons_df["relate"].isin([2, 3, 4, 14]), 1, 0)
    persons_df["person"] = 1
    persons_df["senior"] = np.where(persons_df["age"] >= 65, 1, 0)
    households_stats = persons_df.groupby(["household_id"]).agg(
        children=("child", "sum"), seniors=("senior", "sum"), size=("person", "sum")
    )
    households_stats["hh_children"] = np.where(
        households_stats["children"] > 0, "yes", "no"
    )
    households_stats["gt55"] = np.where(households_stats["seniors"] > 0, 1, 0)
    households_stats["hh_seniors"] = np.where(
        households_stats["seniors"] > 0, "yes", "no"
    )
    # Update age of household head in household table
    # print("Updating household and persons tables...")
    orca.get_table("households").update_col("age_of_head", households_df["age_of_head"])
    orca.get_table("households").update_col(
        "hh_age_of_head", households_df["hh_age_of_head"]
    )
    orca.get_table("households").update_col(
        "hh_children", households_stats["hh_children"]
    )
    orca.get_table("households").update_col("gt55", households_stats["gt55"])
    orca.get_table("households").update_col(
        "hh_seniors", households_stats["hh_seniors"]
    )
    orca.get_table("persons").update_col("age", persons_df["age"])

    log_execution_time(start_time, orca.get_injectable("year"), "aging")