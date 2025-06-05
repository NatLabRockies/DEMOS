import orca
import numpy as np
import pandas as pd
from templates.utils.models import columns_in_formula
from templates import estimated_models, modelmanager as mm
import time
from datasources import log_execution_time

@orca.step("update_income")
def update_income(persons, households, year):
    """
    Updating income for persons and households

    Args:
        persons (DataFrameWrapper): DataFrameWrapper of persons table
        households (DataFrameWrapper): DataFrameWrapper of households table
        year (int): simulation year
    """
    start_time = time.time()

    # Pulling data, income rates, and county IDs
    persons_df = orca.get_table("persons").local
    households_df = orca.get_table("households").local

    households_local_cols = households_df.columns
    persons_local_cols = persons_df.columns
    # print(persons_local_cols)
    hh_counties = households_df["lcm_county_id"].copy()
    print("hh_counties: ", hh_counties.unique())
    income_rates = orca.get_table("income_rates").to_frame()
    income_rates = income_rates[income_rates["year"] == year]

    print("Households size before merging: ", persons_df["household_id"].unique().shape[0])
    persons_df = (persons_df.reset_index().merge(hh_counties.reset_index(), on=["household_id"]).set_index("person_id"))
    print("Households size after merging 1: ", persons_df["household_id"].unique().shape[0])
    persons_df = (persons_df.reset_index().merge(income_rates, on=["lcm_county_id"]).set_index("person_id"))
    print("Households size after merging 2: ", persons_df["household_id"].unique().shape[0])

    persons_df["earning"] = persons_df["earning"] * (1 + persons_df["rate"])

    new_incomes = persons_df.groupby("household_id").agg(income=("earning", "sum"))

    households_df.update(new_incomes)
    households_df["income"] = households_df["income"].astype(int)
    persons_df["member_id"] = persons_df.groupby("household_id")["relate"].rank(method="first", ascending=True).astype(int)
    persons_local_columns = orca.get_injectable("persons_local_cols")
    orca.add_table("persons", persons_df[persons_local_columns])
    orca.add_table("households", households_df[households_local_cols])
    # orca.add_table("persons", persons_df[persons_local_cols])
    # Update income stats at the persons level
    income_over_time = orca.get_table("income_over_time").to_frame()
    if income_over_time.empty:
        income_over_time = pd.DataFrame(
            data={"year": [year], "mean_income": [persons_df["earning"].mean()]}
        )
    else:
        new_income_over_time = pd.DataFrame(
                data={"year": [year], "mean_income": [persons_df["earning"].mean()]}
            )
        income_over_time = pd.concat([income_over_time,
                                      new_income_over_time],
                                     ignore_index=True)
    orca.add_table("income_over_time", income_over_time)
    log_execution_time(start_time, orca.get_injectable("year"), "income_adjustment")