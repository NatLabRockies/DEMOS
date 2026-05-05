import orca
import numpy as np
import pandas as pd
from demos.config import DEMOSConfig, get_config
from templates.utils.models import columns_in_formula
from templates import estimated_models, modelmanager as mm
import time
from logging_logic import log_execution_time
from .constants import STATE_QUARTILE_LABELS

STEP_NAME = "income"


@orca.step(STEP_NAME)
def income(households):
    """ """
    start_time = time.time()
    predicted_income = run_and_calibrate_income_model(households)
    households.local["income"] = predicted_income
    log_execution_time(start_time, orca.get_injectable("year"), "income")


def run_and_calibrate_income_model(households):
    # Load calibration config
    demos_config: DEMOSConfig = get_config()
    income_config = demos_config.income_module_config

    # Get model data
    model = mm.get_step("income_nworkers")
    model_variables = columns_in_formula(model.model_expression)
    model_data = households.to_frame(model_variables)

    # Calibrate if needed
    if income_config.calibration_procedure is not None:
        predicted = income_config.calibration_procedure.calibrate_and_run_model(
            model, model_data
        )
    else:
        predicted = np.exp(model.predict(model_data))

    # Inflation adjustment: scale from reference year to current simulation year
    if (
        income_config.inflation_adjustment_table is not None
        and income_config.inflation_reference_year is not None
    ):
        predicted = _apply_inflation_adjustment(
            predicted,
            reference_year=income_config.inflation_reference_year,
            current_year=orca.get_injectable("year"),
            table_name=income_config.inflation_adjustment_table,
        )

    return predicted


def _apply_inflation_adjustment(
    income: "pd.Series",
    reference_year: int,
    current_year: int,
    table_name: str,
) -> "pd.Series":
    """
    Scale *income* from *reference_year* nominal dollars to *current_year*
    nominal dollars using annual CPI adjustment factors stored in an orca
    table.

    The adjustment table must be indexed by ``year`` and contain an
    ``adjustment`` column with decimal annual inflation rates (e.g. ``0.030``
    for 3.0 %).  The cumulative factor is computed as the product of
    ``(1 + adjustment)`` for every year in the half-open interval
    ``(reference_year, current_year]`` when projecting forward, or its
    reciprocal when projecting backward.

    Parameters
    ----------
    income:
        Series of predicted household income values (in reference-year
        nominal dollars).
    reference_year:
        The year for which the income model was estimated.
    current_year:
        The current simulation year.
    table_name:
        Name of the orca table containing the adjustment factors.

    Returns
    -------
    pandas.Series
        Income values scaled to *current_year* nominal dollars.
    """
    if reference_year == current_year:
        return income

    adj_table = orca.get_table(table_name).to_frame()

    if current_year > reference_year:
        years = range(reference_year + 1, current_year + 1)
        forward = True
    else:
        years = range(current_year + 1, reference_year + 1)
        forward = False

    missing = [y for y in years if y not in adj_table.index]
    if missing:
        raise KeyError(
            f"Inflation adjustment table '{table_name}' is missing rows for "
            f"years: {missing}.  Ensure the table covers all simulation years."
        )

    cumulative_factor = float(
        (1 + adj_table.loc[list(years), "adjustment"]).prod()
    )
    if not forward:
        cumulative_factor = 1.0 / cumulative_factor

    return income * cumulative_factor


###################
# MODEL VARIABLES #
###################


@orca.column("households")
def true_hh_size(persons, households):
    return persons.local.groupby("household_id").size().loc[households.index]


@orca.column(table_name="households")
def true_hh_workers(persons):
    """
    Compute the number of workers per household.

    Parameters
    ----------
    persons : orca.Table
        The persons table.

    Returns
    -------
    pandas.Series
        Sum of total workers in each household, indexed by household_id.
    """
    return persons.worker.groupby(persons.household_id).sum()


# @orca.column("households")
# def not_met_area(households):
#     return pd.Series(np.ones(households.local.shape[0]), index=households.local.index)


# Education variables
# TODO: This numbers for education are not updated beyon 19 in the education model
@orca.column("households")
def hh_head_edu_bin1(households, persons):
    # Get the persons row for the head of every household (head is when relate == 0)
    heads = persons.local[persons["relate"] == 0]
    return (
        heads.set_index("household_id")
        .loc[households.index, "edu"]
        .isin([15, 16, 17])
        .astype(int)
    )


@orca.column("households")
def hh_head_edu_bin2(households, persons):
    heads = persons.local[persons["relate"] == 0]
    return (heads.set_index("household_id").loc[households.index, "edu"] == 18).astype(
        int
    )


@orca.column("households")
def hh_head_edu_bin3(households, persons):
    heads = persons.local[persons["relate"] == 0]
    return (heads.set_index("household_id").loc[households.index, "edu"] >= 19).astype(
        int
    )


# Job industry variables
# TODO: This column should be implemented more rigorously based on actual job industry data rather than random assignment
# @orca.column("households", cache=True, cache_scope="step")
# def job_industry(households):
#     return pd.Series(np.random.choice([1, 2, 3, 4]), index=households.index)


@orca.column("households")
def job_industry_bin1(households):  # First quartile
    return (households["job_industry"] == 1).astype(int)


@orca.column("households")
def job_industry_bin2(households):
    return (households["job_industry"] == 2).astype(int)


@orca.column("households")
def job_industry_bin3(households):
    return (households["job_industry"] == 3).astype(int)


@orca.column("households")
def job_industry_bin4(households):
    return (households["job_industry"] == 4).astype(int)


# TODO: This column should be implemented more rigorously based on actual job industry data rather than random assignment
# @orca.column("households", cache=True, cache_scope="step")
# def job_occupation(households):
#     return pd.Series(np.random.choice([1, 2, 3, 4]), index=households.index)


@orca.column("households")
def job_occupation_bin1(households):  # First quartile
    return (households["job_occupation"] == 1).astype(int)


@orca.column("households")
def job_occupation_bin2(households):
    return (households["job_occupation"] == 2).astype(int)


@orca.column("households")
def job_occupation_bin3(households):
    return (households["job_occupation"] == 3).astype(int)


@orca.column("households")
def job_occupation_bin4(households):
    return (households["job_occupation"] == 4).astype(int)


@orca.column("households")
def state_quartile(households):
    return pd.Series(
        households.local["lcm_county_id"]
        .apply(lambda s: f"{int(s) // 1000:02d}")
        .map(STATE_QUARTILE_LABELS)
        .values,
        index=households.local.index.values,
    )


@orca.column("households")
def state_quart_2(households):
    return (households["state_quartile"] == 2).astype(int)


@orca.column("households")
def state_quart_3(households):
    return (households["state_quartile"] == 3).astype(int)


@orca.column("households")
def state_quart_4(households):
    return (households["state_quartile"] == 4).astype(int)


@orca.column("households")
def hh_head_race_black(households, persons):
    heads = persons.to_frame(["household_id", "race_black"])[persons["relate"] == 0]
    return (heads.set_index("household_id").loc[households.index, "race_black"]).astype(
        int
    )


@orca.column("households")
def hh_head_race_native_am(households, persons):
    heads = persons.to_frame(["household_id", "race_native_am"])[persons["relate"] == 0]
    return (
        heads.set_index("household_id").loc[households.index, "race_native_am"]
    ).astype(int)


@orca.column("households")
def hh_head_race_asian(households, persons):
    heads = persons.to_frame(["household_id", "race_asian"])[persons["relate"] == 0]
    return (heads.set_index("household_id").loc[households.index, "race_asian"]).astype(
        int
    )


@orca.column("households")
def hh_head_race_hawaiian(households, persons):
    heads = persons.to_frame(["household_id", "race_hawaiian"])[persons["relate"] == 0]
    return (
        heads.set_index("household_id").loc[households.index, "race_hawaiian"]
    ).astype(int)


@orca.column("households")
def hh_head_race_acs_other(households, persons):
    heads = persons.to_frame(["household_id", "race_acs_other"])[persons["relate"] == 0]
    return (
        heads.set_index("household_id").loc[households.index, "race_acs_other"]
    ).astype(int)
