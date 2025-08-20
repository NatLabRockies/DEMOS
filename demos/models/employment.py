import orca
import numpy as np
import pandas as pd
from templates import estimated_models, modelmanager as mm
from templates.calibration import CalibrationConfig
import time
from logging_logic import log_execution_time
from config import (
    DEMOSConfig,
    EmploymentModuleConfig,
    SimultaneousCalibrationConfig,
    get_config,
)
from templates.utils.models import columns_in_formula
from loguru import logger

STEP_NAME = "employment"
REQUIRED_COLUMNS = ["persons.age", "persons.worker", "persons.earning"]


@orca.step(STEP_NAME)
def employment(persons):
    """
    Simulate labor force transitions for eligible persons.

    This step applies estimated models to determine which unemployed persons become employed
    and which employed persons exit the workforce. Only persons aged 18 or older are considered.
    The function updates the `worker` and `earning` columns in the `persons` table.

    Parameters
    ----------
    persons : orca.Table
        The persons table containing individual-level attributes.

    Notes
    -----
    - Requires the `persons` table with columns: `age`, `worker`, `earning`.
    - Modifies `persons.worker` and `persons.earning` in place.
    - Uses module configuration from the TOML config file.
    - Triggers caching of the `income_dist` table for income assignment.

    Example
    -------
    This step is executed as part of the annual simulation loop:
        `orca.run(['employment'], iter_vars=[...])`
    """
    start_time = time.time()

    # Load calibration config
    demos_config: DEMOSConfig = get_config()
    module_config: EmploymentModuleConfig = demos_config.employment_module_config

    if module_config.simultaneous_calibration_config is not None:
        stay_unemployed_list, exit_workforce_list = run_simultaenous_calibration(
            persons, module_config.simultaneous_calibration_config
        )
    else:
        stay_unemployed_list = run_and_calibrate_in_workforce_model(
            persons, module_config.enter_model_calibration_procedure
        )
        exit_workforce_list = run_and_calibrate_out_workforce_model(
            persons, module_config.exit_model_calibration_procedure
        )

    # Re-index to help querying below
    reindexed_remain_unemployed = stay_unemployed_list.reindex(
        persons.local.index
    ).fillna(2)
    reindexed_exit_workforce = exit_workforce_list.reindex(persons.local.index).fillna(
        2
    )

    # Trigger a cache for `income_dist`
    _ = orca.get_table("income_dist").to_frame()

    # Updating working status and income
    persons.local.loc[reindexed_exit_workforce == 1, "worker"] = 0
    persons.local.loc[reindexed_exit_workforce == 1, "earning"] = 0
    persons.local.loc[reindexed_remain_unemployed == 0, "worker"] = 1
    persons.local.loc[reindexed_remain_unemployed == 0, "earning"] = (
        persons["new_earning"].loc[reindexed_remain_unemployed == 0].values
    )

    log_execution_time(start_time, orca.get_injectable("year"), STEP_NAME)


def sample_income(mean, std):
    """
    Draw samples from a lognormal distribution.

    Parameters
    ----------
    mean : float or array-like
        The mean(s) of the underlying normal distribution.
    std : float or array-like
        The standard deviation(s) of the underlying normal distribution.

    Returns
    -------
    float or np.ndarray
        Sample(s) from the lognormal distribution.
    """
    return np.random.lognormal(mean, std)


def run_and_calibrate_in_workforce_model(
    persons: pd.DataFrame, calibration_procedure: CalibrationConfig
) -> pd.Series:
    """
    Run the 'enter_labor_force' estimated model for eligible persons.

    Parameters
    ----------
    persons : pandas.DataFrame
        DataFrame of persons with required model variables.
    calibration_procedure : CalibrationConfig or None
        Calibration procedure to apply, if any.

    Returns
    -------
    pandas.Series
        Model predictions for each eligible person (0 = remain unemployed, 1 = become employed).
    """
    # Get model data
    model = mm.get_step("enter_labor_force")
    model_variables = columns_in_formula(model.model_expression)
    model_filters = (persons.worker == 0) & (persons.age >= 18)
    model_data = persons.to_frame(model_variables)[model_filters]

    # Calibrate if needed
    if calibration_procedure is not None:
        return calibration_procedure.calibrate_and_run_model(model, model_data)
    return model.predict(model_data)


def run_and_calibrate_out_workforce_model(
    persons: pd.DataFrame, calibration_procedure: CalibrationConfig
) -> pd.Series:
    """
    Run the 'exit_labor_force' estimated model for eligible persons.

    Parameters
    ----------
    persons : pandas.DataFrame
        DataFrame of persons with required model variables.
    calibration_procedure : CalibrationConfig or None
        Calibration procedure to apply, if any.

    Returns
    -------
    pandas.Series
        Model predictions for each eligible person (0 = remain employed, 1 = become unemployed).
    """
    # Get model data
    model = mm.get_step("exit_labor_force")
    model_variables = columns_in_formula(model.model_expression)
    model_filters = (persons.worker == 1) & (persons.age >= 18)
    model_data = persons.to_frame(model_variables)[model_filters]

    # Calibrate if needed
    if calibration_procedure is not None:
        return calibration_procedure.calibrate_and_run_model(model, model_data)
    return model.predict(model_data)


def run_simultaenous_calibration(
    persons: pd.DataFrame,
    simultaneous_calibration_config: SimultaneousCalibrationConfig,
) -> pd.Series:
    """
    Run simultaneous calibration for both 'enter' and 'exit' labor force models.

    Parameters
    ----------
    persons : pandas.DataFrame
        DataFrame of persons with required model variables.
    simultaneous_calibration_config : SimultaneousCalibrationConfig
        Configuration for simultaneous calibration.

    Returns
    -------
    tuple of pandas.Series
        Predictions for entering and exiting the workforce.
    """
    # Get enter model data
    enter_model = mm.get_step("enter_labor_force")
    enter_model_variables = columns_in_formula(enter_model.model_expression)
    enter_model_filters = (persons.worker == 0) & (persons.age >= 18)
    enter_model_data = persons.to_frame(enter_model_variables)[enter_model_filters]

    # Get exit model data
    exit_model = mm.get_step("exit_labor_force")
    exit_model_variables = columns_in_formula(exit_model.model_expression)
    exit_model_filters = (persons.worker == 1) & (persons.age >= 18)
    exit_model_data = persons.to_frame(exit_model_variables)[exit_model_filters]

    # Get calibration data
    observed_workers_table = orca.get_table("observed_employment").local
    observed_workers = observed_workers_table[
        observed_workers_table.index == orca.get_injectable("year")
    ]["count"].iloc[0]

    enter_model_predictions, exit_model_predictions = enter_model.predict(
        enter_model_data
    ), exit_model.predict(exit_model_data)

    # enter_model_predictions == 0 are those who do NOT remain unemployed (will now be employed)
    # exit_model_preditions == 0 are those that remain employed
    predicted_total_workers = (enter_model_predictions == 0).sum() + (
        exit_model_predictions == 0
    ).sum()

    # Initialize optimization algorithm
    config: SimultaneousCalibrationConfig = simultaneous_calibration_config
    gradient = 0
    momentum_weight = config.momentum_weight
    total_iterations = 0
    error = abs(predicted_total_workers - observed_workers)
    while error > config.tolerance and total_iterations < config.max_iter:
        logger.info(
            f"Simultaneous Calibration: Iteration {total_iterations} error: {error}"
        )
        lr = (
            config.learning_rate
            * ((config.max_iter - total_iterations) + 0.5)
            / config.max_iter
        )

        # Calculate gradient
        update_coeff = np.log(observed_workers / predicted_total_workers)
        gradient = lr * (
            momentum_weight * gradient + (1 - momentum_weight) * update_coeff / 2
        )

        # Apply gradient
        enter_model.fitted_parameters[0] -= gradient
        exit_model.fitted_parameters[0] -= gradient

        enter_model_predictions, exit_model_predictions = enter_model.predict(
            enter_model_data
        ), exit_model.predict(exit_model_data)
        predicted_total_workers = (enter_model_predictions == 0).sum() + (
            exit_model_predictions == 0
        ).sum()
        error = abs(predicted_total_workers - observed_workers)
        total_iterations += 1
    print(f"Final error after Simultaneous calibration: {error}")
    return enter_model_predictions, exit_model_predictions


@orca.column(table_name="persons")
def new_earning(persons, income_dist):
    """
    Compute new earnings for persons entering the workforce.

    For each eligible person, samples income from a lognormal distribution
    parameterized by age and education group, using the cached `income_dist` table.

    Parameters
    ----------
    persons : orca.Table
        The persons table.
    income_dist : orca.Table
        Table with income distribution parameters.

    Returns
    -------
    pandas.Series
        Sampled earnings for each person.
    """
    persons_df = persons.to_frame(["age_group", "education_group"])
    merged_df = persons_df.merge(
        income_dist.local, on=["age_group", "education_group"], how="left"
    )
    return pd.Series(
        sample_income(merged_df["mu"], merged_df["sigma"]), index=persons_df.index
    )


@orca.column(table_name="households")
def hh_workers(persons):
    """
    Compute the number of workers per household.

    Parameters
    ----------
    persons : orca.Table
        The persons table.

    Returns
    -------
    pandas.Series
        Categorical summary: "none", "one", or "two or more" workers per household.
    """
    return (
        persons.to_frame(["household_id", "worker"])
        .groupby("household_id")
        .sum()["worker"]
        .apply(lambda r: "none" if r == 0 else ("one" if r == 1 else "two or more"))
    )


@orca.column(table_name="households")
def income(persons):
    """
    Aggregate household income from person-level earnings.

    Parameters
    ----------
    persons : orca.Table
        The persons table.

    Returns
    -------
    pandas.Series
        Total income per household.

    Notes
    -----
    This is for `HOUSEHOLDS` table
    """
    return (
        persons.to_frame(["household_id", "earning"])
        .groupby("household_id")
        .sum()["earning"]
    )


@orca.table(cache=True, cache_scope="forever")
def income_dist(persons):
    """
    Compute and cache income distribution parameters by age and education group.

    This table is used to sample earnings for new workers in the employment model.

    Parameters
    ----------
    persons : orca.Table
        The persons table.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns: age_group, education_group, data_mean, data_std, mu, sigma.
    """
    income_dist = (
        persons.to_frame(["worker", "age_group", "education_group", "earning"])[
            persons["worker"] == 1
        ]
        .groupby(["age_group", "education_group"])
        .agg(data_mean=("earning", "mean"), data_std=("earning", "std"))
        .reset_index()
    )

    # Convert to the parameters of the underlying normal distribution
    mu = pd.Series(
        np.log(
            income_dist["data_mean"] ** 2
            / np.sqrt(income_dist["data_std"] ** 2 + income_dist["data_mean"] ** 2)
        ),
        name="mu",
    )
    sigma = pd.Series(
        np.sqrt(
            np.log(1 + income_dist["data_std"] ** 2 / income_dist["data_mean"] ** 2)
        ),
        name="sigma",
    )
    income_dist = pd.concat([income_dist, mu, sigma], axis=1)
    return income_dist
