import orca
import numpy as np
import pandas as pd
from demos.config import DEMOSConfig, get_config
from templates.utils.models import columns_in_formula
from templates import estimated_models, modelmanager as mm
import time
from logging_logic import log_execution_time

STEP_NAME = "income"


@orca.step(STEP_NAME)
def income(persons):
    """ """
    start_time = time.time()
    predicted_income = run_and_calibrate_income_model(persons)

    log_execution_time(start_time, orca.get_injectable("year"), "income")


# TODO: Refactor this
def run_and_calibrate_income_model(persons):
    # Load calibration config
    demos_config: DEMOSConfig = get_config()
    # calibration_procedure = demos_config.income_module_config.calibration_procedure
    calibration_procedure = None

    # Get model data
    model = mm.get_step("income")
    model_variables = columns_in_formula(model.model_expression)
    model_data = persons.to_frame(model_variables)

    # Calibrate if needed
    if calibration_procedure is not None:
        return calibration_procedure.calibrate_and_run_model(model, model_data)
    return np.exp(model.predict(model_data))

###################
# MODEL VARIABLES #
###################

@orca.column("persons")
def true_hh_size(persons, households):
    return pd.Series(
        households.local.loc[persons.household_id, "hh_size"]
        .replace({"one": 1, "two": 2, "three": 3, "four or more": 4})
        .values,
        index=persons.local.index.values,
    )

@orca.column("persons")
def not_met_area(persons):
    return pd.Series(np.ones(persons.local.shape[0]), index=persons.local.index)

# Education variables
# TODO: This numbers for education are not updated beyon 19 in the education model
@orca.column("persons")
def income_model_edu_bin1(persons):
    return persons["edu"].isin([15, 16, 17]).astype(int)

@orca.column("persons")
def income_model_edu_bin2(persons):
    return (persons["edu"] == 18).astype(int)

@orca.column("persons")
def income_model_edu_bin3(persons):
    return (persons["edu"] >= 19).astype(int)

# Job industry variables
# TODO: This column should be implemented more rigorously based on actual job industry data rather than random assignment
@orca.column("persons", cache=True, cache_scope="step")
def job_industry(persons):
    return pd.Series(np.random.choice([1, 2, 3, 4]), index=persons.index) 

@orca.column("persons")
def job_industry_bin1(persons): # First quartile
    return (persons["job_industry"] == 1).astype(int)

@orca.column("persons")
def job_industry_bin2(persons):
    return (persons["job_industry"] == 2).astype(int)

@orca.column("persons")
def job_industry_bin3(persons):
    return (persons["job_industry"] == 3).astype(int)

@orca.column("persons")
def job_industry_bin4(persons):
    return (persons["job_industry"] == 4).astype(int)

# TODO: This column should be implemented more rigorously based on actual job industry data rather than random assignment
@orca.column("persons", cache=True, cache_scope="step")
def job_occupation(persons):
    return pd.Series(np.random.choice([1, 2, 3, 4]), index=persons.index) 

@orca.column("persons")
def job_occupation_bin1(persons): # First quartile
    return (persons["job_occupation"] == 1).astype(int)

@orca.column("persons")
def job_occupation_bin2(persons):
    return (persons["job_occupation"] == 2).astype(int)

@orca.column("persons")
def job_occupation_bin3(persons):
    return (persons["job_occupation"] == 3).astype(int)

@orca.column("persons")
def job_occupation_bin4(persons):
    return (persons["job_occupation"] == 4).astype(int)


