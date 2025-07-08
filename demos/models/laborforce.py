import orca
import numpy as np
import pandas as pd
from templates import estimated_models, modelmanager as mm
import time
from datasources import log_execution_time
from config import DEMOSConfig, get_config
from templates.utils.models import columns_in_formula

@orca.step("laborforce_model")
def laborforce_model(persons,
                     observed_entering_workforce,
                     observed_exiting_workforce,
                     entering_workforce,
                     exiting_workforce,
                     year):
    """
    Run the education model and update the persons table

    Modifies State Variables:
        - persons.worker
        - persons.earning
        - entering_workforce
        - exiting_workforce

    Args:
        persons (DataFrameWrapper): DataFrameWrapper of the persons table

    Returns:
        None
    """
    start_time = time.time()
    stay_unemployed_list = run_and_calibrate_in_workforce_model(persons)
    exit_workforce_list = run_and_calibrate_out_workforce_model(persons)
    
    # Re-index to help querying below
    reindexed_remain_unemployed = stay_unemployed_list.reindex(persons.local.index).fillna(2)
    reindexed_exit_workforce = exit_workforce_list.reindex(persons.local.index).fillna(2)

    # Updating working status and income
    persons.local.loc[reindexed_exit_workforce == 1, "worker"] = 0
    persons.local.loc[reindexed_exit_workforce == 1, "earning"] = 0
    persons.local.loc[reindexed_remain_unemployed == 0, "worker"] = 1
    persons.local.loc[reindexed_remain_unemployed == 0, "earning"] = persons["new_earning"]\
                                                                        .loc[reindexed_remain_unemployed == 0].values

    # Update entering and exiting workforce tables (Seems to be just for records)
    orca.add_table("entering_workforce", 
                   pd.concat([entering_workforce.local,
                              pd.DataFrame(data={"year": [year], "count": [(stay_unemployed_list == 0).sum()]})
                              ]))
    orca.add_table("exiting_workforce", 
                   pd.concat([exiting_workforce.local,
                              pd.DataFrame(data={"year": [year], "count": [(exit_workforce_list == 1).sum()]})
                              ]))
    
    log_execution_time(start_time, orca.get_injectable("year"), "laborforce")

def sample_income(mean, std):
    return np.random.lognormal(mean, std)


# TODO: Refactor this
def run_and_calibrate_in_workforce_model(persons):
    # Load calibration config
    demos_config: DEMOSConfig = get_config()
    calibration_procedure = demos_config.employment_module_config.enter_model_calibration_procedure
    
    # Get model data
    model = mm.get_step("enter_labor_force")
    model_variables = columns_in_formula(model.model_expression)
    model_filters = (persons.worker == 0) & (persons.age >= 18)
    model_data = persons.to_frame(model_variables)[model_filters]

    # Calibrate if needed
    if calibration_procedure is not None:
        return calibration_procedure.calibrate_and_run_model(model, model_data)
    return model.predict(model_data)

# TODO: Refactor this
def run_and_calibrate_out_workforce_model(persons):
        # Load calibration config
    demos_config: DEMOSConfig = get_config()
    calibration_procedure = demos_config.employment_module_config.exit_model_calibration_procedure
    
    # Get model data
    model = mm.get_step("exit_labor_force")
    model_variables = columns_in_formula(model.model_expression)
    model_filters = (persons.worker == 1) & (persons.age >= 18)
    model_data = persons.to_frame(model_variables)[model_filters]

    # Calibrate if needed
    if calibration_procedure is not None:
        return calibration_procedure.calibrate_and_run_model(model, model_data)
    return model.predict(model_data)


@orca.column(table_name="persons")
def age_group(data="persons.age"):
    age_intervals = [0, 20, 30, 40, 50, 65, 900]
    age_labels = ['lte19', '20-29', '30-39', '40-49', '50-64', 'gte65']
    return pd.cut(data, bins=age_intervals, labels=age_labels, include_lowest=True).astype(str)

@orca.column(table_name="persons")
def education_group(data="persons.edu"):
    education_intervals = [0, 18, 22, 200]
    education_labels = ['lte17', '18-21', 'gte22']
    return pd.cut(data, bins=education_intervals, labels=education_labels, include_lowest=True).astype(str)

@orca.column(table_name="persons")
def new_earning(persons, income_dist):
    persons_df = persons.to_frame(["age_group", "education_group"])
    merged_df = persons_df.merge(income_dist.local, on=['age_group', 'education_group'], how='left')
    return pd.Series(sample_income(merged_df["mu"], merged_df["sigma"]), index=persons_df.index)

@orca.column(table_name="households")
def hh_workers(persons):
    return persons.to_frame(["household_id", "worker"]) \
           .groupby("household_id") \
           .sum()["worker"] \
           .apply(lambda r: "none" if r == 0 else
                  ("one" if r == 1 else "two or more"))

@orca.column(table_name="households")
def income(persons):
    return persons.to_frame(["household_id", "earning"]) \
                  .groupby("household_id") \
                  .sum()["earning"]
