import orca
import numpy as np
import pandas as pd
from templates import estimated_models, modelmanager as mm
import time
from datasources import log_execution_time
from config import DEMOSConfig, EmploymentModuleConfig, SimultaneousCalibrationConfig, get_config
from templates.utils.models import columns_in_formula

@orca.step("laborforce_model")
def laborforce_model(persons,
                     entering_workforce,
                     exiting_workforce,
                     year):
    """
    Run the education model and update the persons table

    Modifies State Variables:
        - persons.worker
        - persons.earning
    
    Modifies Reporting tables:
        - entering_workforce
        - exiting_workforce

    Args:
        persons (DataFrameWrapper): DataFrameWrapper of the persons table

    Returns:
        None
    """
    start_time = time.time()
    # Load calibration config
    demos_config: DEMOSConfig = get_config()
    module_config: EmploymentModuleConfig = demos_config.employment_module_config

    if module_config.simultaneous_calibration_config is not None:
        stay_unemployed_list, exit_workforce_list= run_simultaenous_calibration(persons, module_config.simultaneous_calibration_config)
    else:
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


def run_and_calibrate_in_workforce_model(persons, calibration_procedure):
    # Get model data
    model = mm.get_step("enter_labor_force")
    model_variables = columns_in_formula(model.model_expression)
    model_filters = (persons.worker == 0) & (persons.age >= 18)
    model_data = persons.to_frame(model_variables)[model_filters]

    # Calibrate if needed
    if calibration_procedure is not None:
        return calibration_procedure.calibrate_and_run_model(model, model_data)
    return model.predict(model_data)


def run_and_calibrate_out_workforce_model(persons, calibration_procedure):
    # Get model data
    model = mm.get_step("exit_labor_force")
    model_variables = columns_in_formula(model.model_expression)
    model_filters = (persons.worker == 1) & (persons.age >= 18)
    model_data = persons.to_frame(model_variables)[model_filters]

    # Calibrate if needed
    if calibration_procedure is not None:
        return calibration_procedure.calibrate_and_run_model(model, model_data)
    return model.predict(model_data)


def run_simultaenous_calibration(persons, simultaneous_calibration_config):
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
    observed_workers = observed_workers_table[observed_workers_table.year == orca.get_injectable("year")]["count"].iloc[0]

    enter_model_predictions, exit_model_predictions = enter_model.predict(enter_model_data), exit_model.predict(exit_model_data)
    
    # enter_model_predictions == 0 are those who do NOT remain unemployed (will now be employed)
    # exit_model_preditions == 0 are those that remain employed
    predicted_total_workers = (enter_model_predictions == 0).sum() + (exit_model_predictions == 0).sum()
    

    # Initialize optimization algorithm
    config: SimultaneousCalibrationConfig = simultaneous_calibration_config
    gradient = 0
    momentum_weight = config.momentum_weight
    total_iterations = 0
    error = abs(predicted_total_workers - observed_workers)
    while error > config.tolerance and total_iterations < config.max_iter:
        print(f"Simultaneous Calibration: Iteration {total_iterations} error: {error}")
        lr = config.learning_rate * ((config.max_iter - total_iterations) + .5) / config.max_iter
        
        # Calculate gradient
        update_coeff = np.log(observed_workers / predicted_total_workers)
        gradient = lr * (momentum_weight * gradient + (1 - momentum_weight) * update_coeff / 2)

        # Apply gradient
        enter_model.fitted_parameters[0] -= gradient
        exit_model.fitted_parameters[0] -= gradient

        enter_model_predictions, exit_model_predictions = enter_model.predict(enter_model_data), exit_model.predict(exit_model_data)
        predicted_total_workers = (enter_model_predictions == 0).sum() + (exit_model_predictions == 0).sum()
        error = abs(predicted_total_workers - observed_workers)
        total_iterations += 1
    print(f"Final error after Simultaneous calibration: {error}")
    return enter_model_predictions, exit_model_predictions

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
