import orca
import numpy as np
import pandas as pd
from templates import estimated_models, modelmanager as mm

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
        - persons.work_at_home
        - entering_workforce
        - exiting_workforce

    Args:
        persons (DataFrameWrapper): DataFrameWrapper of the persons table

    Returns:
        None
    """
    stay_unemployed_list = run_and_calibrate_in_workforce_model(persons, observed_entering_workforce, year)
    exit_workforce_list = run_and_calibrate_out_workforce_model(persons, observed_exiting_workforce, year)
    
    # Re-index to help querying below
    reindexed_remain_unemployed = stay_unemployed_list.reindex(persons.local.index).fillna(2)
    reindexed_exit_workforce = exit_workforce_list.reindex(persons.local.index).fillna(2)

    # Fix "work_at_home" - TODO: Not sure where would a NaN be produced for this
    persons["work_at_home"].fillna(0, inplace=True)

    # Updating working status and income
    persons.local.loc[reindexed_exit_workforce == 1, "worker"] = 0
    persons.local.loc[reindexed_exit_workforce == 1, "earning"] = 0
    persons.local.loc[reindexed_remain_unemployed == 0, "worker"] = 1
    persons.local.loc[reindexed_remain_unemployed == 0, "earning"] = persons["new_earning"]\
                                                                        .loc[reindexed_remain_unemployed == 0].values

    # Comments left by previous developer:
    # TODO: Make sure that the actual workers don't get restorted due to difference in indexing
    # TODO: Make sure there is a better way to do this

    agg_households = persons.local.groupby("household_id").agg(
        sum_workers = ("worker", "sum"), income = ("earning", "sum"))
    orca.get_table("households").local.update(agg_households)

    # Update entering and exiting workforce tables (Seems to be just for records)
    orca.add_table("entering_workforce", 
                   pd.concat([entering_workforce.local,
                              pd.DataFrame(data={"year": [year], "count": [(stay_unemployed_list == 0).sum()]})
                              ]))
    orca.add_table("exiting_workforce", 
                   pd.concat([exiting_workforce.local,
                              pd.DataFrame(data={"year": [year], "count": [(exit_workforce_list == 1).sum()]})
                              ]))

def sample_income(mean, std):
    return np.random.lognormal(mean, std)


# TODO: Refactor this
def run_and_calibrate_in_workforce_model(persons, observed_entering_workforce, year):
    # Observed values for calibration
    observed_stay_unemployed = observed_entering_workforce.to_frame()

    # Dummy value for output column
    persons["stay_out"] = -99
    np.random.seed(year + 200)
    # Get estimated model object and run it
    in_workforce_model = mm.get_step("enter_labor_force")
    in_workforce_model.run()

    stay_unemployed_list = in_workforce_model.choices.astype(int)
    predicted_share = stay_unemployed_list.sum() / stay_unemployed_list.shape[0]
    target_share = observed_stay_unemployed[observed_stay_unemployed["year"]==year]["share"]
    target = target_share * stay_unemployed_list.shape[0]
    error = np.sqrt(np.mean((predicted_share.sum() - target_share)**2))
    print("The Labor Force In Model Calibration:")
    calibrate_time = 0
    while error >= 0.01:
        print(f"{calibrate_time} time: {error}")
        in_workforce_model.fitted_parameters[0] += np.log(target.sum()/stay_unemployed_list.sum())
        in_workforce_model.run()
        stay_unemployed_list = in_workforce_model.choices.astype(int)
        predicted_share = stay_unemployed_list.sum() / stay_unemployed_list.shape[0]
        error = np.sqrt(np.mean((predicted_share.sum() - target_share)**2))
        calibrate_time += 1
    print(f"{calibrate_time} time: {error}")

    return stay_unemployed_list

# TODO: Refactor this
def run_and_calibrate_out_workforce_model(persons, observed_exiting_workforce, year):
    # Observed values for calibration
    observed_exit_workforce = observed_exiting_workforce.to_frame()

    # Dummy value for output column
    persons["leaving_workforce"] = -99
    np.random.seed(year + 210)

    # Get estimated model object and run it
    out_workforce_model = mm.get_step("exit_labor_force")
    out_workforce_model.run()
    
    exit_workforce_list = out_workforce_model.choices.astype(int)
    predicted_share = exit_workforce_list.sum() / exit_workforce_list.shape[0]
    target_share = observed_exit_workforce[observed_exit_workforce["year"]==year]["share"]
    target = target_share * exit_workforce_list.shape[0]

    error = np.sqrt(np.mean((predicted_share.sum() - target_share)**2))
    print("The Labor Force Out Model Calibration:")
    calibrate_time = 0
    while error >= 0.01:
        print(f"{calibrate_time} time: {error}")
        out_workforce_model.fitted_parameters[0] += np.log(target.sum()/exit_workforce_list.sum())
        out_workforce_model.run()
        exit_workforce_list = out_workforce_model.choices.astype(int)
        predicted_share = exit_workforce_list.sum() / exit_workforce_list.shape[0]
        error = np.sqrt(np.mean((predicted_share.sum() - target_share)**2))
        calibrate_time += 1
    print(f"{calibrate_time} time: {error}")

    return exit_workforce_list

@orca.column(table_name="persons", cache=True, cache_scope="iteration")
def age_group(data="persons.age"):
    age_intervals = [0, 20, 30, 40, 50, 65, 900]
    age_labels = ['lte20', '21-29', '30-39', '40-49', '50-64', 'gte65']
    return pd.cut(data, bins=age_intervals, labels=age_labels, include_lowest=True).astype(str)

@orca.column(table_name="persons", cache=True, cache_scope="iteration")
def education_group(data="persons.edu"):
    education_intervals = [0, 18, 22, 200]
    education_labels = ['lte17', '18-21', 'gte22']
    return pd.cut(data, bins=education_intervals, labels=education_labels, include_lowest=True).astype(str)

@orca.column(table_name="persons", cache=True, cache_scope="iteration")
def new_earning(persons, income_dist):
    persons_df = persons.to_frame(["age_group", "education_group"])
    merged_df = persons_df.merge(income_dist.local, on=['age_group', 'education_group'], how='left')
    return pd.Series(sample_income(merged_df["mu"], merged_df["sigma"]), index=persons_df.index)

@orca.column(table_name="households", cache=True, cache_scope="iteration")
def hh_workers(persons):
    return persons.to_frame(["household_id", "worker"]) \
           .groupby("household_id") \
           .sum()["worker"] \
           .apply(lambda r: "none" if r == 0 else
                  ("one" if r == 1 else "two or more"))


# @orca.column(table_name="households")
# def income(persons):
#     return persons.to_frame(["household_id", "earning"]) \
#                   .groupby("household_id") \
#                   .sum()["earning"]