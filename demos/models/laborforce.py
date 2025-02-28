import orca
import numpy as np
import pandas as pd
from templates import estimated_models, modelmanager as mm

@orca.step("laborforce_model")
def laborforce_model(persons, year):
    """
    Run the education model and update the persons table

    YE: ***

    Args:
        persons (DataFrameWrapper): DataFrameWrapper of the persons table

    Returns:
        None
    """
    # Add temporary variable
    persons_df = orca.get_table("persons").local
    persons_df["stay_out"] = -99
    persons_df["leaving_workforce"] = -99
    orca.add_table("persons", persons_df)
    persons_df = orca.get_table("persons").local

    in_workforce_model = mm.get_step("enter_labor_force")
    in_workforce_model.run()
    stay_unemployed_list = in_workforce_model.choices.astype(int)
    predicted_share = stay_unemployed_list.sum() / stay_unemployed_list.shape[0]
    observed_stay_unemployed = orca.get_table("observed_entering_workforce").to_frame()
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
    
    out_workforce_model = mm.get_step("exit_labor_force")
    out_workforce_model.run()
    exit_workforce_list = out_workforce_model.choices.astype(int)
    predicted_share = exit_workforce_list.sum() / exit_workforce_list.shape[0]
    observed_exit_workforce = orca.get_table("observed_exiting_workforce").to_frame()
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

    # Update labor status
    update_labor_status(persons, stay_unemployed_list, exit_workforce_list, year)

def update_labor_status(persons, stay_unemployed_list, exit_workforce_list, year):
    """
    Function to update the worker status in persons table based
    on the labor participation model

    YE: ***

    Args:
        persons (DataFrameWrapper): DataFrameWrapper of the persons table
        student_list (pd.Series): Pandas Series containing the output of
        the education model

    Returns:
        None
    """
    # Pull Data
    persons_df = orca.get_table("persons").local
    persons_cols = orca.get_injectable("persons_local_cols")
    households_df = orca.get_table("households").local
    households_cols = orca.get_injectable("households_local_cols")
    income_summary = orca.get_table("income_dist").local

    #####################################################
    age_intervals = [0, 20, 30, 40, 50, 65, 900]
    education_intervals = [0, 18, 22, 200]
    # Define the labels for age and education groups
    age_labels = ['lte20', '21-29', '30-39', '40-49', '50-64', 'gte65']
    education_labels = ['lte17', '18-21', 'gte22']
    # Create age and education groups with labels
    persons_df['age_group'] = pd.cut(persons_df['age'], bins=age_intervals, labels=age_labels, include_lowest=True)
    persons_df['education_group'] = pd.cut(persons_df['edu'], bins=education_intervals, labels=education_labels, include_lowest=True)
    #####################################################

    # Function to sample income from a normal distribution
    # Sample income for each individual based on their age and education group
    persons_df = persons_df.reset_index().merge(income_summary, on=['age_group', 'education_group'], how='left').set_index("person_id")
    persons_df['new_earning'] = persons_df.apply(lambda row: sample_income(row['mu'], row['sigma']), axis=1)

    persons_df["exit_workforce"] = exit_workforce_list
    persons_df["exit_workforce"].fillna(2, inplace=True)

    persons_df["remain_unemployed"] = stay_unemployed_list
    persons_df["remain_unemployed"].fillna(2, inplace=True)

    # Update education levels
    persons_df["worker"] = np.where(persons_df["exit_workforce"]==1, 0, persons_df["worker"])
    persons_df["worker"] = np.where(persons_df["remain_unemployed"]==0, 1, persons_df["worker"])

    persons_df["work_at_home"] = persons_df["work_at_home"].fillna(0)

    persons_df.loc[persons_df["exit_workforce"]==1, "earning"] = 0
    persons_df["earning"] = np.where(persons_df["remain_unemployed"]==0, persons_df["new_earning"], persons_df["earning"])

    # TODO: Similarly, do something for work from home
    agg_households = persons_df.groupby("household_id").agg(
        sum_workers = ("worker", "sum"),
        income = ("earning", "sum")
    )
    
    agg_households["hh_workers"] = np.where(
        agg_households["sum_workers"] == 0,
        "none",
        np.where(agg_households["sum_workers"] == 1, "one", "two or more"))
          
    # TODO: Make sure that the actual workers don't get restorted due to difference in indexing
    # TODO: Make sure there is a better way to do this
    #orca.get_table("households").update_col("workers", agg_households["workers"])
    #orca.get_table("households").update_col("hh_workers", agg_households["hh_workers"])
    households_df.update(agg_households)

    workers = persons_df[persons_df["worker"] == 1]
    exiting_workforce_df = orca.get_table("exiting_workforce").to_frame()
    entering_workforce_df = orca.get_table("entering_workforce").to_frame()
    if entering_workforce_df.empty:
        entering_workforce_df = pd.DataFrame(
            data={"year": [year], "count": [persons_df[persons_df["remain_unemployed"]==0].shape[0]]}
        )
    else:
        entering_workforce_df_new = pd.DataFrame(
            data={"year": [year], "count": [persons_df[persons_df["remain_unemployed"]==0].shape[0]]}
        )
        entering_workforce_df = pd.concat([entering_workforce_df, entering_workforce_df_new])

    if exiting_workforce_df.empty:
        exiting_workforce_df = pd.DataFrame(
            data={"year": [year], "count": [persons_df[persons_df["exit_workforce"]==1].shape[0]]}
        )
    else:
        exiting_workforce_df_new = pd.DataFrame(
            data={"year": [year], "count": [persons_df[persons_df["exit_workforce"]==1].shape[0]]}
        )
        exiting_workforce_df = pd.concat([exiting_workforce_df, exiting_workforce_df_new])
        
    orca.add_table("entering_workforce", entering_workforce_df)
    orca.add_table("exiting_workforce", exiting_workforce_df)
    orca.add_table("persons", persons_df[persons_cols])
    orca.add_table("households", households_df[households_cols])

def sample_income(mean, std):
    return np.random.lognormal(mean, std)