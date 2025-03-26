import os
import orca
import warnings
import indicators
import pandas as pd

# TODO: This seems to be logging. Integrate all logging in a consistent way.
print("Importing models for region", orca.get_injectable("region_code"))
# TODO: Handle this
warnings.filterwarnings("ignore")
# -----------------------------------------------------------------------------------------
# DEMOS
# -----------------------------------------------------------------------------------------
@orca.step("income_stats")
def income_stats(persons, households):
    """Function to print the number of households from both the households and pers

    Args:
        persons (DataFrame): Pandas DataFrame of the persons table
        households (DataFrame): Pandas DataFrame of the households table
    """

    persons_df = orca.get_table("persons").local
    households_df = orca.get_table("households").local
    print("Households median Income: ", households_df["income"].median())
    print("Households median income persons table: ", persons_df.groupby("household_id")["earning"].sum().median())


@orca.step("update_income")
def update_income(persons, households, year):
    """
    Updating income for persons and households

    Args:
        persons (DataFrameWrapper): DataFrameWrapper of persons table
        households (DataFrameWrapper): DataFrameWrapper of households table
        year (int): simulation year
    """
    # Pulling data, income rates, and county IDs
    persons_df = orca.get_table("persons").local
    households_df = orca.get_table("households").local

    households_local_cols = households_df.columns
    persons_local_cols = persons_df.columns
    # print(persons_local_cols)
    hh_counties = households_df["lcm_county_id"].copy()

    income_rates = orca.get_table("income_rates").to_frame()
    income_rates = income_rates[income_rates["year"] == year]

    persons_df = (persons_df.reset_index().merge(hh_counties.reset_index(), on=["household_id"]).set_index("person_id"))
    persons_df = (persons_df.reset_index().merge(income_rates, on=["lcm_county_id"]).set_index("person_id"))
    persons_df["earning"] = persons_df["earning"] * (1 + persons_df["rate"])

    new_incomes = persons_df.groupby("household_id").agg(income=("earning", "sum"))

    households_df.update(new_incomes)
    households_df["income"] = households_df["income"].astype(int)
    persons_df["member_id"] = persons_df.groupby("household_id")["relate"].rank(method="first", ascending=True).astype(int)
    persons_local_columns = orca.get_injectable("persons_local_cols")
    orca.add_table("persons", persons_df[persons_local_columns])
    orca.add_table("households", households_df[households_local_cols])
    orca.add_table("persons", persons_df[persons_local_cols])
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


# TODO refactor this method so it can be cleanly used with any model
# Might need to add some stuff (update more variables) or have a flag to determine which function is calling this
def update_households(alive, dead, old_incomes, subtract=True):
    """
    Function to update the households characteristics, namely income, num of workers, race, and age of head.
    """
    # By default the dead variable represents people dying or leaving a household
    if subtract:
        alive = alive.sort_values("relate")
        dead_aggs = dead.groupby("household_id").agg({"earning": "sum"})

        aggregates = alive.groupby("household_id").agg(
            {"earning": "sum", "worker": "sum", "race_id": "first", "age": "first"}
        )
        # print(list(orca.get_table("households").to_frame().columns))
        # TODO -- REPLACE THESE OPERATIONS BY PANDAS OPERATIONS
        orca.get_table("households").update_col(
            "income", old_incomes.subtract(dead_aggs["earning"], fill_value=0)
        )
        orca.get_table("households").update_col("workers", aggregates["worker"])
        orca.get_table("households").update_col("race_of_head", aggregates["race_id"])
        orca.get_table("households").update_col("age_of_head", aggregates["age"])
    # In a certain case, the dead variable actually represents living people being added -- What is this case?
    else:
        aggregates = dead.groupby("household_id").agg(
            {"earning": "sum", "worker": "sum"}
        )
        house_df = orca.get_table("households").to_frame(columns=["income", "workers"])
        # TODO -- REPLACE THESE OPERATIONS WITH PANDAS OPERATIONS
        orca.get_table("households").update_col(
            "income", house_df["income"].add(aggregates["earning"], fill_value=0)
        )
        orca.get_table("households").update_col(
            "workers", house_df["workers"].add(aggregates["worker"], fill_value=0)
        )

    # return households


@orca.step("print_marr_stats")
def print_marr_stats():
    persons_stats = orca.get_table("persons").local
    persons_stats = persons_stats[persons_stats["age"]>=15]
    print(persons_stats["MAR"].value_counts().sort_values())


# -----------------------------------------------------------------------------------------
# POSTPROCESSING
# -----------------------------------------------------------------------------------------
@orca.step("generate_outputs")
def generate_outputs(year, base_year, forecast_year, tracts):
    print(
        "Generating outputs for (year {}, forecast year {})...".format(
            year, forecast_year
        )
    )
    if not os.path.exists("runs"):
        os.makedirs("./runs")

    if orca.get_injectable("all_local"):
        return

    if year == base_year:
        indicators.export_indicator_definitions()

    cfg = orca.get_injectable("output_parameters")

    # Layer indicators
    indicators.gen_all_indicators(cfg["output_indicators"], year)

    # Chart indicators
    # if year == forecast_year:
    #     chart_data, geo_small, geo_large = indicators.prepare_chart_data(cfg, year)
    #     # indicators.gen_all_charts(cfg['output_charts'], base_year, forecast_year, chart_data, geo_large)

    # Calibration metrics for pdf report in calibration/microsimulation routine
    if (year == 2018) and (orca.get_injectable("local_simulation") == True):
        indicators.gen_calibration_metrics(tracts)


@orca.step("generate_metrics")
def generate_metrics(year, persons, households):
    """
    Update metrics of persons and households.

    Args:
        year (int): simulation year
        persons (DataFrameWrapper): DataFrameWrapper of the persons table

    Returns:
        None
    """
    persons_df = orca.get_table("persons").local
    households_df = orca.get_table("households").local
    age_over_time = orca.get_table("age_dist_over_time").to_frame()
    pop_over_time = orca.get_table("pop_size_over_time").to_frame()
    hh_over_time = orca.get_table("hh_size_over_time").to_frame()
    students = orca.get_table("student_population").to_frame()
    # age
    if age_over_time.empty:
        age_over_time = (
            persons_df.groupby("sex")["age"]
            .value_counts(
                bins=[
                    0,
                    0.9,
                    4,
                    9,
                    14,
                    19,
                    24,
                    29,
                    34,
                    39,
                    44,
                    49,
                    54,
                    59,
                    64,
                    69,
                    74,
                    79,
                    84,
                    89,
                    94,
                    99,
                    1000,
                ],
                sort=False,
            )
            .reset_index(name="count_" + str(year))
            .T
        )
    else:
        age_over_time_new = (
            persons_df.groupby("sex")["age"]
            .value_counts(
                bins=[
                    0,
                    0.9,
                    4,
                    9,
                    14,
                    19,
                    24,
                    29,
                    34,
                    39,
                    44,
                    49,
                    54,
                    59,
                    64,
                    69,
                    74,
                    79,
                    84,
                    89,
                    94,
                    99,
                    1000,
                ],
                sort=False,
            )
            .reset_index(name="count_" + str(year))
            .T
        )
        age_over_time = pd.concat([age_over_time, age_over_time_new])

    # pop
    if pop_over_time.empty:
        pop_over_time = pd.DataFrame.from_dict({
            "year": [str(year)],
            "count":  [persons_df.index.unique().shape[0]]
            })
    else:
        pop_over_time_new = pd.DataFrame.from_dict({
            "year": [str(year)],
            "count":  [persons_df.index.unique().shape[0]]
            })
        pop_over_time = pd.concat([pop_over_time, pop_over_time_new])

    # hh
    if hh_over_time.empty:
        hh_over_time = households_df.reset_index().groupby(["lcm_county_id","hh_size"]).agg(count = ("household_id", "size")).reset_index()
        hh_over_time["year"] = year
        hh_over_time["year"] = hh_over_time["year"].astype(str)
        hh_over_time["lcm_county_id"] = hh_over_time["lcm_county_id"].astype(str)
    else:
        hh_over_time_new = households_df.reset_index().groupby(["lcm_county_id","hh_size"]).agg(count = ("household_id", "size")).reset_index()
        hh_over_time_new["year"] = year
        hh_over_time_new["year"] = hh_over_time_new["year"].astype(str)
        hh_over_time_new["lcm_county_id"] = hh_over_time_new["lcm_county_id"].astype(str)
        hh_over_time = pd.concat([hh_over_time, hh_over_time_new])

    # students
    if students.empty:
        students = pd.DataFrame.from_dict({
            "year": [str(year)],
            "count":  [persons_df[
                    persons_df["edu"].isin(
                        [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
                    )
                ]["student"].sum()]
            })
    else:
        new_students = pd.DataFrame.from_dict({
            "year": [str(year)],
            "count":  [persons_df[
                    persons_df["edu"].isin(
                        [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
                    )
                ]["student"].sum()]
            })
        students = pd.concat([students, new_students])
        # print(students)
    # marrital = orca.get_table("marrital").to_frame()
    # if marrital.empty:
    #     persons_stats = persons_df[persons_df["age"]>=15]["MAR"].value_counts().reset_index()
    #     marrital = pd.DataFrame(persons_stats)
    #     marrital["year"] = year
    # else:
    #     persons_stats = persons_df[persons_df["age"]>=15]["MAR"].value_counts().reset_index()
    #     new_marrital = pd.DataFrame(persons_stats)
    #     new_marrital["year"] = year
    #     marrital = pd.concat([marrital, new_marrital])
    # print(marrital)
        
    orca.add_table("age_dist_over_time", age_over_time)
    orca.add_table("pop_size_over_time", pop_over_time)
    orca.add_table("student_population", students)
    orca.add_table("hh_size_over_time", hh_over_time)
    # orca.add_table("marrital", marrital)


# -----------------------------------------------------------------------------------------
# STEP DEFINITION
# -----------------------------------------------------------------------------------------

all_local = orca.get_injectable("all_local")
if orca.get_injectable("running_calibration_routine") == False:
    if orca.get_injectable("local_simulation") == True:
        demo_models = [
            # "update_age",
            # "laborforce_model",
            "households_reorg",
            "kids_moving_model",
            "fatality_model",
            "birth_model",
            "education_model",
            "export_demo_stats",
        ]
        steps_all_years = (
            demo_models
        )
    orca.add_injectable("sim_steps", steps_all_years)
