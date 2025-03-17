import orca
import pandas as pd

orca.add_table("age_evolution", pd.DataFrame())


def compute_age_dist(year, age):
    return {
        "year": [year],
        "20-30": [((age >= 20) & (age <= 30)).sum()],
        "31-40": [((age >= 31) & (age <= 40)).sum()],
        "41-50": [((age >= 41) & (age <= 50)).sum()],
        "51-70": [((age >= 51) & (age <= 70)).sum()],
        "70+":   [((age > 70) ).sum()]
    }

@orca.step("export_demo_stats")
def export_demo_stats(year, forecast_year, persons):
    """
    Export Demographic Stats tables

    Args:
        year (int): simulation year
        forecast_year (int): final forecast year

    Returns:
        None
    """
    orca.add_table("age_evolution", compute_age_dist(year, persons.to_frame(["age"])))

    if year == forecast_year:
        export("pop_over_time")
        export("hh_size_over_time")
        export("age_over_time")
        export("edu_over_time")
        export("income_over_time")
        export("kids_move_table")
        export("divorce_table")
        export("marriage_table")
        export("btable")
        export("age_dist_over_time")
        export("pop_size_over_time")
        export("student_population")
        export("mortalities")
        export("btable_elig")
        export("marrital")
    
def export(table_name):
    """
    Export the tables

    Args:
        table_name (string): Name of the orca table
    """
    
    region_code = orca.get_injectable("region_code")
    output_folder = orca.get_injectable("output_folder")
    df = orca.get_table(table_name).to_frame()
    # scenario_name = orca.get_injectable("scenario_name")
    # if scenario_name is False:
    #     csv_name = table_name + "_" + region_code +".csv"
    # else:
    #     csv_name = table_name + "_" + region_code + "_" + scenario_name + ".csv"
    csv_name = table_name + "_" + region_code +".csv"
    df.to_csv(output_folder+csv_name, index=False)