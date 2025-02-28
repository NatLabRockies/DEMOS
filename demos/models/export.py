import orca

@orca.step("export_demo_stats")
def export_demo_stats(year, forecast_year):
    """
    Export Demographic Stats tables

    Args:
        year (int): simulation year
        forecast_year (int): final forecast year

    Returns:
        None
    """

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