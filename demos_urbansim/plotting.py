"""Plot average trips on weekdays by hour of day for dry and wet weather.
"""
import sys
import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Args ------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Plotting script.")
parser.add_argument("-png", "--save_png", action="store_true",
    help="Output PNG file.")
parser.add_argument("-sy", "--start_year", type=int, default='2011', help="Start year",
    type=str)
parser.add_argument("-ey", "--end_year", type=int, default='2020', help="End year",
    type=str)
parser.add_argument("-r", "--region_code", type=str, help="region code")
parser.add_argment("-sn", "--scenario_name", type=str, help="scenario_name")

args = parser.parse_args()
png = args.png if args.png else False
start_year = args.start_year
end_year = args.end_year
region_code = args.region_code
scenario_name = args.scenario_name if args.scenario_name else False

simulation_output_folder = "outputs/%s/simulation/", region_code
calibration_output_folder = "outputs/%s/calibration/", region_code

def read_data(name, region_code, simulation_folder, calibration_folder):
    if not scenario_name:
        SIM_PATH_NAME = simulation_folder+name+region_code+scenario_name+".csv"
        CAL_PATH_NAME = calibration_folder+name+region_code+scenario_name+".csv"
    else:
        SIM_PATH_NAME = simulation_folder+name+region_code+".csv"
        CAL_PATH_NAME = calibration_folder+name+region_code+".csv"

    simulation = pd.read_csv(SIM_PATH_NAME)
    calibration = pd.read_csv(CAL_PATH_NAME)
    simulation["model"] = "Simulation"
    calibration["model"] = "Observed"
    combined_results = pd.concat([simulation, calibration]).reset_index(drop=True)
    combined_results = combined_results[combined_results["year"].between(2011, 2019, inclusive='both')].reset_index(drop=True)
    combined_results = combined_results.sort_values(by=["model", "year"],
                                                    ascending=[True, True]).reset_index(drop=True)
    return combined_results

def plot_results(name, data, region_code, scenario_name):
    plt.figure(figsize=(20, 10))
    sns.barplot(x="year", y="count", hue="model", data=data)
    plt.xlabel("Year", size=16)
    if name=="pop_over_time":
        plt.ylabel("Population Size (Millions)", size=16)
    elif name=="student_counts":
        plt.ylabel("Student Enrollment", size=16)
    elif name=="hh_size_over_time":
        plt.ylabel("Number of Households (Millions)", size=16)
    elif name=="births_over_time":
        plt.ylabel("Number of Births", size=16)
    elif name=="mortalities":
        plt.ylabel("Number of Mortalities", size=16)
    # count
    # plt.yticks(ticks=np.arange(0, 2.6e6, 0.5e6), labels=np.arange(0, 2.6, 0.5))
    plt.legend(title="")
    plt.grid()
    if not scenario_name:
        fig_name = name+"_"+region_code
    else:
        fig_name = name+"_"+region_code+"_"+scenario_name
    plt.savefig(fig_name + ".png")
    print("Figure plotted!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plotting script.")
    parser.add_argument("-png", "--save_png", action="store_true",
        help="Output PNG file.")
    parser.add_argument("-sy", "--start_year", default='2011', help="Start year",
        type=str)
    parser.add_argument("-ey", "--end_year", default='2020', help="End year",
        type=str)
    parser.add_argument("-r", "--region_code", help="region code")
    parser.add_argment("-sn", "--scenario_name", help="scenario_name")

    args = parser.parse_args()
    png = args.png if args.png else False
    start_year = args.start_year
    end_year = args.end_year
    region_code = args.region_code
    scenario_name = args.scenario_name if args.scenario_name else False

    simulation_output_folder = "outputs/%s/simulation/", region_code
    calibration_output_folder = "outputs/%s/calibration/", region_code
    
    names = ["mortalities", "births", "students", "pop_size", "hh_size"]
    
    for name in names:
        results = read_data(name, region_code, simulation_output_folder,
                            calibration_output_folder)
        plot_results(name, results, region_code, scenario_name)