"""Plot average trips on weekdays by hour of day for dry and wet weather.
"""
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def read_data(name, region_code, simulation_folder, calibration_folder):
    if scenario_name:
        #TODO add path with scenario (if we have scenario)
        pass
    else:
        SIM_PATH_NAME = os.path.join(simulation_folder, name+"_"+region_code+".csv")
        CAL_PATH_NAME = os.path.join(calibration_folder, name+"_over_time_obs"+".csv")

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
    elif name=="households_over_time":
        plt.ylabel("Number of Households (Millions)", size=16)
    elif name=="births_over_time":
        plt.ylabel("Number of Births", size=16)
    elif name=="mortalities":
        plt.ylabel("Number of Mortalities", size=16)
    # count
    # plt.yticks(ticks=np.arange(0, 2.6e6, 0.5e6), labels=np.arange(0, 2.6, 0.5))
    plt.legend(title="")
    plt.grid()
    os.makedirs(os.path.join("outputs", "figures"), exist_ok=True)
    if not scenario_name:
        fig_name = name+"_"+region_code
    else:
        fig_name = name+"_"+region_code+"_"+scenario_name
    plt.savefig(os.path.join("outputs", "figures", fig_name + ".png"))
    print(f"Figure plotted in {fig_name}!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plotting script.")
    parser.add_argument("-r", "--region_code", default='06197001', help="region code")
    parser.add_argument("-sn", "--scenario_name", help="scenario_name")

    args = parser.parse_args()
    region_code = args.region_code
    scenario_name = args.scenario_name if args.scenario_name else False

    simulation_output_folder = os.path.join("outputs", "simulation", region_code)
    calibration_output_folder = os.path.join("outputs", "calibration", region_code)
    print("output simulation path: ", simulation_output_folder)
    print("output calibration path: ", calibration_output_folder)
    
    #names = ["mortalities", "births", "pop_size", "hh_size"]
    # TODO: the file names in 'outputs/simulation' should be aliged with 'outputs/calibration'
    names = ["mortalities"]
    
    for name in names:
        results = read_data(name, region_code, simulation_output_folder, calibration_output_folder)
        plot_results(name, results, region_code, scenario_name)