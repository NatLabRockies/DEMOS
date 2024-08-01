import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import namedtuple

cmp = namedtuple("cmp", ["name", "calibration", "simulation"])

def read_data(cmp, region_code, simulation_folder, calibration_folder):
    if scenario_name:
        #TODO add path with scenario (if we have scenario)
        pass
    else:
        SIM_PATH_NAME = os.path.join(simulation_folder, cmp.simulation + ".csv")
        CAL_PATH_NAME = os.path.join(calibration_folder, cmp.calibration + ".csv")

    simulation = pd.read_csv(SIM_PATH_NAME)
    calibration = pd.read_csv(CAL_PATH_NAME)
    simulation["model"] = "Simulation"
    calibration["model"] = "Observed"
    combined_results = pd.concat([simulation, calibration]).reset_index(drop=True)
    combined_results = combined_results[combined_results["year"].between(2011, 2019, inclusive='both')].reset_index(drop=True)
    combined_results = combined_results.sort_values(by=["model", "year"], ascending=[True, True]).reset_index(drop=True)
    return combined_results

def plot_results(name, data, region_code, scenario_name):
    plt.figure(figsize=(20, 10))
    sns.barplot(x="year", y="count", hue="model", data=data)
    plt.xlabel("Year", size=16)
    if name=="population":
        plt.ylabel("Population Size (Millions)", size=16)
    elif name=="household":
        plt.ylabel("Number of Households (Millions)", size=16)
    elif name=="birth":
        plt.ylabel("Number of Births", size=16)
    elif name=="mortality":
        plt.ylabel("Number of Mortalities", size=16)
    # count
    # plt.yticks(ticks=np.arange(0, 2.6e6, 0.5e6), labels=np.arange(0, 2.6, 0.5))
    plt.legend(title="")
    plt.grid()
    os.makedirs("figures", exist_ok=True)
    if not scenario_name:
        fig_name = name+"_"+region_code
    else:
        fig_name = name+"_"+region_code+"_"+scenario_name
    plt.savefig(os.path.join("", "figures", fig_name + ".png"))
    print(f"Figure plotted in {fig_name}!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plotting script.")
    parser.add_argument("-r", "--region_code", default='06197001', help="region code")
    parser.add_argument("-sn", "--scenario_name", help="scenario_name")

    args = parser.parse_args()
    region_code = args.region_code
    scenario_name = args.scenario_name if args.scenario_name else False

    simulation_output_folder = os.path.join("simulation", region_code)
    calibration_output_folder = os.path.join("calibration", region_code)
    print("output simulation path: ", simulation_output_folder)
    print("output calibration path: ", calibration_output_folder)
    
    # add compare objects here.
    # create cmp object follows below:
    # model name, file name of calibration, file name of simulation.
    cmps = [cmp("mortality", "mortalities_over_time_obs", "mortalities_"+str(region_code)),
            cmp("birth", "births_over_time_obs", "btable_"+str(region_code))]
    
    for cmp in cmps:
        results = read_data(cmp, region_code, simulation_output_folder, calibration_output_folder)
        plot_results(cmp.name, results, region_code, scenario_name)