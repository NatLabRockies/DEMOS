# DEMOS

This repository contains code that runs simulations for the DEMOS model suite, as fully integrated with Urbansim models. The current DEMOS model suite is implemented for the San Francisco Bay Area and the Austin Area. Several demographic models are implemented in the current framework, in addition to several Urbansim models. Detailed documentation of the models will be added to this repository. Currently, the following demographic models are implemented:

- Aging Model: simulates the age evolution of the synthetic population
- Birth Model: simulates new births in the population in each year
- Mortality Model: simulates new mortalities in the population in each year
- Education Model: simulates dropouts out of the student population in each year
- Single to X Model: simulates changes in partnership status from Single to either cohabitating with a partner or getting married
- Cohabitation to X Model: simulates changes in partnership status from cohabitating with a partner to getting married or getting divorced
- Divorce Model: simulates new divorces in the synthetic population each year
- Kids leaving Household: simulates kids leaving their current households in each year
- In-Migration/Out-migration models: simulates households/individuals migrating into or out of the area each year
- Income growth models: simulates income growth for the population in each year

This repository is currently in active development and new updates will be continuously added. For any issues with the current codebase or to suggest improvements, please open an issue or create a separate branch to add your feature and submit pull request for review.


## I. Setup Guide

This repository contains only code and configuration/setup files necessary 


1. Clone this repository into your local machine

2. Create Environment


3. Run the local setup

```
python3 setup.py develop
```

4. Install dependencies (**WORK IN PROGRESS**)

The current dependencies installation is based on Unix based systems. Installation instructions for Windows systems will be added later.

It is also preferred to create a new environment to isolate the environment for this software from all other installed environments. This is to avoid creating any conflicts with already installed libraries or software. The recommended way to install environments is through Anaconda. Follow this guide to create a new environment and install the dependendecies in it.

```
apt-get update

apt-get install -y gcc libhdf5-serial-dev
```

Install Anaconda by following these instructions for [Linux](https://docs.anaconda.com/anaconda/install/linux/) or[MacOS](https://docs.anaconda.com/anaconda/install/mac-os/).

Create a conda environment
```
conda create -n DEMOS python=3.8
```
Activate the created conda environment

```
conda activate DEMOS
```
Install the project requirements.
```
pip install -r requirements.txt
```
If you run into any issues in setting up your environment, please submit an issue in this repository.

5. Download Relevant Data

Before running the simulation framework, create a data folder within the main project directory by typing the following in your terminal:

```
cd DEMOS

mkdir data
```

Access the data directory and download the relevant input data using the following:

- SF Bay Area
```
wget https://bayarea-urbansim.s3.us-east-2.amazonaws.com/custom_mpo_06197001_model_data.h5
wget -O bay_area_skims.csv.gz https://beam-outputs.s3.amazonaws.com/output/sfbay/sfbay-smartbaseline-WALK-TAZ-activitySimSkims-allTimePeriods-r5%2Bgh__2021-02-24_21-55-42_gnh/ITERS/it.0/0.activitySimODSkims.UrbanSim.TAZ.Full.csv.gz
```

- Austin, Texas Area:
```
Links will be added here
```

6. SKIMS preprocessing

Before running any simulations, it's necessary to preprocess the skims files downloaded in Step 5 for your relevant region. To process the skims file, first activate your environment using the following command:
```
conda activate DEMOS_ENV
```

Access the main DEMOS project folder and running the skims processing file using the following command:
```
python process_skims.py
```

7. Running Simulation

Before running the simulation, navigate to the project directory and activate the created conda environment:

```
conda activate DEMOS
```

To run a simple simulation of DEMOS for the bay area (region code 06197001) from 2010 to 2020, run the following command:
```
python -u simulate.py -c -cf custom -l -r 06197001
```

The general command for running the DEMOS simulation is the following:
```
python simulate.py -- -c -cf custom -l -sg -r <region ID> -i <input_year> -y <forecast year> -f <iter year frequency> -ss <skims source> -rm
```

The following are the arguments used in the above command:
```
-r REGION_CODE, --region_code REGION_CODE
                        region fips code
  -y YEAR, --year YEAR  forecast year to simulate to
  -c, --calibrated      whether to run with calibrated coefficients
  -cf CALIBRATED_FOLDER, --calibrated_folder CALIBRATED_FOLDER
                        name of the calibration folder to read configs from
  -sg, --segmented      run with segmented LCMs
  -l, --all_local       no cloud access whatsoever
  -i INPUT_YEAR, --input_year INPUT_YEAR
                        input data (base) year
  -f FREQ_INTERVAL, --freq_interval FREQ_INTERVAL
                        intra-simulation frequency interval
  -o OUTPUT_FNAME, --output_fname OUTPUT_FNAME
                        output file name
  -ss SKIM_SOURCE, --skim_source SKIM_SOURCE
                        skims format, e.g. "beam", "polaris"
  -rm, --random_matching Random matching in the Single to X model to 
                         reduce computational time due to 
                         the matchmaking process
```

This command will direct the software to the correct input data and configuration files and run the simulation based on the specified parameters. After the simulation is completed, the software will save output data and summary statistics in the output folder. The analyst can then use these saved objects to run any additional analyses necessary.

8. Simulation results
The DEMOS simulation will produce the following sets of data and results:
  - A synthetic population file showing the evolution of the synthetic population throughout the simulation years. The file should be named `model_data_SCENARIO_NAME_OUTPUT_YEAR.h5`.
  - Series of aggregated statistics for the population size, number of households, household size distribution, gender distribution, number of births, number of mortalities, number of student enrollments, number of total marriages, number of total divorces, the age distribution of the synthetic population, and income distribution for each simulation year.

## II. Project Structure

The main folder of this repository contains several python scripts that contain the different steps necessary to import, process, and run the DEMOS framework. The following is a description of the different folder and scripts used to run the DEMOS simulation

1. The `configs\` directory: this folder contains the different `.yaml` configuration files to run each of the DEMOS and urbansim models. The configuration files for each region are located in subdirectories with the name of the region
2. The `data\` directory: contains all the data needed to run the simulation
3. `variables.py`: this script defines all the temporary variables needed to run the models. Each variable is created as an Orca column.
4. `datasources.py`: this script imports all the necessary data for the specified simulation region and create simulation output folders, if needed.
5. `models.py`: this script defines all the models as orca steps and defines all pre-processing and post-processing steps needed for each of the models.
6. `simulate.py`: this script defines all the simulation parameters and runs the rest of the scripts desribed above.
7. The `outputs\` directory: contains the different results produced by the simulation. Simulation results for each region are stored in their respective subdirectories.


**DETAILED DESCRIPTION OF EACH OF THE SCRIPTS WILL BE ADDED HERE AS THE STRUCTURE IS BEING MODIFIED FOR GENERALIZABILITY**