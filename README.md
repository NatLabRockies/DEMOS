# Demographic Microsimulator(DEMOS)

>**TODO:** The previous description from Urbansim it outdated. Add description of DEMOS, aligning with the code (paper).

# Usage
## Docker Container
The docker image for demos is stored in `registry/demos:latest`. The input data and configuration file are fed to the container through volumes. Alternatively, we provide a `docker-compose` workflow that can be used.

For running the `docker-compose` workflow:
```bash
DEMOS_CONFIG_PATH=<path-to-config> DEMOS_DATA_DIR=<path-to-data-dir> docker-compose up
```

By default `DEMOS_CONFIG_PATH` is set to `./demos_config.toml` and `DEMOS_DATA_DIR` is set to `./data`, so if `data` and `demos_config.toml` are part of the cuurent directory, no additional input is needed.

Alternatively,
```bash
docker run --volume <path-to-config>:/demos/config.toml:ro --volume <path-to-data-dir>:/demos/data --platform=linux/amd64 demos
```

### Building the docker image (development only)
```bash
docker build -t demos:0.0.1 --platform=linux/amd64 -f Dockerfile .
```

## From Source

This repository contains only code and configuration/setup files necessary 


1. clone this repository into your local machine

2. create environment if needed

	```
	conda create --name {myenv} python=3.8
	```

3. Enter into DEMOS environment

	```
	conda activate {myenv}
	```

4. Install all packages if needed

	```
	pip install -r requirements.txt
	```

5. Download input data [data_nrel.zip](https://app.box.com/s/tox2nflumia2g4n6rk2i0navca9pskep). You may also refer to the data description [here](https://cloud.urbansim.com/docs/general/documentation/urbansim%20block%20model%20data.html)

6. Put all files of input data into `DEMOS_NREL/demos/data`

7. Run DEMOS. First, enter into `DEMOS_NREL/demos`, then run:

	```
	python -u simulate.py -c -y 2011 -cf custom -l -r 06197001 -s 100
	```

    the following are the arguments used in the above command:
	```
	  -r region_code, --region_code region_code
							region fips code
	  -y year, --year year  forecast year to simulate to
	  -c, --calibrated      whether to run with calibrated coefficients
	  -cf calibrated_folder, --calibrated_folder calibrated_folder
							name of the calibration folder to read configs from
	  -sg, --segmented      run with segmented lcms
	  -l, --all_local       no cloud access whatsoever
	  -i input_year, --input_year input_year
							input data (base) year
	  -f freq_interval, --freq_interval freq_interval
							intra-simulation frequency interval
	  -o output_fname, --output_fname output_fname
							output file name
	  -ss skim_source, --skim_source skim_source
							skims format, e.g. "beam", "polaris"
	  -rm, --random_matching random matching in the single to x model to 
							 reduce computational time due to 
							 the matchmaking process
	  -s --random_seed random seed settng
	```

8. simulation results
the demos simulation will produce the following sets of data and results:
  - a synthetic population file showing the evolution of the synthetic population throughout the simulation years. the file should be named `model_data_<scenario_name_output_year>.h5` in directory `DEMOS_NREL/demos/data`.
  - series of aggregated statistics for the population size, number of households, household size distribution, gender distribution, number of births, number of mortalities, number of student enrollments, number of total marriages, number of total divorces, the age distribution of the synthetic population, and income distribution for each simulation year. The files are located at `DEMOS_NREL/demos/outputs/simulation`

## ii. project structure
>**TODO:** this part still need to be complemented

the main folder of this repository contains several python scripts that contain the different steps necessary to import, process, and run the demos framework. the following is a description of the different folder and scripts used to run the demos simulation

1. the `configs\` directory: this folder contains the different `.yaml` configuration files to run each of the demos and urbansim models. the configuration files for each region are located in subdirectories with the name of the region
2. the `data\` directory: contains all the data needed to run the simulation
3. `variables.py`: this script defines all the temporary variables needed to run the models. each variable is created as an orca column.
4. `datasources.py`: this script imports all the necessary data for the specified simulation region and create simulation output folders, if needed.
5. `models.py`: this script defines all the models as orca steps and defines all pre-processing and post-processing steps needed for each of the models.
6. `simulate.py`: this script defines all the simulation parameters and runs the rest of the scripts desribed above.
7. the `outputs\` directory: contains the different results produced by the simulation. simulation results for each region are stored in their respective subdirectories.