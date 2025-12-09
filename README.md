# Demographic Microsimulator (DEMOS)

[![Docs](https://github.com/NREL/DEMOS_NREL/actions/workflows/docs.yml/badge.svg)](https://nrel.github.io/DEMOS_NREL/)

## Overview
Demographic Microsimulator (DEMOS) is an agent-based simulation framework used to evolve population demographic characteristics or lifecycle events such as education, marital status etc. DEMOS modules are designed to capture the interdependencies of short-term and long-term lifecycle events often influential in downstream transportation and land use modeling. An important facet of DEMOS is that it model captures the impact of an agent's demographic characteristics in year 't' on their demographic status in year 't+1'. This has important consequences on medium- and long-term transportation decisions such as household vehicle transactions (i.e., buying, selling, or replacing a vehicle), or work location choice. The key features of DEMOS include the modeling of 10+ lifecycel events, behaivoral patterns supported by long running panel data, the representation of model interdependencies, and flexible simulation structure and modularity. 

The overall framework of DEMOS consists of three major components: i) migration module, ii) individual-level demographic evolution, and iii) household-level demographic evolution. The demographic evolution process is initiated with a baseline-year (t) synthetic population. Household-, and individual-level characteristics are then updated and provided as inputs to subsequent year's (t+1) population evolution. This process is repeated to evolve the population of a study region over a span of 10-30 years, which is the general duration for long range transportation planning. The model as such can be used to evolve populations for any duration of interest to the user.

DEMOS technical memorandum can be found here. The memo provides an overview of DEMOS functionality, framework, input and output data and how DEMOS can be utilized to enhance transportation planning process and broder application scenarios.

Inerested users can refer to the paper below for more details of DEMOS methodology.

*Sun, Bingrong, Shivam Sharda, Venu M. Garikapati, Mohamed Amine Bouzaghrane, Juan Caicedo, Srinath Ravulaparthy, Isabel Viegas de Lima, Ling Jin, C. Anna Spurlock, and Paul Waddell. "Demographic Microsimulator for Integrated Urban Systems: Adapting Panel Survey of Income Dynamics to Capture the Continuum of Life." Transportation Research Record (2025): 03611981251333339.*

## Usage
> A public Docker image of DEMOS has been released. Please follow the `From Source` instructions.

### Docker Container
The docker image for demos is stored in `registry/demos:latest`. The input data and configuration file are fed to the container through volumes. Alternatively, we provide a `docker-compose` workflow that can be used.

For running the `docker-compose` workflow:
```bash
DEMOS_CONFIG_PATH=<path-to-config> DEMOS_DATA_DIR=<path-to-data-dir> docker-compose up
```

By default `DEMOS_CONFIG_PATH` is set to `./demos_config.toml` and `DEMOS_DATA_DIR` is set to `./data`, so if `data` and `demos_config.toml` are part of the current directory, no additional input is needed.

Alternatively,
```bash
docker run --volume <path-to-config>:/demos/config.toml:ro --volume <path-to-data-dir>:/demos/data --platform=linux/amd64 demos
```

#### IMPORTANT for MacOS and Windows users
> Docker imposes a global limit on how much RAM containers can allocate. DEMOS easily surpases those limits, so in order to run DEMOS in Docker, users need to access the Docker Desktop GUI and `Preferences → Resources → Memory → Increase it (at least 16-20gb)`

#### Building the docker image (development only)
```bash
docker build -t demos:0.0.1 --platform=linux/amd64 -f Dockerfile .
```

### From Source

1. Clone this repository
	```
	git clone https://github.com/NREL/DEMOS_NREL.git
	```

1. Create a virtual environment.

	**If using conda**, prefer the provided `.lock` files
	```
	conda create --name demos-env --file conda-{system}.lock
	```

	Alternatively, create a `python 3.10` environment and install dependencies
	```
	conda create -n demos-env python=3.10
	conda activate demos-env
	pip install .
	```

## Running DEMOS

DEMOS requires a series of input tables. Example tables are provided [here](https://app.box.com/s/tox2nflumia2g4n6rk2i0navca9pskep) for internal NREL use. It is recommended to store all the input values in the folder `data` in root of the project, but absolute values can be used by specifying them in the configuration file. You may also refer to the data description [here](https://cloud.urbansim.com/docs/general/documentation/urbansim%20block%20model%20data.html)

To run demos:
```
python simulate.py -cfg {configuration_file}
```



A default configuration file is provided in `configuration/demos_config.toml`. The `[[tables]]` entries outline tables to be loaded. For example, we can load the `persons` and `households` table from an H5 source:

```toml
[[tables]]
file_type = "h5"
table_name = "persons"
filepath = "../data/custom_mpo_06197001_model_data.h5"
h5_key = "persons"

[[tables]]
file_type = "h5"
table_name = "households"
filepath = "../data/custom_mpo_06197001_model_data.h5"
h5_key = "households"
```
