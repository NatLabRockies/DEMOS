# Get Started

## Overview
The Demographic Microsimulator (DEMOS) is an agent-based simulation framework used to model the evolution of population demographic characteristics and lifecycle events, such as education attainment, marital status, and other key transitions. DEMOS modules are designed to capture the interdependencies between short-term and long-term lifecycle events, which are often influential in downstream transportation and land-use modeling.

A key feature of DEMOS is its ability to track changes in an agent’s demographic status from year *t* to year *t + 1*. This structure allows the model to evolve populations over any user-defined time horizon. As a result, DEMOS is well suited for analyzing medium- and long-term transportation-related decisions, including household vehicle transactions (e.g., purchasing, selling, or replacing vehicles) and work location choices.
Core features of DEMOS include the modeling of more than ten lifecycle events, behaviorally realistic patterns informed by long-running panel data, explicit representation of interdependencies among lifecycle processes, and a flexible, modular simulation architecture.

A technical memorandum describing DEMOS is available [here](https://github.com/NatLabRockies/DEMOS/blob/main/DEMOS_Technical_Memo.pdf). The memorandum provides an overview of the framework’s functionality, model structure, input and output data, and its applications in transportation planning and broader policy analysis contexts. Interested readers are also encouraged to consult the paper listed below for additional details on the DEMOS methodology.

---

DEMOS operates on tabular data representing agents or entities (primarily persons and households), and is configured via a simple TOML file. DEMOS can be run from source or using Docker for reproducibility.

This document summarizes instructions to install, configure and run DEMOS. Sections 1-4 will help you correctly organize the data and configuration file, so we recommend reading them once before attempting to run DEMOS.

> DEMOS requires at least 16gb of available RAM to execute. This greatly depends on the size and resolution of the input data, but from our internal testing, we recommend at least 16gb of RAM

```{contents}
:local:
:depth: 2
:backlinks: none
:class: this-will-duplicate-information-and-it-is-still-useful-here
```

## 1. Installation

### Docker Compose (recommended)
The latest docker image for demos is stored in `ghcr.io/NatLabRockies/demos:latest`. The input data and configuration file are fed to the container through volumes ([more info about Docker volumes](https://docs.docker.com/engine/storage/volumes/)). We provide a `docker-compose` workflow that can be used to make the process of mounting volumes easier. Make sure you have [Docker](https://docs.docker.com/desktop/) and [Docker Compose](https://docs.docker.com/compose/install/) installed before you proceed.

The following instructions will guide you through running DEMOS with example data representative of a small area in California. The data and configuration files required to run this example are located in `./data/` and `./configuration` folders. You can change where DEMOS will look for your data following the instructions [in the Docs](https://NatLabRockies.github.io/DEMOS/).

#### Clone this repository
By cloning this repository you download the configuration and data for an example run of DEMOS. You can also use the `Download ZIP` option available through the green `Code` button in the [GitHub repo](https://github.com/NatLabRockies/DEMOS) and decompress it to achieve the same results as the clone command.

Run the following command in the Terminal App (MacOS) or Command Prompt/PowerShell (Windows):
```bash
git clone https://github.com/NatLabRockies/DEMOS.git

# Move into the project folder
cd DEMOS

# This folder contains (among other files) a data and configuration folder
# as well as a docker-compose.yml file

# This command runs DEMOS on a docker container
docker compose up
```


> **Note:**  
> Make sure the Docker Daemon is running. This changes from system to system but Docker Desktop should have a status flag indicating if the daemon is live, if Desktop is available

<!-- **Important Note:**
> While the pipeline to build a docker image is implemented, there is no public docker image available, please execute [from source](#From-Source) -->



By default, this assumes that your config file is located in `./configuration/demos_config.toml` and the data folder is `./data`, with `./` being the root of the project (See the [file stucture section](#file-tree-structure-for-data-and-configuration) for details on how to organize the input data).
If you need to specify a different location for them, you can run:

```bash
DEMOS_CONFIG_PATH=<path-to-config> DEMOS_DATA_DIR=<path-to-data-dir> docker compose up
```

> **Note for MacOS/Windows:**  
> Docker imposes a global limit on how much RAM containers can allocate. DEMOS easily surpases those limits, so in order to run DEMOS in Docker, users need to access the Docker Desktop GUI and `Preferences → Resources → Memory → Increase it (at least 16-20gb)`. The amount of memory required to run DEMOS will primarily depend on the size of the input data.

---

### From Source (if you are not using Docker)

If you prefer to create your own Python environment and run the Python code directly (for debugging or enhancing DEMOS capabilities), you can do so following these instructions:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/NREL/DEMOS_NREL.git
    cd DEMOS_NREL
    ```

2. **Create and activate a Python 3.10 environment**:
    ```bash
    conda create -n demos-env python=3.10
    conda activate demos-env
    pip install .
    ```

3. **Run DEMOS**:
    ```bash
    cd demos
    python simulate.py -cfg ../configuration/demos_config.toml
    ```


## 2. Preparing Your Configuration

DEMOS is configured via a TOML file (see [example configuration](default_configuration) for a full example and [configuration API](../api/configuration_module.rst) for descriptions on each of the accepted parameters).  
At minimum, you must define the `persons` and `households` tables:

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

Other tables and module configurations can be added as needed. The default configuration exposes all options with default values.

(file-tree-structure-for-data-and-configuration)=
## 3. File Tree Structure for Data and Configuration

To run DEMOS, organize your files as follows:

```
DEMOS_NREL/
├── configuration/
│   └── demos_config.toml                   # Main configuration file (TOML)
├── data/
|   ├── custom_mpo_06197001_model_data.h5   # Example HDF5 data file
│   ├── relmap_06197001.csv                 # Example CSV data file
│   ├── income_rates_06197001.csv           # Example CSV data file
│   ├── hsize_ct_06197001.csv               # Example CSV data file 
|   └── calibrated_configs/                 # Here is were the parameters of the estimated models go
|       └── ...
├── demos/ # Source code (if not using docker)
|   ├── simulate.py                         # Main entry point for running DEMOS
|   ├── models                              # Logic of individual modules
|       └── ... 
|   └── ... 
└── ...
```

- Place your **TOML configuration file** in the `configuration/` directory.
- Place all **input data files** (CSV, HDF5, etc.) in the `data/` directory.
- Make sure the paths in your `demos_config.toml` match the location of your data files.
- If you are using the provided Docker Compose workflow, the data volume will always be accesible as `./data`, therefore, you should use `./data` in your configuration file as the name of the folder where the data for the tables is found, even if the folder is named differently in your computer.

> **Tip:**  
> You can use different data files or directories, but make sure the paths in your configuration file are correct relative to the project root (`DEMOS/`).

The output of DEMOS will be stored in `data/output/demos_output_{year}.h5`.


## 4. Understanding the Workflow

- **Tables**: Each row in the `persons` and `households` tables represents an agent or entity.
- **Modules**: Simulation logic is organized into modules, each operating on the tables for each simulated year.
- **Orca**: DEMOS uses the [orca](https://github.com/UDST/orca) library to manage tables and define lazily-computed columns (see documentation for details).
- **Configuration**: All simulation options, data sources, and module settings are controlled via the TOML config file.


## 5. Troubleshooting Common Errors

- **Missing Required Tables**:  
  If either `persons` or `households` is missing from your config, DEMOS will raise an error:
  ```
  ValueError: Both 'persons' and 'households' tables are required. Tables defined: [...]
  ```

- **Inconsistent Persons Table**:  
  DEMOS checks for data consistency (e.g., every household must have exactly one head).  
  Behavior is controlled by `inconsistent_persons_table_behavior` in the config:
    - `"error"`: Stop on inconsistency (default)
    - `"fix"`: Attempt to fix common issues (e.g., assign head to single-person households)
    - `"ignore"`: Proceed without checks (not recommended)

- **Households with Multiple Partners**:  
  If a household has more than two partners, it will be dropped or raise an error, depending on your config.

- **Memory Errors (Docker)**:  
  DEMOS requires a lot of memory. If you see out-of-memory errors, increase Docker's memory allocation.

## 6. Next Steps

- Explore the [example configuration](default_configuration) for more options.
- Review the [configuration fields and structure](configuration) to understand all available settings.
- See the documentation for details on [modules](../api/modules.rst), orca columns, and model calibration.


## 7. Need Help?

- [DEMOS GitHub Repository](https://github.com/NREL/DEMOS_NREL)
- [Orca Documentation](https://github.com/UDST/orca)


*For more details on modules, orca columns, and advanced configuration, see the full documentation.*