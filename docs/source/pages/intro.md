# Get Started

DEMOS is a modular demographic microsimulator. It operates on tabular data representing agents or entities (primarily persons and households), and is configured via a simple TOML file. DEMOS can be run from source or using Docker for reproducibility.

This document summarizes instructions to install, configure and run DEMOS. Sections 1-4 will help you correctly organize the data and configuration file, so we recommend reading them once before attempting to run DEMOS.

```{contents}
:local:
:depth: 2
:backlinks: none
:class: this-will-duplicate-information-and-it-is-still-useful-here
```

## 1. Installation

### Using Docker (Recommended)

<!-- **Important Note:**
> While the pipeline to build a docker image is implemented, there is no public docker image available, please execute [from source](#From-Source) -->

1. **Clone the repository**:
    ```bash
    git clone https://github.com/NREL/DEMOS_NREL.git
    cd DEMOS_NREL
    ```
  
    **Build Docker Image** *(Development only)*
    ```bash
    docker build -t demos:0.0.1 --platform=linux/amd64 -f Dockerfile .
    ```

1. **Run with Docker Compose**:
    ```bash
    docker compose up
    ```

    By default, this assumes that your config file is located in `./configuration/demos_config.toml` and the data folder is `./data`, with `./` being the root of the project (See the [file stucture section](#file-tree-structure-for-data-and-configuration) for details on how to organize the input data).
    If you need to specify a different location for them, you can run:

    ```bash
    DEMOS_CONFIG_PATH=<path-to-config> DEMOS_DATA_DIR=<path-to-data-dir> docker compose up
    ```

1. **Or run with Docker directly**:
    ```bash
    docker run --volume <path-to-config>:/demos/config.toml:ro --volume <path-to-data-dir>:/demos/data --platform=linux/amd64 demos
    ```

> **Note for MacOS/Windows:**  
> Increase Docker's memory allocation to at least 16–20 GB via Docker Desktop:  
> `Preferences → Resources → Memory`.

---

### From Source

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

### Compiling documentation (Optional but recommended)
From the root of the project:
```bash
cd docs
make html
open build/html/index.html
```


## 2. Preparing Your Configuration

DEMOS is configured via a TOML file (see [example configuration](default_configuration) for a full example).  
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

---

Examples of important configuration options are:

### Selection of modules to run:

The `modules` parameter accepts a list of strings identifying the modules. By default all are included, but if you'd like to only run a selection of them you can change it. For instance to run only `aging` and `education`:
```
modules = [
    "aging",
    "education_model",
]
```

### Configuration of calibration procedures:

Certain modules support calibration of the simulation to observed values. Specific calibration parameters can be set for each module that supports it. If no configuration is provided, calibration is not performed. Additionally, some modules (namely `employment` and `household_reorganization`) implement simultaneous calibration. While the `employment` module implements both types of calibration, only one of the two can be used. An error will be raised if two types of calibration are defined.

Calibration configuration is defined at `module-level-config.calibration_procedure` (`module-level-config` is defined differently for every module. The options are displayed [here](../api/configuration_module.rst) and in each module's documentation). We will use the `employment` module as an example.

**If you want to use simultaneous calibration for the `employment` module**

```toml
[employment_module_config.simultaneous_calibration_config]
tolerance = 100
max_iter = 2
learning_rate = 2
momentum_weight = 0.3
```

Due to the complexity and nuances of simultaneous calibration, the required tables of observed values (`observed_entering_workforce` and `observed_exiting_workforce`) are hard-coded, and an error will be raised if they are not present.

**If you instead want to use simple calibration**

We need to define the following:
```toml
[employment_module_config.enter_model_calibration_procedure]
procedure_type = "rmse_error"
observed_values_table = "observed_entering_workforce"
tolerance_type = "relative"
tolerance = 0.01
max_iter = 1000

[employment_module_config.exit_model_calibration_procedure]
procedure_type = "rmse_error"
observed_values_table = "observed_exiting_workforce"
tolerance_type = "relative"
tolerance = 0.01
max_iter = 1000
```

This will allow DEMOS to execute calibration on each of the two modules in the employment module.

**If you want to skip calibration, just delete these entrances from the configuration file.**

<!-- See the example config for more options, including output tables, calibration, and module selection. -->

(file-tree-structure-for-data-and-configuration)=
## 3. File Tree Structure for Data and Configuration

To run DEMOS, organize your files as follows:

```
DEMOS_NREL/
├── configuration/
│   └── demos_config.toml # Main configuration file (TOML)
├── data/
|   ├── custom_mpo_06197001_model_data.h5 # Example HDF5 data file
│   ├── relmap_06197001.csv # Example CSV data file
│   ├── income_rates_06197001.csv # Example CSV data file
│   ├── hsize_ct_06197001.csv # Example CSV data file 
|   └── calibrated_configs/ # Here is were the parameters of the estimated models go
|       └── ...
├── demos/ # Source code 
|   └── simulate.py # Main entry point for running DEMOS 
├── docs/ # Documentation
└── ...
```

- Place your **TOML configuration file** in the `configuration/` directory.
- Place all **input data files** (CSV, HDF5, etc.) in the `data/` directory.
- Make sure the paths in your `demos_config.toml` match the location of your data files.

> **Tip:**  
> You can use different data files or directories, but make sure the paths in your configuration file are correct relative to the project root (`DEMOS_NREL/`).

## 4. Running DEMOS

From the project root, run:

```bash
cd demos
python simulate.py -cfg configuration/demos_config.toml
```

Or use Docker as described above.

The output of DEMOS will be stored in `data/output/demos_output_{year}.h5`.


## 5. Understanding the Workflow

- **Tables**: Each row in the `persons` and `households` tables represents an agent or entity.
- **Modules**: Simulation logic is organized into modules, each operating on the tables for each simulated year.
- **Orca**: DEMOS uses the [orca](https://github.com/UDST/orca) library to manage tables and define lazily-computed columns (see documentation for details).
- **Configuration**: All simulation options, data sources, and module settings are controlled via the TOML config file.


## 6. Troubleshooting Common Errors

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

## 7. Next Steps

- Explore the [example configuration](default_configuration) for more options.
- Review the [configuration fields and structure](configuration) to understand all available settings.
- See the documentation for details on [modules](../api/modules.rst), orca columns, and model calibration.


## 8. Need Help?

- [DEMOS GitHub Repository](https://github.com/NREL/DEMOS_NREL)
- [Orca Documentation](https://github.com/UDST/orca)


*For more details on modules, orca columns, and advanced configuration, see the full documentation.*