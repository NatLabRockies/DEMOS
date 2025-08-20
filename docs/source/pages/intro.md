# Get Started

DEMOS is a modular demographic microsimulator. It operates on tabular data representing agents or entities (primarily persons and households), and is configured via a simple TOML file. DEMOS can be run from source or using Docker for reproducibility.


## 1. Installation

### Using Docker (Recommended)

1. **Clone the repository**:
    ```bash
    git clone https://github.com/NREL/DEMOS_NREL.git
    cd DEMOS_NREL
    ```

2. **Run with Docker Compose**:
    ```bash
    DEMOS_CONFIG_PATH=<path-to-config> DEMOS_DATA_DIR=<path-to-data-dir> docker-compose up
    ```
    By default, `DEMOS_CONFIG_PATH=./demos_config.toml` and `DEMOS_DATA_DIR=./data`.

3. **Or run with Docker directly**:
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


## 2. Preparing Your Configuration

DEMOS is configured via a TOML file (see `configuration/demos_config.toml` for a full example).  
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

Other tables and module configurations can be added as needed.  
See the example config for more options, including output tables, calibration, and module selection.


## 3. Running DEMOS

From the project root, run:

```bash
python simulate.py -cfg configuration/demos_config.toml
```

Or use Docker as described above.


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