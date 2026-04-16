# Model Calibration

DEMOS includes a calibration system that adjusts the internal parameters of its estimated models (logit, multinomial logit, and regression) so that the simulation output matches observed real-world data. Without calibration, models will produce output strictly derived from their original estimation (fitting) data, which may not match conditions in the target geography or time period. Calibration shifts model predictions closer to observed aggregate statistics — such as the total number of births, deaths, or employed persons in a given year — while preserving the underlying behavioral relationships captured during model estimation.

This page explains:
1. [How Calibration Works](#how-calibration-works)
2. [Modules That Support Calibration](#modules-that-support-calibration)
3. [Preparing Observed Data Files](#preparing-observed-data-files)
4. [Configuring Calibration in the TOML File](#configuring-calibration-in-the-toml-file)
5. [Modules with Simultaneous Calibration: Employment and Household Reorganization](#modules-with-simultaneous-calibration-employment-and-household-reorganization)

---
(how-calibration-works)=
## How Calibration Works

DEMOS uses an iterative adjustment procedure based on the **Root Mean Square Error (RMSE)** between the model's predictions and a target value drawn from an observed data table. At each iteration, a single parameter (the model's *intercept*) is nudged so that the aggregate predicted count (or share) moves toward the observed value. This continues until either the error falls below a user-specified tolerance, or the maximum number of iterations is reached.

### The update rule

At each iteration, the intercept $\beta_0$ is updated as:

$$\beta_0 \leftarrow \beta_0 + \ln\!\left(\frac{\text{target}}{\sum \hat{y}}\right)$$

where $\sum \hat{y}$ is the sum of the current model predictions and $\text{target}$ is the observed value for the simulation year. This log-ratio update is a standard technique for intercept adjustment in discrete-choice and regression models: shifting the intercept by this amount is equivalent to re-scaling the aggregate predicted probability so it matches the aggregate observed rate.

(absolute-vs-relative-tolerance)=
### Absolute vs relative tolerance

The calibration procedure supports two ways of measuring how close "close enough" is:

- **Absolute** (`tolerance_type = "absolute"`): The error is computed as the RMSE between the *total predicted count* and the *observed count*. The tolerance value is therefore expressed in the same units as the counts (e.g., number of people). Use this when your observed data contains raw counts.
- **Relative** (`tolerance_type = "relative"`): The error is computed as the RMSE between the *predicted share* (predicted count divided by the eligible population size) and the *observed share*. The tolerance value is a fraction (e.g., `0.01` means 1%). Use this when your observed data contains rates or proportions.

---
(modules-that-support-calibration)=
## Modules That Support Calibration

The following modules support calibration via the standard `calibration_procedure` configuration block:

| Module | Config key | Calibrated model | What is calibrated |
|---|---|---|---|
| **Mortality** (`fatality`) | `mortality_module_config` | `mortality` | Total deaths per year |
| **Birth** (`birth`) | `birth_module_config` | `birth` | Total births per year |

The following modules have a **different calibration interface** (see [below](#modules-with-simultaneous-calibration-employment-and-household-reorganization)):

| Module | Config key | Note |
|---|---|---|
| **Employment** (`employment`) | `employment_module_config` | Simultaneous calibration across two sub-models |
| **Household Reorganization** (`household_reorg`) | `hh_reorg_module_config` | Simultaneous calibration across three sub-models |

The **Kids Moving** module (`kids_moving`) also performs hardcoded calibration internally; its parameters are controlled directly under `kids_moving_module_config` (see the [reference configuration](../api/configuration_module.rst)).

---
(preparing-observed-data-files)=
## Preparing Observed Data Files

Each module that uses the standard calibration procedure requires a CSV file containing observed aggregate statistics, one row per year. The file must have a column named **`year`** (used as the row index) and either a **`count`** or **`share`** column depending on your chosen `tolerance_type`.

### File format for absolute calibration (`tolerance_type = "absolute"`)

The file must contain a column named `count` with the observed total number of events (deaths, births, etc.) for each year.

```
year,count
2010,1452
2011,1489
2012,1503
...
```

### File format for relative calibration (`tolerance_type = "relative"`)

The file must contain a column named `share` with the observed *rate* (a number between 0 and 1) for each year.

```
year,share
2010,0.0092
2011,0.0094
2012,0.0096
...
```

> **Important:** The index column must be named `year`. Make sure this column contains integer years matching the years in your simulation run (between `base_year` and `forecast_year`). If a year is missing from the file, the simulation will raise an error when it tries to look up the target value.

The file can be stored anywhere accessible from the configuration file. A common convention is to place these files in a subdirectory of your data folder, e.g.:

```
data/
  my_region/
    observed_calibration_values/
      mortalities_over_time.csv
      births_over_time.csv
```

---
(configuring-calibration-in-the-toml-file)=
## Configuring Calibration in the TOML File

Calibration is configured inside the relevant module's configuration block in your `demos_config.toml` file. Each calibratable module has a `calibration_procedure` sub-block.

### Full example: mortality module with absolute calibration

```toml
[mortality_module_config.calibration_procedure]
procedure_type = "rmse_error"
tolerance_type = "absolute"
tolerance = 30
max_iter = 500

[mortality_module_config.calibration_procedure.observed_values_table]
file_type = "csv"
table_name = "observed_fatalities_data"
filepath = "../data/my_region/observed_calibration_values/mortalities_over_time.csv"
index_col = "year"
```

### Full example: birth module with absolute calibration

```toml
[birth_module_config.calibration_procedure]
procedure_type = "rmse_error"
tolerance_type = "absolute"
tolerance = 60
max_iter = 1000

[birth_module_config.calibration_procedure.observed_values_table]
file_type = "csv"
table_name = "observed_births_data"
filepath = "../data/my_region/observed_calibration_values/births_over_time.csv"
index_col = "year"
```

### Parameter reference

| Parameter | Type | Required | Description |
|---|---|---|---|
| `procedure_type` | string | Yes | Must be `"rmse_error"`. This is the only available calibration procedure. |
| `tolerance_type` | string | No | `"absolute"` (default) or `"relative"`. Controls how the error is measured. See [above](#absolute-vs-relative-tolerance). |
| `tolerance` | float | Yes | The error threshold below which calibration stops. Units depend on `tolerance_type`. |
| `max_iter` | int | No | Maximum number of calibration iterations (default: 20). Calibration stops here even if the tolerance has not been reached. |
| `observed_values_table` | table block | Yes | Defines the CSV file with observed values. Follows the same format as any `[[tables]]` entry. |

### The `observed_values_table` sub-block

The `observed_values_table` block defines the file containing observed data. It uses the same fields as a regular data source entry:

| Field | Description |
|---|---|
| `file_type` | Must be `"csv"` for observed calibration data. |
| `table_name` | An internal name for this table in DEMOS memory. Must be unique across all tables. Pick a descriptive name like `"observed_fatalities_data"`. |
| `filepath` | Path to the CSV file, relative to your configuration file. |
| `index_col` | Must be `"year"`. |

### Disabling calibration

To disable calibration for a module, simply remove or comment out the entire `calibration_procedure` block for that module:

```toml
# [mortality_module_config.calibration_procedure]
# procedure_type = "rmse_error"
# ...
```

If the block is absent, DEMOS will run the model using the original estimated parameters with no adjustment.

---
(modules-with-simultaneous-calibration-employment-and-household-reorganization)=
## Modules with Simultaneous Calibration: Employment and Household Reorganization

Two modules — **Employment** and **Household Reorganization** — use a more complex calibration strategy called *simultaneous calibration*. Instead of calibrating a single model against a single target, these modules adjust multiple models at the same time so that their *joint output* matches aggregate observed counts.

This approach is necessary because the outcome of interest (e.g., total number of employed persons, total number of married persons) depends on the combined output of several sub-models that each influence the same population. Calibrating them independently would cause the adjustments to conflict with each other.

### The simultaneous calibration algorithm

Simultaneous calibration uses a **momentum-based gradient descent** approach. At each iteration:

1. All sub-models are run on their respective data to produce predictions.
2. The combined predicted aggregate is compared to the observed aggregate.
3. An update step $g$ is computed using a momentum term to smooth out oscillations:

$$g_t = \alpha_t \left( \mu \cdot g_{t-1} + (1-\mu) \cdot \delta_t \right)$$

where:
- $\alpha_t$ is a *decaying learning rate* that starts at `learning_rate` and decreases toward zero as the iteration count approaches `max_iter`
- $\mu$ is the `momentum_weight` (a value between 0 and 1 that controls how much the previous gradient update is retained)
- $\delta_t = \ln(\text{target} / \hat{y}_t)$ is the log-ratio between the target and current prediction

4. The intercepts of each sub-model are adjusted by $g_t$.
5. Steps 1–4 repeat until the absolute error falls below `tolerance` or `max_iter` is reached.

The decaying learning rate helps the algorithm converge: large adjustments happen early, and corrections become finer as the prediction approaches the target.

---

### Employment module

The Employment module calibrates two sub-models simultaneously:

- **`enter_labor_force`**: A logit model predicting which unemployed persons (age ≥ 18) transition into employment.
- **`exit_labor_force`**: A logit model predicting which employed persons (age ≥ 18) exit the workforce.

The calibration target is the **total number of workers** (employed persons) predicted after running both models, compared against an observed count loaded from the `observed_employment` table.

> **Important:** The table name `observed_employment` is hard-coded. You must load this table via the `[[tables]]` section of your configuration with exactly that name.

#### Required observed data table

The `observed_employment` table must be a CSV with a `year` index and a `count` column containing the total observed employed persons per year:

```
year,count
2010,18423
2011,18751
...
```

Add it to your configuration under `[[tables]]`:

```toml
[[tables]]
file_type = "csv"
table_name = "observed_employment"
filepath = "../data/my_region/observed_calibration_values/employment_obs.csv"
index_col = "year"
```

#### Calibration configuration block

```toml
[employment_module_config.simultaneous_calibration_config]
tolerance = 100
max_iter = 20
learning_rate = 2.0
momentum_weight = 0.3
```

#### Parameter reference

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `tolerance` | float | Yes | — | Absolute error threshold (in number of workers) below which calibration stops. |
| `max_iter` | int | No | 20 | Maximum number of calibration iterations. |
| `learning_rate` | float | No | 2.5 | Controls the step size at the first iteration. Decays linearly toward zero. |
| `momentum_weight` | float | No | 0.3 | Weight applied to the previous gradient update. Values closer to 1 introduce more smoothing. |

#### Disabling employment calibration

To disable simultaneous calibration for employment, comment out or remove the block:

```toml
# [employment_module_config.simultaneous_calibration_config]
# ...
```

When the block is absent, DEMOS checks whether individual model calibration procedures are defined under `employment_module_config.enter_model_calibration_procedure` and `employment_module_config.exit_model_calibration_procedure`. If neither is present, no calibration is performed. Note that simultaneous calibration and per-model calibration are mutually exclusive: defining both will raise a validation error.

---

### Household Reorganization module

The Household Reorganization module calibrates three sub-models simultaneously:

- **`marriage`**: A multinomial logit model predicting transitions for unmarried, non-cohabitating persons — options are: stay single, start cohabitating, or get married.
- **`cohabitation`**: A multinomial logit model predicting transitions for cohabitating persons — options are: stay cohabitating, break up, or get married.
- **`divorce`**: A binary logit model predicting which married households undergo a divorce.

The combined output of these three models determines the predicted number of married and divorced persons after the step. Calibration targets both of these counts simultaneously, weighting each proportionally to its magnitude in the observed data.

> **Important:** The table name `observed_marrital_data` is hard-coded. You must load this table via the `[[tables]]` section with exactly that name.

#### Required observed data table

The `observed_marrital_data` table must be a CSV with a `year` index and columns `MAR` and `count`, containing observed counts for each marital status category per year. The values used during calibration are `MAR == 1` (married) and `MAR == 3` (divorced/separated):

```
year,MAR,count
2010,1,14200
2010,3,2300
2011,1,14350
2011,3,2280
...
```

Add it to your configuration under `[[tables]]`:

```toml
[[tables]]
file_type = "csv"
table_name = "observed_marrital_data"
filepath = "../data/my_region/observed_calibration_values/marrital_status_over_time_obs.csv"
index_col = "year"
```

#### Calibration configuration block

```toml
[hh_reorg_module_config.simultaneous_calibration_config]
tolerance = 5000
max_iter = 100
learning_rate = 1.5
momentum_weight = 0.3
```

#### Parameter reference

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `tolerance` | float | Yes | — | Combined RMSE threshold (in number of persons) below which calibration stops. The error combines the residuals for the married and divorced counts. |
| `max_iter` | int | No | 20 | Maximum number of calibration iterations. |
| `learning_rate` | float | No | 2.5 | Controls the initial step size. Decays linearly toward zero. |
| `momentum_weight` | float | No | 0.3 | Weight applied to the previous gradient update. |

#### How intercept updates are applied

Each of the three sub-models is adjusted through a different parameter:

- **`divorce`**: The model's primary intercept (`fitted_parameters[0]`) is shifted by the divorce gradient.
- **`marriage`**: The `married` outcome coefficient in the model's coefficient table is shifted by the marriage gradient.
- **`cohabitation`**: The `marriage` outcome coefficient in the model's coefficient table is shifted by the cohabitation gradient (cohabitation → marriage transitions are steered to match the marriage target).

The marriage and cohabitation gradients are weighted by the *proportion of the married count* relative to the total observed marital transitions. The divorce gradient is weighted by the *proportion of the divorced count*. This ensures that the calibration effort is distributed in proportion to the size of each phenomenon.

#### Disabling household reorganization calibration

To disable calibration for household reorganization, comment out or remove the block:

```toml
# [hh_reorg_module_config.simultaneous_calibration_config]
# ...
```

When absent, models run with their original estimated parameters.

---

## Choosing Good Tolerance Values

Choosing a tolerance that is too tight (very small) can cause calibration to run for the full `max_iter` without converging, adding unnecessary compute time. Choosing one that is too loose means the model output may deviate substantially from observed data.

A practical starting point:

- For **absolute** tolerances, set the tolerance to approximately 1–5% of the observed count for the most common year. For example, if you observe ~1,500 deaths per year, a tolerance of `30` (≈2%) is reasonable.
- For **relative** tolerances, values between `0.005` and `0.02` (0.5%–2%) work well for most demographic rates.
- For **simultaneous calibration**, the tolerance represents the absolute difference in person counts. Values of 100–5,000 are typical depending on the size of the synthetic population.

If calibration is not converging even after increasing `max_iter`, consider increasing the `learning_rate` slightly. If the algorithm oscillates (error fluctuates without declining), try increasing `momentum_weight` toward 0.5–0.7 or decreasing `learning_rate`.
