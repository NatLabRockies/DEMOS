# Advanced Options for DEMOS

## Configuration of calibration procedures

Certain modules support calibration of the simulation output to observed values. Specific calibration parameters can be set for each module that supports it. If no configuration is provided, calibration is not performed.

Calibration configuration is defined at `module-level-config.calibration_procedure` (`module-level-config` is defined differently for every module. The options are displayed [here](../api/configuration_module.rst) and in each module's documentation).

For example, in the mortality module configuration we find the following:

```toml
[mortality_module_config.calibration_procedure]
procedure_type = "rmse_error"
tolerance_type = "absolute"
tolerance = 1000
max_iter = 1000
[mortality_module_config.calibration_procedure.observed_values_table]
file_type = "csv"
table_name = "observed_fatalities_data"
filepath = "../data/sf_bay_example/observed_calibration_values/mortalities_over_time_obs.csv"
index_col = "year"
```

This section of the configuration file does a couple of things:
- Sets the procedure type to `rmse_error` (currently the only available option)
- Sets the Tolerance type to `absolute`. This means that DEMOS will continue to optimize the output until the absolute difference between the current value predicted and the observed is smaller than this tolerance. An alternative value for this parameter is `relative`, which changes the logic to interpret the tolerance value as relative.
- Sets the tolerance level. If `tolerance_type` is `absolute`, this is an absolute value. Otherwise, this is a percentage
- Sets the maximum number of iterations
- Identifies which data to use for validation by assigning a value to `mortality_module_config.calibration_procedure.observed_values_table`. This has the same format as any other table loade d in the `tables` section of the configuration.

If for example you would like to skip calibration on the mortality module, just delete or comment out all these lines corresponding to `mortality_module_config.calibration_procedure`.

---

Additionally, some modules (namely `employment` and `household_reorganization`) implement simultaneous calibration.

**If you want to use simultaneous calibration for the `employment` module**

```toml
[employment_module_config.simultaneous_calibration_config]
tolerance = 100
max_iter = 2
learning_rate = 2
momentum_weight = 0.3
```

Due to the complexity and nuances of simultaneous calibration, the required tables of observed values (`observed_entering_workforce` and `observed_exiting_workforce`) are hard-coded, and an error will be raised if they are not loaded.

**If you want to skip calibration, just delete or comment out these entries from the configuration file like this:**

```toml
# [employment_module_config.simultaneous_calibration_config]
# tolerance = 100
# max_iter = 2
# learning_rate = 2
# momentum_weight = 0.3
```

<!-- **If you instead want to use simple calibration**

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


<!-- See the example config for more options, including output tables, calibration, and module selection. -->

## Selection of modules to run

The `modules` parameter in the configuration file accepts a list of strings identifying the modules. By default all are included, but if you'd like to only run a selection of them you can change it. For instance to run only `aging` and `education`:
```
modules = [
    "aging",
    "education",
]
```

---

## Lazily computed columns

DEMOS uses [Orca](https://udst.github.io/orca/) as its internal data pipeline framework. One of
Orca's most important features for working with DEMOS data is the concept of **lazily computed
columns**, and understanding them will save you a lot of confusion when inspecting or extending
the simulation.

### What is a lazily computed column?

In a normal table (a pandas DataFrame), every column holds actual stored values. A lazily computed
column is different: instead of a pre-stored array of values, it is a **Python function**
registered with Orca that *returns* a pandas Series on demand. Orca only calls that function when
some part of the simulation actually asks for the column — hence "lazy" (computed only when needed,
not upfront).

This is useful because:

- Columns that depend on other columns (for example, a column that categorises people by age into
  bins) are always guaranteed to be up to date, since they are recalculated from the current data
  every time they are requested.
- It avoids storing redundant derived data, which keeps memory usage lower.
- Dependencies between columns are explicit and traceable in the code.

In the DEMOS codebase these columns are registered using the `@orca.column` decorator. For example,
the `child` indicator column (persons table) is defined in `aging.py` as:

```python
@orca.column(table_name="persons")
def child(data="persons.relate"):
    return data.isin([2, 3, 4, 14]).astype(int)
```

Every time any module requests `persons["child"]`, Orca calls this function and passes in
the current `relate` column automatically (Orca matches function argument names to registered
data).

Other examples of lazily computed columns in DEMOS:

| Column | Table | Defined in | What it computes |
|---|---|---|---|
| `child` | `persons` | `aging.py` | 1 if `relate` ∈ {2, 3, 4, 14}, else 0 |
| `senior` | `persons` | `aging.py` | 1 if `age` ≥ configured senior age (default 65) |
| `age_group` | `persons` | `aging.py` | Categorical age bin (e.g. `"20-29"`, `"30-39"`) |
| `is_head` | `persons` | `household_reorg.py` | 1 if `relate == 0` |
| `is_not_married` | `persons` | `household_reorg.py` | True if `age ≥ 15` and `MAR != 1` |
| `cohabitate` | `persons` | `household_reorg.py` | True if person is a cohabiting partner or head of a cohabiting household |
| `hh_size` | `households` | `household_reorg.py` | Categorical household size (e.g. `"one"`, `"two"`, `"four or more"`) |
| `hh_workers` | `households` | `employment.py` | Categorical worker count (`"none"`, `"one"`, `"two or more"`) |
| `income` | `households` | `employment.py` | Total household income aggregated from person-level earnings |

### Caching

Some lazily computed columns are marked with `cache=True`. When caching is enabled, the function
is called once and the result is stored in memory. Subsequent requests for that column return
the stored value without recomputing it, which improves performance for columns that are expensive
to compute.

You will also see a `cache_scope` parameter alongside `cache=True`. This controls *when* the
cached value is discarded and recomputed:

- `cache_scope="step"` — The cached value is discarded after each simulation step finishes.
  The column will be recomputed fresh the next time a step requests it. This is appropriate for
  columns that should not change *within* a step but may change between steps.
- `cache_scope="forever"` — The value is computed once and never cleared for the duration
  of the simulation run. Use this only for columns that genuinely cannot change (for example,
  a fixed income distribution table).

For example, `is_not_married` and `cohabitate` in `household_reorg.py` both use
`cache_scope="step"` because they are read multiple times within a single household
reorganisation step but must reflect an updated persons table at the start of the next step.

### Input columns are overwritten by computed columns

> **Important:** If a column with the same name exists both in the input data (i.e., it is a
> column in your `persons.h5` or `households.h5` file) **and** as a registered Orca computed
> column, **the computed column takes precedence and the input value is silently ignored**.

This is by design in Orca: registered column functions are always considered authoritative. In
practice this means:

- If your input synthetic population includes a column called `hh_size` in the `households`
  table, it will be **replaced** by the value computed dynamically from the persons table.
- Similarly, `child`, `senior`, `age_group`, `is_head`, and every other column in the table
  above will override any same-named column in your input files.

This is usually the correct behaviour (computed values are up to date), but it is worth being
aware of when preparing input data or when debugging unexpected values.

---

## GEOID columns and geographic assignment

Several modules in DEMOS create or reorganise households during the simulation (for example,
when two people get married they may form a new household, or when a young adult moves out they
start a new household). Whenever a new household is created, DEMOS needs to assign it a
geographic unit — otherwise the new household would have no location and downstream
processing steps that depend on geography would fail or produce incorrect results.

DEMOS handles this through a configurable **GEOID column** setting in each relevant module.
The GEOID column is the name of the column in the `households` table that stores the geographic
identifier for each household (for example, a TAZ code, a Census tract GEOID, or a county ID).
When a new household is created, DEMOS copies the GEOID value from the original household
(the one the person is splitting from or merging with) into the new household row.

### Modules that use `geoid_col`

**Household Reorganization** (`household_reorg_module_config.geoid_col`)

The household reorganisation step creates new households for three events:

- two people who were not heads of household form a new household together (new marriage or
  cohabitation where neither person was a household head);
- a cohabitating partner leaves to form their own household after a break-up;
- divorce (one of the partners moves out and starts a new household).

In all three cases the new household inherits the GEOID from the departing person's original
household. If `geoid_col` is not set (left as `null`), no geographic assignment is performed
for newly created households.

```toml
[hh_reorg_module_config]
geoid_col = "taz_id"   # name of the column in the households table that holds the geographic ID
```

**Kids Moving** (`kids_moving_module_config.geoid_col`)

When a young adult moves out of the parental household, a new single-person household is created.
The `geoid_col` setting tells DEMOS which column to copy from the original household to the
new one. The `lcm_county_id` column is always copied in addition to `geoid_col`.

```toml
[kids_moving_module_config]
geoid_col = "taz_id"
calibration_target_share = 0.12
calibration_tolerance = 0.01
max_iter = 100
```

**Household Rebalancing** (`hh_rebalancing_module_config.geoid_col`)

The rebalancing step duplicates or removes households to match external control totals. The
`geoid_col` parameter here indicates which column in the **households** table denotes the
geographic unit over which the control totals are stratified. The control totals table (set
via `control_table`) must also contain a column with this exact name.

```toml
[hh_rebalancing_module_config]
control_table = "hsize_ct"
control_col = "hh_size"
geoid_col = "lcm_county_id"
```

### Making sure your GEOID column is consistent

Because GEOID values are inherited (copied from original to new household), consistency of the
GEOID column in your input data is critical:

1. The column name you set in `geoid_col` must exist in the `households` table of your input
   file. If it does not, DEMOS will raise a `KeyError` when the first household-creation event
   occurs.
2. The same column name must be used consistently across all module configurations. Mixing
   `"taz_id"` in one module and `"TAZ"` in another will cause some new households to be missing
   their geographic assignment.
3. For the rebalancing module specifically, the control totals CSV must contain a column with
   the exact same name and values as the GEOID column in the households table. A mismatch will
   cause the rebalancing step to fail with an assertion error.