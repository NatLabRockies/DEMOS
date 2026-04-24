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