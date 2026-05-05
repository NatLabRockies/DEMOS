Income Module
=============

This module predicts household income using a regression model and optionally
applies two post-prediction adjustments:

1. **Inflation adjustment** – scales nominal income from the model's estimation
   (reference) year to the current simulation year using annual CPI factors
   supplied by the user.
2. **Calibration** – nudges the aggregate predicted income toward observed
   values using the same RMSE-based intercept-shift procedure used by the
   Mortality and Birth modules.

Key features:

- Predicts log income via the ``income_nworkers`` regression model and
  exponentiates to recover level predictions.
- Applies a cumulative inflation factor derived from a user-supplied table of
  annual adjustment rates when *inflation_reference_year* and
  *inflation_adjustment_table* are configured.
- Optionally calibrates aggregate income to per-year observed targets.

Caveats:

- Inflation adjustment requires all simulation years between
  *inflation_reference_year* and the forecast year to be present in the
  adjustment table; a ``KeyError`` is raised for missing years.
- When *inflation_reference_year* equals the current simulation year no
  adjustment is applied.
- If calibration is enabled it runs *before* inflation adjustment so that the
  calibration target values should also be expressed in reference-year nominal
  dollars (or consistently in the same price level as the model).

.. seealso::

   :doc:`income_adjustment_module` for the companion module that applies
   county-specific earnings growth rates to person-level wages each year.

Module configuration
--------------------

.. autoclass:: demos.config.IncomeModuleConfig
   :members:

Module function
---------------

.. autofunction:: demos.models.income.income
