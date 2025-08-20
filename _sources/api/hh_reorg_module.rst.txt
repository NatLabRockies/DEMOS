Household Reorganization Module
===============================

This module simulates household structure changes by executing three sub-models:
marriage, divorce, and cohabitation. Each sub-model is controlled by an estimated logit model,
and their outputs are used to update the persons and households tables in place.

Key features:

- Runs marriage, divorce, and cohabitation models in a coordinated step.
- Supports optional simultaneous calibration to fit model outputs to observed data.
- Updates household and marital status for all persons, including moving individuals between households as needed.
- Handles cohabitation transitions, marriage formation, and divorce events.

Notes
-----
- The outputs of the estimated models are as follows:
    * `marriage_list`: 0 = stay single, 1 = cohabitate, 2 = get married (opposite sex partner).
    * `cohabitate_x_list`: For cohabitating couples, indicates whether they break up (1), stay cohabitating (0), or get married (2).
    * `divorce_list`: For married couples, indicates whether they break up (1) or stay married (0).
- Simultaneous calibration is optional and only runs if calibration data is provided in the configuration.
- Most errors (e.g., inconsistent household structure) are handled silently; users should review logs for warnings.
- Orca columns defined in this module are essential for correct logic and should not be overridden by users.
- Caveats:
    * Households with multiple partners (relate == 1 or 13) are dropped silently.
    * Some logic (e.g., partner matching, household ID assignment) relies on random sampling and may not be fully reproducible unless a random seed is set.
    * The module assumes that all required columns are present in the persons and households tables.
    * If geoid_col is set in the config, county assignments are propagated to new households.
    * The logic for updating member_id after divorce is not fully documented and may need review.
    * The module does not enforce all possible invariants (e.g., every household has exactly one head) but attempts to fix common issues.

Module function
---------------
Module configuration options: :py:class:`~demos.config.HHReorgModuleConfig`

.. autofunction:: demos.models.household_reorg.household_reorg

Orca Functions
---------------
.. automodule:: demos.models.household_reorg
   :members:
   :undoc-members:
   :exclude-members: household_reorg