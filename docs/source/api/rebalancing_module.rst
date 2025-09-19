Household Rebalancing Module
============================

This module adjusts the synthetic population to match observed household size distributions by geography.
It duplicates or removes households to align current counts with control totals, maintaining population
consistency while preserving demographic characteristics. The module tracks marital status before and
after rebalancing operations and stores removed households/persons in separate tables.

Key features:

- Matches household counts to control totals by geography and household size.
- Duplicates households (and their members) when counts are below target.
- Removes households (and their members) when counts exceed target.
- Preserves demographic characteristics through exact duplication/removal.
- Tracks marital status changes for validation purposes.
- Stores removed records in rebalanced_households and rebalanced_persons tables.

Caveats:

- Requires a control table with year, geography, household size, and target count columns.
- The control table index must be 'year' and must have exactly 3 columns.
- Sampling with replacement is used when duplicating more households than available.
- Most errors are handled with assertions; users should ensure data consistency.
- Geographic and household size columns must match between households and control tables.

Module function
---------------

Module configuration options: :py:class:`~demos.config.HHRebalancingModuleConfig`

.. autofunction:: demos.models.rebalancing.household_rebalancing