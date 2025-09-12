Income Adjustment Module
============================

This module updates person-level earnings based on county-specific income growth rates.
It applies annual adjustment factors to maintain realistic income trajectories over time,
accounting for geographic variation in economic conditions. The module reads rate data
from an external table and applies multiplicative adjustments to current earnings.

Key features:

- Applies county-specific income growth rates to person earnings.
- Updates earnings multiplicatively based on annual rate data.
- Maintains geographic variation in income growth patterns.
- Simple, direct adjustment without complex modeling.

Caveats:

- Requires an `income_rates` table with columns: year, lcm_county_id, rate.
- Assumes households table has `lcm_county_id` column for geographic mapping.
- All persons in the same county receive the same proportional adjustment.
- No validation of rate values; extreme rates could produce unrealistic results.
- Designed for use as-is; users should ensure rate data is reasonable and complete.

Module function
---------------

This module does not have configuration options

.. autofunction:: demos.models.income_adjustment.income_adjustment