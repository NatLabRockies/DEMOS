Income Adjustment Module
========================

This module updates **person-level earnings** based on county-specific annual
income growth rates.  It runs after the income prediction step and applies a
multiplicative adjustment to every person's ``earning`` attribute, keeping
geographic variation in wage growth over the simulation period.

Unlike the :doc:`income_module`, this module does not use an estimated
regression model: it reads pre-computed annual rate data from the
``income_rates`` table and directly scales existing earnings.

Key features:

- Applies county-specific income growth rates to person-level earnings each
  simulation year.
- Updates earnings multiplicatively: a rate of ``0.05`` means a 5% increase.
- Maintains geographic variation in income growth patterns across counties.

Caveats:

- Requires an ``income_rates`` table with columns ``year``, ``lcm_county_id``,
  and ``rate`` (loaded via the ``[[tables]]`` section of the configuration).
- Assumes the households table has a ``lcm_county_id`` column for geographic
  mapping.
- All persons in the same county receive the same proportional adjustment.
- No validation of rate values is performed; extreme rates could produce
  unrealistic results.

.. seealso::

   :doc:`income_module` for the companion module that predicts household income
   from a regression model and optionally applies CPI-based inflation scaling.

Module function
---------------

This module does not have dedicated configuration options.

.. autofunction:: demos.models.income_adjustment.income_adjustment
