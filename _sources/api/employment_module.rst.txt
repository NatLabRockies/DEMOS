Employment Module
=================

This module simulates labor force transitions for persons in the synthetic population.
It applies estimated models to determine which individuals enter or exit the workforce,
updates employment status and earnings, and provides household-level employment and income summaries.

* Supports calibration and simultaneous calibration procedures.
* Relies on estimated logit models for labor force transitions.

Module function
---------------

Module configuration options: :py:class:`~demos.config.EmploymentModuleConfig`

.. autofunction:: demos.models.employment.employment

Orca Columns
------------
.. autofunction:: demos.models.employment.new_earning
.. autofunction:: demos.models.employment.income_dist
.. autofunction:: demos.models.employment.hh_workers
.. autofunction:: demos.models.employment.income

Other Functions
---------------
.. automodule:: demos.models.employment
   :members:
   :undoc-members:
   :exclude-members: employment, new_earning, income_dist, hh_workers, income