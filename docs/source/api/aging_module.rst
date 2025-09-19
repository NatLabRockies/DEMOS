Aging Module
============

This module handles the annual aging process for all persons in the simulation.
It also provides orca columns for identifying children, seniors, and age groups.

* Increments age for all persons each simulation year.
* Identifies children and seniors based on configurable criteria.
* Categorizes persons into age groups for use in other modules and models.

Module function
---------------
Module configuration options: :py:class:`~demos.config.AgingModuleConfig`

.. autofunction:: demos.models.aging.aging

Orca Columns
------------
.. autofunction:: demos.models.aging.child
.. autofunction:: demos.models.aging.senior
.. autofunction:: demos.models.aging.age_group

Other Functions
---------------
.. automodule:: demos.models.aging
   :members:
   :undoc-members:
   :exclude-members: aging,child,senior,age_group