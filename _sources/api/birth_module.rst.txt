Birth Module
================

This module simulates the process of new births in the synthetic population. It uses an estimated model
to determine which households have a birth event in the current year, then adds new person records
(babies) to the persons table with appropriate attributes. The module ensures unique person IDs,
assigns demographic characteristics, and handles race assignment based on household composition.

Key features:

- Applies a logit model to eligible households to predict births.
- Adds new babies to the persons table with default and inferred attributes.
- Assigns race based on household head or marks as "other" if ambiguous.
- Ensures unique person IDs across the simulation.
- Supports calibration of the birth model if configured.

Caveats:

- Babies are always assigned `relate=2` (child), `age=0`, and default values for other attributes.
- Race assignment uses household head's race if all members share the same race; otherwise, "other".
- Some attributes (e.g., `race_id`, `race`) may have duplication or missing values if not set in input data.
- Most errors are silently handled; users should ensure input data is consistent.
- Designed for use as-is; users should not override orca columns or internal logic.

Module function
---------------

Module configuration options: :py:class:`~demos.config.BirthModuleConfig`

.. autofunction:: demos.models.birth.birth