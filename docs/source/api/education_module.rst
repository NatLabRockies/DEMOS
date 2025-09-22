Education Module
================

This module simulates educational attainment and student status transitions for persons in the synthetic population.
It uses an estimated model to determine which students drop out, advances students through grades and degrees,
and maintains proportions of high school and GED graduates. The module updates education years and student status
in the persons table, and provides orca columns for education groupings.

Key features:

- Applies a logit model to determine which students stop attending school.
- Advances students through grades, high school, and college based on model predictions and observed proportions.
- Maintains realistic proportions of high school graduates and GED recipients.
- Updates `edu` (education years/level) and `student` status in place.
- Provides orca columns for education groupings and proportions.

Caveats:

- Only persons older than 15 and currently students are considered for dropout modeling.
- Proportions for transitions (e.g., GED vs. diploma) are maintained using observed data.
- Some transitions use random assignment based on empirical proportions.
- Most errors are silently handled; users should ensure input data is consistent.
- Designed for use as-is; users should not override orca columns or internal logic.

Module function
---------------

This module does not have configuration options

.. autofunction:: demos.models.education.education