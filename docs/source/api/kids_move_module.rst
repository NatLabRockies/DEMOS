Kids Moving Module
==================

This module simulates the process of children or young adults moving out of their parent's households.
It uses an estimated model to determine which eligible individuals move, and updates the persons and households
tables accordingly. The logic ensures that only a subset of eligible children move, and that new households
are created for them with appropriate geographic assignments.

Key features:

- Applies a logit model to eligible children/young adults to predict moving out.
- Updates household assignments and relationship codes for movers.
- Ensures movers are assigned to new households and that household-level attributes are updated.
- Only moves children if their departure does not empty the household (unless all are moving).
- Designed for use as-is; users should not override orca columns or internal logic.

Caveats:

- The model only considers persons with specific `relate` codes (2, 3, 4, 7, 9, 14) and age >= 16.
- Movers are only reassigned if their departure does not leave the household empty, unless all are moving.
- Geographic assignment for new households is inherited from the original household.
- Most errors are silently handled; users should ensure input data is consistent.

Module function
---------------

Module configuration options: :py:class:`~demos.config.KidsMovingModuleConfig`

.. autofunction:: demos.models.kids_moving.kids_moving