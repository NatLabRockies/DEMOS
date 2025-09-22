Fatality Module
================

This module simulates mortality events in the synthetic population using an estimated model.
It updates the persons and households tables to reflect deaths, reassigns household heads as needed,
and ensures relational consistency using a mapping table. Deceased individuals are moved to a graveyard table.

Key features:

- Applies a logit model to determine which persons die in the current year.
- Removes deceased persons from the persons table and adds them to the graveyard.
- Updates marital status: surviving spouses/partners become widowed.
- Reassigns household heads if the current head dies, using partners or the oldest remaining member.
- Updates the `relate` column for all household members using the relational adjustment mapping.
- Cleans up households and persons to maintain simulation invariants.

Caveats:

- The module assumes the presence of a `relational_adjustment_mapping` table to update relationships.
- If a household head dies and no partner remains, the oldest person becomes the new head.
- Some errors or inconsistencies (e.g., multiple spouses per household) are handled silently.
- The logic for updating relationships may need to be revisited if the mapping table changes structure.
- Designed for use as-is; users should not override orca columns or internal logic.

Module function
---------------

This module does not have configuration options

.. autofunction:: demos.models.fatality.fatality