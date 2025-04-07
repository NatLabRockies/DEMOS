import orca
import numpy as np
import pandas as pd
from templates import estimated_models, modelmanager as mm

@orca.step("kids_moving_model")
def kids_moving_model(persons, households, get_new_households):
    """
    Running the kids moving model and updating household
    stats.

    YE: ***

    Args:
        persons (DataFrameWrapper): DataFrameWrapper of the persons table
        households (DataFrameWrapper): DataFrameWrapper of the households table

    Returns:
        None
    """
    persons_df = orca.get_table("persons").local
    persons_df["kid_moves"] = -99
    orca.add_table("persons", persons_df)

    # print("Running the kids moving model...")
    kids_moving_model = mm.get_step("kids_move")
    kids_moving_model.run()
    kids_moving = kids_moving_model.choices.astype(int)

    update_households_after_kids(persons, kids_moving, get_new_households)

def update_households_after_kids(persons, kids_moving, get_new_households):
    """
    Add and update households after kids move out.

    Modifies State Variables:
    - persons.household_id

    Args:
        persons (DataFrameWrapper): DataFrameWrapper of persons table
        households (DataFrameWrapper): DataFrameWrapper of households table
        kids_moving (pd.Series): Pandas Series of kids moving out of household

    Returns:
        None
    """

    # Kids moving to a new household conditions
    ## Condition 1: Kids flagged by kids_moving
    ## Condition 2: Households with more than 1 people
    ## Condition 3: Households with some people staying

    household_sizes = persons.local.groupby("household_id").size()
    person_household_size_index = (household_sizes.loc[persons["household_id"]] > 1).values

    ## Compute kids moving per household
    kids_moving_per_household = kids_moving.groupby(persons.local.loc[kids_moving.index, "household_id"]).sum()
    ### This re-index speeds up querying by a lot
    kids_moving_per_household = kids_moving_per_household.reindex(persons["household_id"].unique()).fillna(0)
    
    ### Household-level filter for condition 3
    household_completely_moving_index = kids_moving_per_household.loc[household_sizes.index] == household_sizes
    person_completely_moving_index = household_completely_moving_index.loc[persons["household_id"]]

    ### Combine both household conditions to know which kids we need to move
    eligeble_households_index = (person_household_size_index & ~person_completely_moving_index).values
    
    # Finally combine all filters into one
    kids_moving_index = kids_moving.reindex(persons.local.index).fillna(0).astype(bool) & eligeble_households_index

    persons.local.loc[kids_moving_index, "household_id"] = get_new_households(kids_moving_index.sum(), persons)
