import orca
from templates import estimated_models, modelmanager as mm
import time
from logging_logic import log_execution_time
from templates.utils.models import columns_in_formula

STEP_NAME = "kids_moving"
REQUIRED_COLUMNS = [
    "persons.age",
    "persons.relate",
]

@orca.step(STEP_NAME)
def kids_moving(persons, households, get_new_households):
    """
    Executes the `kids_move` estimated model and updates the household of kids
    moving out of their parent's home accordingly.

    **Required tables:**
        - persons
        - households

    **Modifies State Variables:**
        - persons.household_id
        - persons.relate
        - households.lcm_county_id
    """
    start_time = time.time()

    # Get model data
    model = mm.get_step("kids_move")
    model_variables = columns_in_formula(model.model_expression)
    model_filters = (persons.relate.isin([2, 3, 4, 7, 9, 14])) & (persons.age >= 16)
    model_data = persons.to_frame(model_variables)[model_filters]

    kids_moving = model.predict(model_data).astype(int)

    update_households_after_kids(persons, households, kids_moving, get_new_households)
    log_execution_time(start_time, orca.get_injectable("year"), "kids_moving")

def update_households_after_kids(persons, households, kids_moving, get_new_households):
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

    # Get the old household_id for the moving kids to retrieve the county_id
    # TODO: Parametrize county_id
    old_household_id = persons.local.loc[kids_moving_index, "household_id"].values
    county_assignment = households.local.loc[old_household_id, "lcm_county_id"].values

    new_households = get_new_households(kids_moving_index.sum())
    persons.local.loc[kids_moving_index, "household_id"] = new_households
    persons.local.loc[kids_moving_index, "relate"] = 0
    households.local.loc[new_households, "lcm_county_id"] = county_assignment
