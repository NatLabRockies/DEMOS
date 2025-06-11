import orca
import numpy as np
import pandas as pd
from templates import estimated_models, modelmanager as mm
import time
from datasources import log_execution_time

@orca.step("fatality_model")
def fatality_model(persons, households, observed_fatalities_data, rel_map, graveyard, year):
    """Function to run the fatality model at the persons level.
    The function also updates the persons and households tables,
    and saves the mortalities table.

    Modifies State Variables:
        - persons.MAR
        - persons.relate
        - (Adds rows from `graveyard` table)
        - (Removes rows from `persons` table)
        - (Removes rows from `households` table)

    Args:
        persons (DataFrameWrapper): DataFrameWrapper of persons table
        households (DataFrameWrapper): DataFrameWrapper of households table
    """
    start_time = time.time()

    persons["dead"] = -99
    fatality_list = run_and_calibrate_mortality_model(persons, observed_fatalities_data, year)

    # Updates necessary:
    ## - Remove rows from persons table where fatality_list == 1
    ## - Update Marital status: Married people become Widows if the spouse dies
    ## - Update Relate column if head died:
    ###   If there is a relate==1 or relate==13 person alive, they are the new head
    ###   Otherwise, oldest person is new head
    ###   In all cases, we need to use the `rel_map` table to map the rest of the relate columns

    fatality_list_idx = fatality_list.astype(bool)
    dead_people_slice = persons.local.loc[fatality_list_idx]
    households_with_dead_people = dead_people_slice.household_id.unique()
    persons_in_relevant_household_index = persons["household_id"].isin(households_with_dead_people)

    # Update Marital status
    ## If dead person is spouse or partner, head of household is now widow
    dead_partners_households = dead_people_slice[(dead_people_slice.relate == 1) | (dead_people_slice.relate == 13)]["household_id"].values
    ## Update widow heads that are still alive
    persons.local.loc[~fatality_list_idx &
                      persons["household_id"].isin(dead_partners_households) &
                      (persons["relate"] == 0), "MAR"] = 3

    # If dead person is head, spouse or partner is now widow
    dead_heads_households = dead_people_slice[dead_people_slice.relate == 0]["household_id"]
    ## Update widow partners that are still alive
    persons.local.loc[~fatality_list_idx &
                      persons["household_id"].isin(dead_heads_households) &
                      ((persons["relate"] == 1) | (persons["relate"] == 13)),
                      "MAR"] = 3

    # Updates to `relate`
    ## Select all the person_id's of alive people where the head died
    ## They are the new heads
    partner_to_head_household_ids = persons.local.loc[
        ~fatality_list_idx &
        persons["relate"].isin([1,13]) &
        persons["household_id"].isin(dead_heads_households)
    ]["household_id"]
    partner_to_head_ids = partner_to_head_household_ids.index
    
    ## Before modifying the dataframe, select the new heads for the rest of households (by age)
    rest_to_head_households = set(dead_heads_households) - set(partner_to_head_household_ids)
    rest_to_head_ids = persons.local.loc[
        ~fatality_list_idx &
        persons["household_id"].isin(rest_to_head_households)
    ].groupby("household_id")["age"].idxmax().values

    ## We need to update the relate column of all the people where the head died
    ### We only need to do this for the "rest to head" because partners keep the same relation as the previous head
    new_heads_ids = rest_to_head_ids.tolist()
    rest_to_head_all_filter = persons["household_id"].isin(rest_to_head_households) & ~fatality_list_idx
    new_heads_old_relate_by_hh = persons.local.loc[new_heads_ids, ["household_id", "relate"]].set_index("household_id")["relate"]
    head_old_relate_by_person_id = persons.local.loc[rest_to_head_all_filter].household_id.map(new_heads_old_relate_by_hh)
    person_old_relate_by_person_id = persons.local.loc[rest_to_head_all_filter].relate

    ### In order to efficiently access the rel_map dataframe, we transform relate values into
    ### indices to the numpy representation of the dataframe
    rel_map_columns = rel_map.to_frame().columns
    rel_map_index = rel_map.to_frame().index

    #### First: The column value that we should query corresponds to the
    #### old relate value of the new household head
    #### NOTE: the str transformation might need to be changed in the future if rel_map is changed
    old_head_relate_index = rel_map_columns.get_indexer(head_old_relate_by_person_id.astype(str))

    #### Second: The row value corresponds with the old relate of the
    #### person changing relate (everyone but the new head)
    old_person_relate_index = rel_map_index.get_indexer(person_old_relate_by_person_id.values)

    #### With both of these we can now get the new relate values
    #### (This step also replaces the relate value of the new heads,
    ####  we take care of that after this processing)
    #### TODO: Here we can create spouses without correct marital status
    persons.local.loc[rest_to_head_all_filter, "relate"] = rel_map.to_frame().values[old_person_relate_index, old_head_relate_index]

    ## Update relate of new heads
    persons.local.loc[rest_to_head_ids.tolist() + partner_to_head_ids.to_list(), "relate"] = 0

    # Finally, remove dead people from the persons dataframe and move them to graveyard
    dead_people = persons.local.loc[fatality_list_idx].copy()
    graveyard.local = pd.concat([graveyard.local, dead_people])
    persons.local = persons.local[~fatality_list_idx]

    # TODO: This needs to be reevaluated after the refactoring
    spouses_per_hh = (persons.relate == 1).groupby(persons.household_id).sum()
    persons.local = persons.local.loc[~persons.household_id.isin(spouses_per_hh[spouses_per_hh > 1].index)]
    households.local = households.local.reindex(sorted(persons.household_id.unique()))

    log_execution_time(start_time, orca.get_injectable("year"), "mortality")


# TODO: Refactor this
def run_and_calibrate_mortality_model(persons, observed_fatalities_data, year):
    # Observed values for calibration
    observed_fatalities = observed_fatalities_data.to_frame()

    # Get estimated model object and run it
    mortality = mm.get_step("mortality")
    mortality.run()

    if orca.is_injectable("fatality_asc"):
        mortality.fitted_parameters[0] = orca.get_injectable("fatality_asc")

    fatality_list = mortality.choices.astype(int)
    predicted_share = fatality_list.sum() / persons.local.shape[0]
    observed_fatalities = orca.get_table("observed_fatalities_data").to_frame()
    target_data = observed_fatalities[observed_fatalities["year"] == year]

    if not target_data.empty:
        target = target_data["count"].sum()
        target_share = target / persons.local.shape[0]
        print(f"The observed mortalities in {year}: {target}")

        error = np.sqrt(np.mean((fatality_list.sum() - target)**2))
        print("The Fatality Model Calibration:")
        calibrate_time = 0

        while error >= 1000:
            print(f"{calibrate_time} time: {error}")
            mortality.fitted_parameters[0] += np.log(target / fatality_list.sum())
            mortality.run()
            fatality_list = mortality.choices.astype(int)
            predicted_share = fatality_list.sum() / persons.local.shape[0]
            error = np.sqrt(np.mean((fatality_list.sum() - target)**2))
            calibrate_time += 1
        print(f"{calibrate_time} time: {error}")
        
        # Persist calibrated parameter for reuse
        orca.add_injectable("fatality_asc", mortality.fitted_parameters[0])
    else:
        print(f"No observed data available for year {year}, skipping calibration.")

    return fatality_list