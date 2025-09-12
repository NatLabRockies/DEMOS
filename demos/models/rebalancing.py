# fmt: off
import orca
import numpy as np
import pandas as pd
from templates.utils.models import columns_in_formula
from templates import estimated_models, modelmanager as mm
from templates.utils import transition
from templates.utils.transition import GrowthRateTransition
from config import DEMOSConfig, HHRebalancingModuleConfig, SimultaneousCalibrationConfig, get_config

import time
from logging_logic import log_execution_time

STEP_NAME = "household_rebalancing"

@orca.step(STEP_NAME)
def household_rebalancing(households, persons, year, get_new_households, get_new_person_id, rebalanced_households, rebalanced_persons):
    """
    Adjust household counts to match control totals by geography and household size.

    This step compares current household counts with control totals and duplicates or removes
    households as needed. It maintains population consistency by also updating the persons table
    and stores removed records for tracking purposes.

    Parameters
    ----------
    households : orca.Table
        The households table containing household-level attributes.
    persons : orca.Table
        The persons table containing individual-level attributes.
    year : int
        The current simulation year.
    get_new_households : callable
        Function to generate new unique household IDs.
    get_new_person_id : callable
        Function to generate new unique person IDs.
    rebalanced_households : orca.Table
        Table for storing removed household records.
    rebalanced_persons : orca.Table
        Table for storing removed person records.

    Returns
    -------
    None

    Notes
    -----
    - Modifies households and persons tables in place by adding/removing records.
    - Uses module configuration to determine control table and column mappings.
    - Tracks marital status before and after operations in marital_rebalanced table.
    - Skips processing if no control data exists for the current year.
    - Sampling with replacement occurs when duplicating more households than available.
    """
    start_time = time.time()

    # Load calibration config
    demos_config: DEMOSConfig = get_config()
    module_config: HHRebalancingModuleConfig = demos_config.hh_rebalancing_module_config

    marital_rebalanced = orca.get_table("marital_rebalanced")
    marital_rebalanced.local = pd.concat([marital_rebalanced.local,
                                          pd.DataFrame([[year, (orca.get_table("persons").local.MAR == 1).sum(), (orca.get_table("persons").local.MAR == 3).sum()]],
                                                       columns=["year", "married_original", "divorced_original"])])

    CONTROL_TABLE = module_config.control_table
    GEOID_COL = module_config.geoid_col
    CONTROL_COL = module_config.control_col

    control_table_wrapped = orca.get_table(CONTROL_TABLE)
    assert GEOID_COL in control_table_wrapped.local_columns, f"{GEOID_COL} must be in {CONTROL_TABLE}"
    assert CONTROL_COL in control_table_wrapped.local_columns, f"{CONTROL_COL} must be in {CONTROL_TABLE}"
    assert control_table_wrapped.index.name == "year", f"The index of {CONTROL_TABLE} must be 'year'"
    assert len(control_table_wrapped.local_columns) == 3, f"{CONTROL_TABLE} needs to have exactly 3 columns: {GEOID_COL}, {CONTROL_COL} and the value column"
    assert persons.household_id.nunique() == households.index.nunique(), f"`persons` and `households` tables do not have coherent sizes. {persons.household_id.nunique()} vs. {households.index.nunique()}"

    if year not in control_table_wrapped.local.index:
        return

    value_column = [c for c in control_table_wrapped.local_columns if c not in [GEOID_COL, CONTROL_COL]][0]
    index_df = households.to_frame([GEOID_COL, CONTROL_COL]).sort_values([GEOID_COL, CONTROL_COL])
    indices = index_df.groupby([GEOID_COL, CONTROL_COL]).indices
    current_count = index_df.groupby([GEOID_COL, CONTROL_COL]).size()
    hh_difference = control_table_wrapped.local.loc[year].astype({GEOID_COL: "str", CONTROL_COL:"str"}).set_index([GEOID_COL, CONTROL_COL])[value_column].loc[current_count.index] - current_count

    to_remove_hh = []
    to_duplicate_hh = []
    for (geo_id, hh_size), adjustment in hh_difference.items():
            valid_indices = index_df.index[indices[(geo_id, hh_size)]]
            selected_hh = np.random.choice(valid_indices, size=abs(adjustment), replace=(adjustment > 0) and (abs(adjustment) > len(valid_indices))).tolist()
            if adjustment < 0:
                to_remove_hh += selected_hh
            if adjustment > 0:
                to_duplicate_hh += selected_hh

    # Duplicate the households accordingly
    ## We duplicate first to reduce the chances of a household_id collision
    to_duplicate_hh.sort()
    new_hh_ids = get_new_households(len(to_duplicate_hh)) # Remeber that this creates household rows
    new_hh_rows = households.local.loc[to_duplicate_hh].copy()
    new_hh_rows.index = new_hh_ids

    hh_mapping = pd.DataFrame({
        "orig_hh":  to_duplicate_hh,
        "new_hh":   new_hh_ids
        })
    new_person_rows = persons.local[persons.household_id.isin(to_duplicate_hh)].copy()
    new_person_rows = new_person_rows.merge(hh_mapping, left_on="household_id", right_on="orig_hh")
    new_person_rows["household_id"] = new_person_rows["new_hh"]
    new_person_rows.drop(["orig_hh", "new_hh"], inplace=True, axis=1)
    # new_person_rows.household_id = new_person_rows.household_id.map(dict(zip(to_duplicate_hh, new_hh_ids)))
    new_person_rows.index = get_new_person_id(len(new_person_rows))
    households.local.loc[new_hh_ids] = new_hh_rows
    persons.local = pd.concat([persons.local, new_person_rows])
    
    # Remove the households accordingly
    to_remove_hh.sort()
    rebalanced_households.local = pd.concat([rebalanced_households.local, households.local.loc[to_remove_hh]])
    rebalanced_persons.local = pd.concat([rebalanced_persons.local, persons.local[persons.household_id.isin(to_remove_hh)]])
    persons.local = persons.local[~persons.household_id.isin(to_remove_hh)]
    households.local = households.local[~households.index.isin(to_remove_hh)]

    log_execution_time(start_time, orca.get_injectable("year"), "rebalancing")
    marital_rebalanced = orca.get_table("marital_rebalanced")
    marital_rebalanced.local = pd.concat([marital_rebalanced.local,
                                          pd.DataFrame([[year, (orca.get_table("persons").local.MAR == 1).sum(), (orca.get_table("persons").local.MAR == 3).sum()]],
                                                       columns=["year", "married_after", "divorced_after"])])
