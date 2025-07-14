import orca
import numpy as np
import pandas as pd
from templates.utils.models import columns_in_formula
from templates import estimated_models, modelmanager as mm
from templates.utils import transition
from templates.utils.transition import GrowthRateTransition

import time
from datasources import log_execution_time

@orca.step('household_rebalancing')
def household_rebalancing(households, persons, year, get_new_households, get_new_person_id, rebalanced_households, rebalanced_persons):
    start_time = time.time()
    marital_rebalanced = orca.get_table("marital_rebalanced")
    marital_rebalanced.local = pd.concat([marital_rebalanced.local,
                                          pd.DataFrame([[year, (orca.get_table("persons").local.MAR == 1).sum(), (orca.get_table("persons").local.MAR == 3).sum()]],
                                                       columns=["year", "married_original", "divorced_original"])])

    CONTROL_TABLE = "hsize_ct"
    GEOID_COL = "lcm_county_id"
    CONTROL_COL = "hh_size"

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
    hh_difference = control_table_wrapped.local.loc[year].set_index([GEOID_COL, CONTROL_COL])[value_column].loc[current_count.index] - current_count
    
    # TODO: Add assertions about rows being enough to make the sampling

    to_remove_hh = []
    to_duplicate_hh = []
    # for geo_id, sub_series in hh_difference.groupby(level=0):
    #     geo_index = index_df[GEOID_COL] == geo_id
    #     for sub_index, adjustment in sub_series.items():
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


def household_transition_old(households, persons, year, metadata):
    # breakpoint()
    # at this breakpoint, look at the persons table
    linked_tables = {'persons': (persons, 'household_id')}
    if ('annual_household_control_totals' in orca.list_tables()) and ('use_database_control_totals' not in orca.list_injectables()):
        control_totals = orca.get_table('annual_household_control_totals').to_frame()
        full_transition(households, control_totals, 'total', year, 'block_id', linked_tables=linked_tables)
    elif ('household_growth_rate' in orca.list_injectables()) and ('use_database_control_totals' not in orca.list_injectables()):
        rate = orca.get_injectable('household_growth_rate')
        simple_transition(households, rate, 'block_id', set_year_built=True, linked_tables=linked_tables)
    elif 'hsize_ct' in orca.list_tables():
        control_totals = orca.get_table('hsize_ct').to_frame()
        full_transition(households, control_totals, 'total_number_of_households', year, 'block_id')
    else:
        control_totals = orca.get_table('hct').to_frame()
        if 'hh_type' in control_totals.columns:
            if control_totals[control_totals.index == year].hh_type.min() == -1:
                control_totals = control_totals[['total_number_of_households']]
        full_transition(households, control_totals, 'total_number_of_households', year, 'block_id', linked_tables=linked_tables)
    households_df = orca.get_table('households').local
    # households_df.loc[households_df['block_id'] == "-1", 'lcm_county_id'] = "-1"
    households_df.index.rename('household_id', inplace=True)
    persons_df = orca.get_table('persons').local
    # persons = persons.loc[persons['household_id'].isin(households.index.unique())]
    orca.add_table('households', households_df)
    orca.add_table('persons', persons_df)
    # orca.add_injectable(
    #     'max_hh_id', max(orca.get_injectable("max_hh_id"), households.index.max())
    # )
    metadata_df = orca.get_table('metadata').to_frame()
    max_hh_id = metadata_df.loc['max_hh_id', 'value']
    max_p_id = metadata_df.loc['max_p_id', 'value']
    if households_df.index.max() > max_hh_id:
        metadata_df.loc['max_hh_id', 'value'] = households_df.index.max()
    if persons_df.index.max() > max_p_id:
        metadata_df.loc['max_p_id', 'value'] = persons_df.index.max()
    orca.add_table('metadata', metadata_df)
    # breakpoint()


def full_transition(
    agents,
    ct,
    totals_column,
    year,
    location_fname,
    linked_tables=None,
    accounting_column=None,
    set_year_built=False,
):
    """
    Run a transition model based on control totals specified in the usual UrbanSim way
    Parameters
    ----------
    agents : DataFrameWrapper
        Table to be transitioned
    agent_controls : DataFrameWrapper
        Table of control totals
    totals_column : str
        String indicating the agent_controls column to use for totals.
    year : int
        The year, which will index into the controls
    location_fname : str
        The field name in the resulting dataframe to set to -1 (to unplace
        new agents)
    linked_tables: dict, optional
        Sets the tables linked to new or removed agents to be updated with
        dict of {'table_name':(DataFrameWrapper, 'link_id')}
    accounting_column : str, optional
        Name of column with accounting totals/quantities to apply toward the
        control. If not provided then row counts will be used for accounting.
    set_year_built: boolean
        Indicates whether to update 'year_built' columns with current
        simulation year
    Returns
    -------
    Nothing
    """
    print("Running full transition")
    if "agg_sector" in ct.columns:
        ct["agg_sector"] = ct["agg_sector"].astype("str")
    add_cols = [col for col in ct.columns if col != totals_column]
    add_cols = [col for col in add_cols if col not in agents.local.columns]
    agnt = agents.to_frame(list(agents.local.columns) + add_cols)
    print("Total agents before transition: {}".format(len(agnt)))
    idx_name = agnt.index.name
    if agents.name == "households":
        agnt = agnt.reset_index()
        hh_sizes = agnt["hh_size"].unique()
        # print(agnt["hh_size"].unique())
        updated = pd.DataFrame()
        added = pd.Index([])
        copied = pd.Index([])
        removed = pd.Index([])
        ct["lcm_county_id"] = ct["lcm_county_id"].astype(str)
        max_hh_id = agnt.index.max()
        for size in hh_sizes:
            # print(size)
            agnt_sub = agnt[agnt["hh_size"] == size].copy()
            # print(agnt_sub.shape[0])
            ct_sub = ct[ct["hh_size"] == size].copy()
            # print(ct_sub.shape[0])
            tran = transition.TabularTotalsTransition(ct_sub, totals_column, accounting_column)
            # print(ct_sub.dtypes)
            updated_sub, added_sub, copied_sub, removed_sub = tran.transition(agnt_sub, year)
            updated_sub.loc[added_sub, location_fname] = "-1"
            # print("Edits shape:")
            # print("==================")
            # print(updated_sub.shape)
            # print(added_sub.shape)
            # print(copied_sub.shape)
            # print(removed_sub.shape)
            # print("==================")
            # max_hh_id = max(agnt.index.max(), updated_sub.index.max())
            # breakpoint()
            if updated.empty:
                updated = updated_sub.copy()
                # print("updated_sub index:", updated_sub.index.name)
                # print("updated_sub has duplicates:", updated.index.has_duplicates)
            else:
                # print("updated_sub index: ", updated_sub.index.name)
                # print("updated before index:", updated.index.name)
                updated = pd.concat([updated, updated_sub])
                # print("updated after index:", updated.index.name)
                # print("Updated Shape after concat:", updated.shape)

            if added.empty:
                added = added_sub.copy()
            else:
                added = added.append(added_sub)
            if copied.empty:
                copied = copied_sub.copy()
            else:
                copied = copied.append(copied_sub)
            if removed.empty:
                removed = removed_sub.copy()
            else:
                # print(removed)
                # print(type(removed_sub))
                removed = removed.append(removed_sub)
        # breakpoint()
        # removed_df = agnt.index.isin(removed)
        # updated = agnt[~removed_df].copy()
        # new_agnts = agnt.loc[copied].copy()

        # updated = pd.concat([updated, new_agnts])
    else:
        tran = transition.TabularTotalsTransition(ct, totals_column, accounting_column)
        updated, added, copied, removed = tran.transition(agnt, year)
        # print(type(updated))
        # print(type(added))
        # print(type(copied))
        # print(type(removed))
    # breakpoint()
    if (len(added) > 0) & (agents.name == "households"):
        metadata = orca.get_table("metadata").to_frame()
        max_hh_id = metadata.loc["max_hh_id", "value"]
        max_p_id = metadata.loc["max_p_id", "value"]
        if updated.loc[added, "household_id"].min() < max_hh_id:
        # if added.min() < max_hh_id:
            # print("HERE")
            # breakpoint()
            persons_df = orca.get_table("persons").local.reset_index()
            unique_hh_ids = updated["household_id"].unique()
            persons_old = persons_df[persons_df["household_id"].isin(unique_hh_ids)]
            updated = updated.sort_values(["household_id"])
            # Get households that are sampled/duplicated
            updated["cum_count"] = updated.groupby("household_id").cumcount()
            # NEW CODE 10/27
            updated = updated.sort_values(by=["cum_count"], ascending=False)
            updated.loc[:,"new_household_id"] = np.arange(updated.shape[0]) + max_hh_id + 1
            updated.loc[:,"new_household_id"] = np.where(updated["cum_count"]>0, updated["new_household_id"], updated["household_id"])
            sampled_persons = updated.merge(persons_df, how="left", left_on="household_id", right_on="household_id")
            sampled_persons = sampled_persons.sort_values(by=["cum_count"], ascending=False)
            sampled_persons.loc[:,"new_person_id"] = np.arange(sampled_persons.shape[0]) + max_p_id + 1
            sampled_persons.loc[:,"person_id"] = np.where(sampled_persons["cum_count"]>0, sampled_persons["new_person_id"], sampled_persons["person_id"])
            sampled_persons.loc[:,"household_id"] = np.where(sampled_persons["cum_count"]>0, sampled_persons["new_household_id"], sampled_persons["household_id"])
            updated.loc[:,"household_id"] = updated.loc[:, "new_household_id"]
            
            # sampled_households = updated[updated["cum_count"]>0]
            # sampled_households.loc[:, location_fname] = "-1"
            # old_households = updated[updated["cum_count"]==0]
            # Sample individuals from such households
            # sampled_households["new_household_id"] = np.arange(sampled_households.shape[0]) + max_hh_id + 1
            # sampled_persons = sampled_households.merge(persons_df, how="left", left_on="household_id", right_on="household_id")
            # # Update the id for households
            # sampled_households["household_id"] = sampled_households["new_household_id"].copy()
            # sampled_persons["person_id"] = np.arange(sampled_persons.shape[0]) + max_p_id + 1
            # sampled_persons["household_id"] = sampled_persons["new_household_id"].copy()
            # persons_df = pd.concat([persons_old, sampled_persons])
            # updated = pd.concat([old_households, sampled_households])
            # print((updated[location_fname]=="-1").sum())
            sampled_persons.set_index("person_id", inplace=True, drop=True)
            updated.set_index("household_id", inplace=True, drop=True)
            persons_local_columns = orca.get_injectable("persons_local_cols")
            # breakpoint()
            # At this breakpoint, figure out what changes from the previous one.
            orca.add_table("persons", sampled_persons.loc[:,persons_local_columns])
        # if added.min() < max_hh_id:
        #     # breakpoint()
        #     # reset "added" row IDs so that new rows do not assign
        #     # IDs of previously removed rows.
        #     new_max = max(agnt.index.max(), updated.index.max())
        #     new_added = np.arange(len(added)) + max_hh_id + 1
        #     updated["new_idx"] = None
        #     print(len(added))
        #     print(len(new_added))
        #     print(len(copied))
        #     print(len(removed))
        #     print(updated.shape)
        #     print(new_agnts.shape)
        #     # breakpoint()

        #     persons = orca.get_table("persons").local
        #     persons_removed = persons["household_id"].isin(removed)
        #     persons = persons[~persons_removed].copy()
        #     # person_times = new_agnts.groupby('household_id').size().to_frame('times').reset_index()
        #     persons_copied = persons[persons["household_id"].isin(copied)].reset_index()
        #     # person_times = person_times.merge(persons_copied, on="household_id")
        #     # persons_copied = person_times.loc[person_times.index.repeat(person_times['times'])]
        #     # breakpoint()

        #     new_agnts["new_idx"] = None
        #     new_agnts["new_idx"] = new_added
        #     # not_added = updated['new_idx'].isnull()
        #     # breakpoint()
        #     new_agnts.index.name = idx_name
        #     # breakpoint()
        #     # new_agnts
        #     persons_multiple = (
        #         new_agnts.groupby(["household_id", "new_idx"]).size().sort_index().to_frame("times").reset_index())
        #     new_persons = persons_copied.merge(
        #         persons_multiple, on="household_id", how="outer"
        #     )
        #     new_persons["household_id"] = new_persons["new_idx"].copy()
        #     # max_p_id = orca.get_table("persons").local.index.max()
        #     new_persons["person_id"] = np.arange(new_persons.shape[0]) + max_p_id + 1
        #     new_persons.set_index("person_id", inplace=True, drop=True)
        #     # breakpoint()
        #     persons = pd.concat([persons, new_persons])
        #     # breakpoint()

        #     updated["new_idx"] = updated.index.values
        #     # updated.loc[not_added, 'new_idx'] = updated.loc[not_added].index.values
        #     updated = pd.concat([updated, new_agnts])
        #     print(updated.shape[0])
        #     updated.set_index("new_idx", inplace=True, drop=True)
        #     updated.index.name = idx_name
        #     added = new_added
        #     # orca.get_table('persons')
        #     persons_local_columns = orca.get_table("persons").local_columns
        #     orca.add_table("persons", persons[persons_local_columns])
    # breakpoint()        
    # print("Updated shape:", updated.shape[0])
    # print("Added shape:", added.shape[0])
    # print("Removed shape:", removed.shape[0])
    # print("Copied shape:", copied.shape[0])
    # updated.loc[added, location_fname] = "-1"
    if agents.name != "households":
        updated.loc[added, location_fname] = "-1"
    if set_year_built:
        updated.loc[added, "year_built"] = year
    updated_links = {}
    if linked_tables:
        for table_name, (table, col) in linked_tables.items():
            print("updating linked table {}".format(table_name))
            updated_links[table_name] = update_linked_table(
                table, col, added, copied, removed
            )
            orca.add_table(table_name, updated_links[table_name])
    print("Total agents after transition: {}".format(len(updated)))
    orca.add_table(agents.name, updated[agents.local_columns])
    return updated, added, copied, removed


def simple_transition(
    tbl, rate, location_fname, linked_tables={}, set_year_built=False
):
    """
    Run a simple growth rate transition model

    Parameters
    ----------
    tbl : DataFrameWrapper
        Table to be transitioned
    rate : float
        Growth rate
    linked_tables : dict, optional
        Sets the tables linked to new or removed agents to be updated with dict of
        {'table_name':(DataFrameWrapper, 'link_id')}
    location_fname : str
        The field name in the resulting dataframe to set to -1 (to unplace
        new agents)
    Returns
    -------
    Nothing
    """
    print("Running simple transition with ", rate * 100, "% rate")
    transition = GrowthRateTransition(rate)
    df_base = tbl.to_frame(tbl.local_columns)
    print("%d agents before transition" % len(df_base.index))
    df, added, copied, removed = transition.transition(df_base, None)
    print("%d agents after transition" % len(df.index))
    if (len(added) > 0) & (tbl.name == "households"):
        metadata = orca.get_table("metadata").to_frame()
        max_hh_id = metadata.loc["max_hh_id", "value"]
        if added.min() < max_hh_id:
            # breakpoint()
            # reset "added" row IDs so that new rows do not assign
            # IDs of previously removed rows.
            new_max = max(df_base.index.max(), df.index.max())
            new_added = np.arange(len(added)) + new_max + 1
            idx_name = df.index.name
            df["new_idx"] = None
            df.loc[added, "new_idx"] = new_added
            not_added = df["new_idx"].isnull()
            # breakpoint()
            df.loc[not_added, "new_idx"] = df.loc[not_added].index.values
            df.set_index("new_idx", inplace=True, drop=True)
            df.index.name = idx_name
            added = new_added

    df.loc[added, location_fname] = "-1"

    if set_year_built:
        df.loc[added, "year_built"] = orca.get_injectable("year")
    updated_links = {}
    for table_name, (table, col) in linked_tables.items():
        updated_links[table_name] = update_linked_table(
            table, col, added, copied, removed
        )
        orca.add_table(table_name, updated_links[table_name])
    orca.add_table(tbl.name, df)


def update_linked_table(tbl, col_name, added, copied, removed):
    """
    Copy and update rows in a table that has a column referencing another
    table that has had rows added via copying.
    Parameters
    ----------
    tbl : DataFrameWrapper
        Table to update with new or removed rows.
    col_name : str
        Name of column in `table` that corresponds to the index values
        in `copied` and `removed`.
    added : pandas.Index
        Indexes of rows that are new in the linked table.
    copied : pandas.Index
        Indexes of rows that were copied to make new rows in linked table.
    removed : pandas.Index
        Indexes of rows that were removed from the linked table.
    Returns
    -------
    updated : pandas.DataFrame
    """
    # max ID should be preserved before rows are removed
    # otherwise new rows could have ID of what was removed.
    max_id = tbl.index.values.max()

    # max ID should be preserved before rows are removed
    # otherwise new rows could have ID of what was removed.
    max_id = tbl.index.values.max()

    # handle removals
    table = tbl.local
    table = table.loc[~table[col_name].isin(set(removed))]
    removed = table.loc[table[col_name].isin(set(removed))]
    if added is None or len(added) == 0:
        return table

    # map new IDs to the IDs from which they were copied
    id_map = pd.concat(
        [pd.Series(copied, name=col_name), pd.Series(added, name="temp_id")], axis=1
    )

    # join to linked table and assign new id
    new_rows = id_map.merge(table, on=col_name)
    new_rows.drop(col_name, axis=1, inplace=True)
    new_rows.rename(columns={"temp_id": col_name}, inplace=True)

    # index the new rows
    starting_index = max_id + 1
    new_rows.index = np.arange(
        starting_index, starting_index + len(new_rows), dtype=np.int)
    new_rows.index.name = table.index.name

    return pd.concat([table, new_rows])