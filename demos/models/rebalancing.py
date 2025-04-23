import orca
import numpy as np
import pandas as pd
from templates.utils.models import columns_in_formula
from templates import estimated_models, modelmanager as mm

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
