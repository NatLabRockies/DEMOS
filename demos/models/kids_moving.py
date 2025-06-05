import orca
import numpy as np
import pandas as pd
from templates import estimated_models, modelmanager as mm
import time 
from datasources import log_execution_time

@orca.step("kids_moving_model")
def kids_moving_model(persons, households):
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
    start_time = time.time()
    persons_df = orca.get_table("persons").local
    persons_df["kid_moves"] = -99
    orca.add_table("persons", persons_df)

    # print("Running the kids moving model...")
    kids_moving_model = mm.get_step("kids_move")
    kids_moving_model.run()
    kids_moving = kids_moving_model.choices.astype(int)

    update_households_after_kids(persons, households, kids_moving)
    log_execution_time(start_time, orca.get_injectable("year"), "kids_moving")

def update_households_after_kids(persons, households, kids_moving):
    """
    Add and update households after kids move out.

    YE: ***

    Args:
        persons (DataFrameWrapper): DataFrameWrapper of persons table
        households (DataFrameWrapper): DataFrameWrapper of households table
        kids_moving (pd.Series): Pandas Series of kids moving out of household

    Returns:
        None
    """
    # print("Updating households...")
    persons_df = orca.get_table("persons").local

    persons_local_cols = persons_df.columns

    households_df = orca.get_table("households").local
    households_local_cols = households_df.columns
    hh_id = (
        orca.get_table("households").to_frame(columns=["lcm_county_id"]).reset_index()
    )

    persons_df = (
        persons_df.reset_index()
        .merge(hh_id, on=["household_id"])
        .set_index("person_id")
    )

    persons_df["moveoutkid"] = kids_moving

    highest_index = households_df.index.max()
    metadata = orca.get_table("metadata").to_frame()
    max_hh_id = metadata.loc["max_hh_id", "value"]    
    current_max_household_id = max(max_hh_id, highest_index)

    kids_leaving = persons_df[persons_df["moveoutkid"] == 1]["household_id"].unique()
    single_per_household = (
        persons_df[persons_df["household_id"].isin(kids_leaving)]
        .groupby("household_id")
        .size()
        == 1
    )
    single_per_nonmoving = single_per_household[
        single_per_household == True
    ].index.unique()
    persons_df["moveoutkid"] = np.where(
        persons_df["household_id"].isin(single_per_nonmoving),
        0,
        persons_df["moveoutkid"],
    )

    kids_leaving = persons_df[persons_df["moveoutkid"] == 1]["household_id"].unique()
    entire_household_moving = (
        persons_df[
            persons_df.index.isin(
                persons_df[persons_df["moveoutkid"] == 1].index.unique()
            )
        ]
        .groupby("household_id")
        .size()
        == persons_df[persons_df["household_id"].isin(kids_leaving)]
        .groupby("household_id")
        .size()
    )
    hh_nonmoving = entire_household_moving[
        entire_household_moving == True
    ].index.unique()
    persons_df["moveoutkid"] = np.where(
        persons_df["household_id"].isin(hh_nonmoving), 0, persons_df["moveoutkid"]
    )

    persons_df.loc[persons_df["moveoutkid"] == 1, "household_id"] = (
        np.arange(persons_df["moveoutkid"].sum()) + current_max_household_id + 1
    )
    persons_df.loc[persons_df["moveoutkid"] == 1, "relate"] = 0

    new_hh = persons_df.loc[persons_df["moveoutkid"] == 1].copy()

    persons_df = persons_df.drop(persons_df[persons_df["moveoutkid"] == 1].index)
    # add to orca
    persons_df["person"] = 1
    persons_df["is_head"] = np.where(persons_df["relate"] == 0, 1, 0)
    persons_df["race_head"] = persons_df["is_head"] * persons_df["race_id"]
    persons_df["age_head"] = persons_df["is_head"] * persons_df["age"]
    persons_df["hispanic_head"] = persons_df["is_head"] * persons_df["hispanic"]
    persons_df["child"] = np.where(persons_df["relate"].isin([2, 3, 4, 7, 9, 14]), 1, 0)
    persons_df["senior"] = np.where(persons_df["age"] >= 65, 1, 0)
    persons_df["age_gt55"] = np.where(persons_df["age"] >= 55, 1, 0)

    persons_df = persons_df.sort_values("relate")

    old_agg_household = persons_df.groupby("household_id").agg(
        income=("earning", "sum"),
        race_of_head=("race_head", "sum"),
        age_of_head=("age_head", "sum"),
        workers=("worker", "sum"),
        hispanic_status_of_head=("hispanic_head", "sum"),
        seniors=("senior", "sum"),
        persons=("person", "sum"),
        age_gt55=("age_gt55", "sum"),
        children=("child", "sum"),
    )
    old_agg_household["hh_age_of_head"] = np.where(
        old_agg_household["age_of_head"] < 35,
        "lt35",
        np.where(old_agg_household["age_of_head"] < 65, "gt35-lt65", "gt65"),
    )
    old_agg_household["hh_race_of_head"] = np.where(
        old_agg_household["race_of_head"] == 1,
        "white",
        np.where(
            old_agg_household["race_of_head"] == 2,
            "black",
            np.where(old_agg_household["race_of_head"].isin([6, 7]), "asian", "other"),
        ),
    )
    old_agg_household["hispanic_head"] = np.where(
        old_agg_household["hispanic_status_of_head"] == 1, "yes", "no"
    )
    old_agg_household["hh_size"] = np.where(
        old_agg_household["persons"] == 1,
        "one",
        np.where(
            old_agg_household["persons"] == 2,
            "two",
            np.where(old_agg_household["persons"] == 3, "three", "four or more"),
        ),
    )
    old_agg_household["hh_children"] = np.where(
        old_agg_household["children"] >= 1, "yes", "no"
    )
    old_agg_household["hh_income"] = np.where(
        old_agg_household["income"] < 30000,
        "lt30",
        np.where(
            old_agg_household["income"] < 60,
            "gt30-lt60",
            np.where(
                old_agg_household["income"] < 100,
                "gt60-lt100",
                np.where(old_agg_household["income"] < 150, "gt100-lt150", "gt150"),
            ),
        ),
    )
    old_agg_household["hh_workers"] = np.where(
        old_agg_household["workers"] == 0,
        "none",
        np.where(old_agg_household["workers"] == 1, "one", "two or more"),
    )
    old_agg_household["hh_seniors"] = np.where(
        old_agg_household["seniors"] >= 1, "yes", "no"
    )
    old_agg_household["gt55"] = np.where(old_agg_household["age_gt55"] > 0, 1, 0)
    old_agg_household["gt2"] = np.where(old_agg_household["persons"] > 2, 1, 0)

    households_df.update(old_agg_household)

    new_hh["person"] = 1
    new_hh["is_head"] = np.where(new_hh["relate"] == 0, 1, 0)
    new_hh["race_head"] = new_hh["is_head"] * new_hh["race_id"]
    new_hh["age_head"] = new_hh["is_head"] * new_hh["age"]
    new_hh["hispanic_head"] = new_hh["is_head"] * new_hh["hispanic"]
    new_hh["child"] = np.where(new_hh["relate"].isin([2, 3, 4, 14]), 1, 0)
    new_hh["senior"] = np.where(new_hh["age"] >= 65, 1, 0)
    new_hh["age_gt55"] = np.where(new_hh["age"] >= 55, 1, 0)
    new_hh["car"] = np.random.choice([0, 1, 2], size=new_hh.shape[0])

    new_hh = new_hh.sort_values("relate")

    agg_households = new_hh.groupby("household_id").agg(
        income=("earning", "sum"),
        race_of_head=("race_head", "sum"),
        age_of_head=("age_head", "sum"),
        workers=("worker", "sum"),
        hispanic_status_of_head=("hispanic_head", "sum"),
        seniors=("senior", "sum"),
        lcm_county_id=("lcm_county_id", "first"),
        persons=("person", "sum"),
        age_gt55=("age_gt55", "sum"),
        cars=("car", "sum"),
        children=("child", "sum"),
    )
    agg_households["serialno"] = "-1"
    agg_households["tenure"] = np.random.choice(
        households_df["tenure"].unique(), size=agg_households.shape[0]
    )  # Needs changed
    agg_households["recent_mover"] = np.random.choice(
        households_df["recent_mover"].unique(), size=agg_households.shape[0]
    )
    agg_households["sf_detached"] = np.random.choice(
        households_df["sf_detached"].unique(), size=agg_households.shape[0]
    )
    agg_households["hh_age_of_head"] = np.where(
        agg_households["age_of_head"] < 35,
        "lt35",
        np.where(agg_households["age_of_head"] < 65, "gt35-lt65", "gt65"),
    )
    agg_households["hh_race_of_head"] = np.where(
        agg_households["race_of_head"] == 1,
        "white",
        np.where(
            agg_households["race_of_head"] == 2,
            "black",
            np.where(agg_households["race_of_head"].isin([6, 7]), "asian", "other"),
        ),
    )
    agg_households["hispanic_head"] = np.where(
        agg_households["hispanic_status_of_head"] == 1, "yes", "no"
    )
    agg_households["hh_size"] = np.where(
        agg_households["persons"] == 1,
        "one",
        np.where(
            agg_households["persons"] == 2,
            "two",
            np.where(agg_households["persons"] == 3, "three", "four or more"),
        ),
    )
    agg_households["hh_cars"] = np.where(
        agg_households["cars"] == 0,
        "none",
        np.where(agg_households["cars"] == 1, "one", "two or more"),
    )
    agg_households["hh_children"] = np.where(
        agg_households["children"] >= 1, "yes", "no"
    )
    agg_households["hh_income"] = np.where(
        agg_households["income"] < 30000,
        "lt30",
        np.where(
            agg_households["income"] < 60,
            "gt30-lt60",
            np.where(
                agg_households["income"] < 100,
                "gt60-lt100",
                np.where(agg_households["income"] < 150, "gt100-lt150", "gt150"),
            ),
        ),
    )
    agg_households["hh_workers"] = np.where(
        agg_households["workers"] == 0,
        "none",
        np.where(agg_households["workers"] == 1, "one", "two or more"),
    )
    agg_households["tenure_mover"] = np.random.choice(
        households_df["tenure_mover"].unique(), size=agg_households.shape[0]
    )
    agg_households["hh_seniors"] = np.where(agg_households["seniors"] >= 1, "yes", "no")
    agg_households["block_id"] = np.random.choice(
        households_df["block_id"].unique(), size=agg_households.shape[0]
    )
    agg_households["gt55"] = np.where(agg_households["age_gt55"] > 0, 1, 0)
    agg_households["gt2"] = np.where(agg_households["persons"] > 2, 1, 0)
    agg_households["hh_type"] = 0  # CHANGE THIS

    households_df["birth"] = -99
    households_df["divorced"] = -99

    agg_households["birth"] = -99
    agg_households["divorced"] = -99

    households_df = pd.concat(
        [households_df[households_local_cols], agg_households[households_local_cols]]
    )
    persons_df = pd.concat([persons_df[persons_local_cols], new_hh[persons_local_cols]])
    # print(households_df["hh_size"].unique())
    # add to orca
    orca.add_table("households", households_df[households_local_cols])
    orca.add_table("persons", persons_df[persons_local_cols])
    # orca.add_injectable(
    #     "max_hh_id", max(households_df.index.max(), orca.get_injectable("max_hh_id"))
    # )

    metadata = orca.get_table("metadata").to_frame()
    max_hh_id = metadata.loc["max_hh_id", "value"]
    max_p_id = metadata.loc["max_p_id", "value"]
    if households_df.index.max() > max_hh_id:
        metadata.loc["max_hh_id", "value"] = households_df.index.max()
    if persons_df.index.max() > max_p_id:
        metadata.loc["max_p_id", "value"] = persons_df.index.max()
    orca.add_table("metadata", metadata)

    # print("Updating kids moving metrics...")
    kids_moving_table = orca.get_table("kids_move_table").to_frame()
    if kids_moving_table.empty:
        kids_moving_table = pd.DataFrame(
            [kids_moving_table.sum()], columns=["kids_moving_out"]
        )
    else:
        new_kids_moving_table = pd.DataFrame(
            {"kids_moving_out": kids_moving_table.sum()}
        )
        kids_moving_table = pd.concat([kids_moving_table, 
                                       new_kids_moving_table],
                                      ignore_index=True)
    orca.add_table("kids_move_table", kids_moving_table)
