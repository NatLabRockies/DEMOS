import orca
import numpy as np
import pandas as pd
from templates import estimated_models, modelmanager as mm

@orca.step("birth_model")
def birth_model(persons, households, year):
    """
    Function to run the birth model at the household level.
    The function updates the persons table.

    Args:
        persons (DataFrameWrapper): DataFrameWrapper of the persons table
        households (DataFrameWrapper): DataFrameWrapper of the households table

    Returns:
        None
    """

    households_df = households.local
    households_df["birth"] = -99
    orca.add_table("households", households_df)
    households_df = households.local
    # persons = orca.get_table('persons')
    col_subset = ["sex", "age", "household_id", "relate"]
    persons_df = persons.to_frame(col_subset)
    ELIGIBILITY_COND = (
        (persons_df["sex"] == 2)
        & (persons_df["age"].between(14, 45))
        # & (persons_df["relate"].isin([0, 1, 13]))
    )
    ELIGIBILITY_COND_2 = (
        (persons_df["sex"] == 2)
        & (persons_df["age"] > 45)
    )
    ELIGIBILITY_COND_3 = (
        (persons_df["sex"] == 2)
        & (persons_df["age"] < 14)
    )

    # Subset of eligible households
    ELIGIBLE_HH = persons_df.loc[ELIGIBILITY_COND, "household_id"].unique()
    eligible_hh_df = households_df.loc[ELIGIBLE_HH]

    btable_elig_df = orca.get_table("btable_elig").to_frame()
    if btable_elig_df.empty:
        btable_elig_df = pd.DataFrame.from_dict({
            "year": [str(year)],
            "count":  [ELIGIBLE_HH.shape[0]]
            })
    else:
        btable_elig_df_new = pd.DataFrame.from_dict({
            "year": [str(year)],
            "count":  [ELIGIBLE_HH.shape[0]]
            })
        btable_elig_df = pd.concat([btable_elig_df, btable_elig_df_new], ignore_index=True)
    orca.add_table("btable_elig", btable_elig_df)
    # print("BIRTH ELIGIBILITY POP")
    # print(btable_elig_df)
    # Run model
    # print("Running the birth model...")
    birth = mm.get_step("birth")
    list_ids = str(eligible_hh_df.index.to_list())
    # print(len(eligible_hh_df.index.to_list()))
    birth.filters = "index in " + list_ids
    birth.out_filters = "index in " + list_ids

    birth.run()
    birth_list = birth.choices.astype(int)
    predicted_share = birth_list.sum() / eligible_hh_df.shape[0]
    observed_births = orca.get_table("observed_births_data").to_frame()
    target = observed_births[observed_births["year"]==year]["count"]
    target_share = target / eligible_hh_df.shape[0]

    error = np.sqrt(np.mean((birth_list.sum() - target)**2))
    print("The Birth Model Calibration:")
    calibrate_time = 0
    while error >= 1000:
        print(f"{calibrate_time} time: {error}")
        birth.fitted_parameters[0] += np.log(target.sum()/birth_list.sum())
        birth.run()
        birth_list = birth.choices.astype(int)
        predicted_share = birth_list.sum() / eligible_hh_df.shape[0]
        error = np.sqrt(np.mean((birth_list.sum() - target)**2))
        calibrate_time += 1
    print(f"{calibrate_time} time: {error}")

    # breakpoint()
    # print("Eligible households >45",
    #       persons_df.loc[ELIGIBILITY_COND_2, "household_id"].unique().shape[0])
    # print("Eligible households <14",
    #       persons_df.loc[ELIGIBILITY_COND_3, "household_id"].unique().shape[0])
    # print(eligible_hh_df.shape[0], " eligible households for birth model")
    # print(birth_list.sum(), " births")
    # print("Updating persons table with newborns...")
    update_birth(persons, households, birth_list)

    # print("Updating birth metrics...")
    btable_df = orca.get_table("btable").to_frame()
    if btable_df.empty:
        btable_df = pd.DataFrame.from_dict({
            "year": [str(year)],
            "count":  [birth_list.sum()]
            })
    else:
        btable_df_new = pd.DataFrame.from_dict({
            "year": [str(year)],
            "count":  [birth_list.sum()]
            })

        btable_df = pd.concat([btable_df, btable_df_new], ignore_index=True)
    orca.add_table("btable", btable_df)


def update_birth(persons, households, birth_list):
    """
    Update the persons tables with newborns and household sizes

    Args:
        persons (DataFrameWrapper): DataFrameWrapper of the persons table
        households (DataFrameWrapper): DataFrameWrapper of the persons table
        birth_list (pd.Series): Pandas Series of the households with newborns

    Returns:
        None
    """
    persons_df = persons.local
    households_df = households.local
    households_columns = households_df.columns

    # Pull max person index from persons table
    highest_index = persons_df.index.max()

    # Check if the pop_over_time is an empty dataframe
    grave = orca.get_table("pop_over_time").to_frame()

    # If not empty, update the highest index with max index of all people
    metadata = orca.get_table("metadata").to_frame()

    max_p_id = metadata.loc["max_p_id", "value"]

    highest_index = max(max_p_id, highest_index)

    if not grave.empty:
        graveyard = orca.get_table("graveyard")
        dead_df = graveyard.to_frame(columns=["member_id", "household_id"])
        highest_dead_index = dead_df.index.max()
        highest_index = max(highest_dead_index, highest_index)

    # Get heads of households
    heads = persons_df[persons_df["relate"] == 0]

    # Get indices of households with babies
    house_indices = list(birth_list[birth_list == 1].index)

    # Initialize babies variables in the persons table.
    babies = pd.DataFrame(house_indices, columns=["household_id"])
    babies.index += highest_index + 1
    babies.index.name = "person_id"
    babies["age"] = 0
    babies["edu"] = 0
    babies["earning"] = 0
    babies["hours"] = 0
    babies["relate"] = 2
    babies["MAR"] = 5
    babies["sex"] = np.random.choice([1, 2])
    babies["student"] = 0

    babies["person_age"] = "19 and under"
    babies["person_sex"] = babies["sex"].map({1: "male", 2: "female"})
    babies["child"] = 1
    babies["senior"] = 0
    babies["dead"] = -99
    babies["person"] = 1
    babies["work_at_home"] = 0
    babies["worker"] = 0
    babies["work_block_id"] = "-1"
    babies["work_zone_id"] = "-1"
    babies["workplace_taz"] = "-1"
    babies["school_block_id"] = "-1"
    babies["school_id"] = "-1"
    babies["school_taz"] = "-1"
    babies["school_zone_id"] = "-1"
    babies["education_group"] = "lte17"
    babies["age_group"] = "lte20"
    household_races = (
        persons_df.groupby("household_id")
        .agg(num_races=("race_id", "nunique"))
        .reset_index()
        .merge(households_df["race_of_head"].reset_index(), on="household_id")
    )
    babies = babies.reset_index().merge(household_races, on="household_id")
    babies["race_id"] = np.where(babies["num_races"] == 1, babies["race_of_head"], 9)
    babies["race"] = babies["race_id"].map(
        {
            1: "white",
            2: "black",
            3: "other",
            4: "other",
            5: "other",
            6: "other",
            7: "other",
            8: "other",
            9: "other",
        }
    )
    babies = (
        babies.reset_index()
        .merge(
            heads[["hispanic", "hispanic.1", "p_hispanic", "household_id"]],
            on="household_id",
        )
        .set_index("person_id")
    )

    # Add counter for member_id to not overlap from dead people for households
    if not grave.empty:
        all_people = pd.concat([grave, persons_df[["member_id", "household_id"]]])
    else:
        all_people = persons_df[["member_id", "household_id"]]
    max_member_id = all_people.groupby("household_id").agg({"member_id": "max"})
    max_member_id += 1
    babies = (
        babies.reset_index()
        .merge(max_member_id, left_on="household_id", right_index=True)
        .set_index("person_id")
    )
    households_babies = households_df.loc[house_indices]
    households_babies["hh_children"] = "yes"
    households_babies["persons"] += 1
    households_babies["gt2"] = np.where(households_babies["persons"] >= 2, 1, 0)
    households_babies["hh_size"] = np.where(
        households_babies["persons"] == 1,
        "one",
        np.where(
            households_babies["persons"] == 2,
            "two",
            np.where(households_babies["persons"] == 3, "three", "four or more"),
        ),
    )

    # Update the households table
    households_df.update(households_babies[households_df.columns])
    # Contactenate the final result
    combined_result = pd.concat([persons_df, babies])
    persons_local_cols = orca.get_injectable("persons_local_cols")
    households_local_cols = orca.get_injectable("households_local_cols")

    orca.add_table("persons", combined_result.loc[:, persons_local_cols])
    orca.add_table("households", households_df.loc[:, households_local_cols])
    metadata = orca.get_table("metadata").to_frame()
    max_hh_id = metadata.loc["max_hh_id", "value"]
    max_p_id = metadata.loc["max_p_id", "value"]
    if households_df.index.max() > max_hh_id:
        metadata.loc["max_hh_id", "value"] = households_df.index.max()
    if combined_result.index.max() > max_p_id:
        metadata.loc["max_p_id", "value"] = combined_result.index.max()
    
    orca.add_table("metadata", metadata)
    # orca.add_injectable("max_p_id", max(highest_index, orca.get_injectable("max_p_id")))
