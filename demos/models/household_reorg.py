import time
import orca
import numpy as np
import pandas as pd
from templates import estimated_models, modelmanager as mm
from templates.utils.models import columns_in_formula
from config import DEMOSConfig, HHReorgModuleConfig, get_config
from .marriage import update_married_households_random, update_divorce
from loguru import logger
from logging_logic import log_execution_time

from templates.calibration.procedures import SimultaneousCalibrationConfig

STEP_NAME = "household_reorg"
REQUIRED_COLUMNS = ["persons.age", "persons.MAR", "persons.relate"]


@orca.step(STEP_NAME)
def household_reorg(persons, households, year, get_new_households):
    """
    Main step for household reorganization: executes marriage, divorce, and cohabitation sub-models.

    This function runs all three estimated models, optionally calibrates them to observed data, and updates
    the persons and households tables in place. It handles the creation and dissolution of households,
    changes in marital status, and cohabitation transitions.

    Parameters
    ----------
    persons : orca.Table
        The persons table containing individual-level attributes.
    households : orca.Table
        The households table containing household-level attributes.
    year : int
        The current simulation year.
    get_new_households : callable
        Function to generate new unique household IDs as needed.

    Returns
    -------
    None

    Notes
    -----
    - The outputs of the estimated models are used to update the simulation state in place.
    - Simultaneous calibration is performed if calibration data is provided in the configuration.
    - See module-level docstring for details on model outputs and caveats.
    """
    start_time = time.time()
    marriage_model = mm.get_step("marriage")
    divorce_model = mm.get_step("divorce")
    cohabitation_model = mm.get_step("cohabitation")

    # Marriage Model
    single_noncohab_index = ~persons["cohabitate"] & persons["is_not_married"]
    marriage_model_data = persons.to_frame(marriage_model.variable_names).loc[
        single_noncohab_index
    ]

    # Divorce model
    married_household_sizes = (
        persons.local.loc[(persons["MAR"] == 1) & persons["relate"].isin([0, 1])]
        .groupby("household_id")
        .size()
    )
    married_households_living_together = married_household_sizes[
        married_household_sizes == 2
    ].index.tolist()
    divorce_model_variables = columns_in_formula(divorce_model.model_expression)
    divorce_model_data = households.to_frame(divorce_model_variables).loc[
        married_households_living_together
    ]

    # Cohabitation to X Model
    ELIGIBLE_HOUSEHOLDS = (
        persons.local[(persons["relate"] == 13) & persons["is_not_married"]][
            "household_id"
        ]
        .unique()
        .astype(int)
    )

    cohabitation_model_data = households.to_frame(
        cohabitation_model.variable_names
    ).loc[ELIGIBLE_HOUSEHOLDS]

    # Load calibration config
    demos_config: DEMOSConfig = get_config()
    sim_cal_config: SimultaneousCalibrationConfig = (
        demos_config.hh_reorg_module_config.simultaneous_calibration_config
    )

    # Calibrate if necessary
    if sim_cal_config is not None:
        simultaneous_calibration(
            sim_cal_config,
            persons,
            marriage_model,
            cohabitation_model,
            divorce_model,
            marriage_model_data,
            cohabitation_model_data,
            divorce_model_data,
        )

    # Execute all the models
    marriage_list, divorce_list, cohabitate_x_list = run_models(
        marriage_model,
        cohabitation_model,
        divorce_model,
        marriage_model_data,
        cohabitation_model_data,
        divorce_model_data,
    )

    # Check number marriages before and after applying the model
    print_marital_count(persons.local)
    n_married_after, min_div, max_div = compute_expected_marital_status(
        persons.local, cohabitate_x_list, marriage_list, divorce_list
    )
    logger.debug("Predicted Marital status after applying models")
    logger.debug(f"Predicted MAR == 1: {n_married_after:,}")
    logger.debug(f"Predicted MAR == 3: [{min_div:,}, {max_div:,}]")

    ######### UPDATING
    logger.info("Restructuring households:")
    logger.info("Cohabitations..")
    update_cohabitating_households(
        persons, households, cohabitate_x_list, get_new_households
    )
    print_household_stats()

    logger.info("Marriages..")
    update_married_households_random(
        persons, households, marriage_list, get_new_households
    )
    print_household_stats()
    fix_erroneous_households(persons)
    print_household_stats()

    print("Divorces..")
    update_divorce(persons, households, divorce_list, get_new_households)
    print_household_stats()

    households.local = households.local.reindex(sorted(persons.household_id.unique()))
    print_marital_count(persons.local)
    log_execution_time(start_time, orca.get_injectable("year"), "household_reorg")


@orca.injectable(autocall=False)
def get_new_households(n):
    """
    Generate new unique household IDs for use in household reorganization.

    Ensures that new household IDs do not overlap with any existing or historical IDs
    in the persons, graveyard, or rebalanced_persons tables. Also expands the households table index as needed.

    Parameters
    ----------
    n : int
        Number of new household IDs to generate.

    Returns
    -------
    np.ndarray
        Array of new unique household IDs.

    Notes
    -----
    - This function is used internally by the household reorganization logic.
    - The method for adding empty rows to the households table may change in the future.
    """
    persons = orca.get_table("persons")
    graveyard = orca.get_table("graveyard")
    rebalanced_persons = orca.get_table("rebalanced_persons")

    current_max = pd.concat(
        [persons.local, graveyard.local, rebalanced_persons.local], ignore_index=True
    ).household_id.max()
    new_hh_ids = (
        np.arange(n)  # = [0, 1, 2 ...] up to the number of households
        + current_max  # = [max_hh_id, max_household_id + 1, ...]
        + 1
    )
    # TODO: Change how we add empty rows to the households table
    households = orca.get_table("households")
    households.local = households.local.reindex(set(households.index).union(new_hh_ids))
    return new_hh_ids


@orca.injectable()
def persons_grouped_household(persons):
    """
    Precompute a groupby object for persons by household_id.

    Parameters
    ----------
    persons : orca.Table
        The persons table.

    Returns
    -------
    pandas.core.groupby.DataFrameGroupBy
        GroupBy object for persons grouped by household_id.

    Notes
    -----
    - Used for efficient aggregation in household-level orca columns.
    """
    return persons.to_frame().groupby("household_id")


@orca.column(table_name="persons", cache=True, cache_scope="step")
def cohabitate(persons):
    """
    Binary column in persons table evaluating to True if person is cohabitating partner of household head or head in
    a household with a cohabitating partner
    """
    unmarried_partner_index = persons["relate"] == 13
    cohab_household_ids = persons["household_id"].loc[unmarried_partner_index].unique()
    return unmarried_partner_index | (
        (persons["relate"] == 0) & persons["household_id"].isin(cohab_household_ids)
    )


@orca.column(table_name="persons", cache=True, cache_scope="step")
def is_not_married(persons):
    """
    Binary column in persons table  evaluating to True if a person is above 15 and is not married (`MAR != 1`). Used in interal logic.
    """
    return (persons["MAR"] != 1) & (persons["age"] >= 15)


@orca.column(table_name="persons")
def is_head(persons):
    """
    Binary column in persons table evaluating to 1 if person is head of household
    """
    return (persons["relate"] == 0).astype(int)


@orca.column(table_name="persons")
def race_head(persons):
    """
    Combination of `is_head` and `race_id`
    """
    return persons["is_head"] * persons["race_id"]


@orca.column(table_name="persons")
def age_head(persons):
    """
    Evaluates to `is_head` * `age`
    """
    return persons["is_head"] * persons["age"]


@orca.column(table_name="persons")
def hispanic_head(persons):
    """
    Evaluates to `is_head` * `hispanic`
    """
    return persons["is_head"] * persons["hispanic"]


@orca.column(table_name="households")
def gt2(persons_grouped_household):
    """
    Binary column in households table evaluating to 1 if there are at least 2 people in the household
    """
    agg_df = persons_grouped_household.size()
    return (agg_df >= 2).astype(int)


@orca.column(table_name="households")
def hh_head_race_str(data="households.hh_head_race_id"):
    """
    Maps `households.hh_head_race_id` (numeric race_id of head) to a string: 'white', 'black', 'asian', or 'other'.
    """
    return data.map({1: "white", 2: "black", 6: "asian", 7: "asian"}).fillna("other")


@orca.column(table_name="households")
def hh_head_race_id(persons_grouped_household):
    """"""
    agg_df = persons_grouped_household.agg(race_of_head=("race_head", "sum"))
    return agg_df["race_of_head"]


@orca.column(table_name="households")
def hh_size_str(persons_grouped_household):
    agg_df = persons_grouped_household.size()
    return agg_df.map({1: "one", 2: "two", 3: "three"}).fillna("four or more")


# -----------------------------------------------------------------------------------------
# HOUSEHOLD AGGREGATE COLUMNS (moved from variables.py)
# -----------------------------------------------------------------------------------------

@orca.column("households")
def hh_n_children(households, persons):
    hh = households.local.copy()
    p = persons.local.copy()
    children = p[p["age"] <= 17].groupby("household_id").count()
    hh = hh.join(children[["age"]]).fillna(0)
    return hh["age"]


@orca.column("households")
def hh_income_bin1(households):
    df = households.to_frame(columns=["income"])
    return df.lt(25_000) * 1


@orca.column("households")
def hh_income_bin2(households):
    df = households.to_frame(columns=["income"])["income"]
    return df.between(25_000, 50_000, inclusive="left") * 1


@orca.column("households")
def hh_income_bin3(households):
    df = households.to_frame(columns=["income"])["income"]
    return df.between(50_000, 75_000, inclusive="left") * 1


@orca.column("households")
def hh_income_bin4(households):
    df = households.to_frame(columns=["income"])["income"]
    return df.between(75_000, 150_000, inclusive="left") * 1


@orca.column("households")
def hh_income_bin5(households):
    df = households.to_frame(columns=["income"])
    return df.ge(150_000) * 1


# Max education year of head or spouse (relate < 2)
@orca.column("households")
def hh_edu_top(persons):
    df = persons.to_frame(columns=["household_id", "edu", "relate"])
    df = df[df["relate"] < 2][["household_id", "edu"]]
    return df.groupby("household_id").agg({"edu": "max"})


# High school or equivalent (head/spouse max edu)
@orca.column("households")
def hh_edu_top_bin2(households):
    df = households.to_frame(columns=["hh_edu_top"])["hh_edu_top"]
    return df.between(16, 17, inclusive="both") * 1


# Some college or more (head/spouse max edu)
@orca.column("households")
def hh_edu_top_bin3(households):
    df = households.to_frame(columns=["hh_edu_top"])
    return df.gt(17) * 1


@orca.column("households")
def hh_fam_work(persons):
    df = persons.to_frame(columns=["relate", "worker", "household_id"])
    df = df[df["relate"] < 2]
    return df.groupby("household_id").agg({"worker": "sum"})


@orca.column("households")
def hh_fam_work2(households):
    df = households.to_frame(columns=["hh_fam_work"])
    return df.eq(2) * 1


# Average age of head and spouse
@orca.column("households")
def hh_age_avg(persons):
    df = persons.to_frame(columns=["relate", "age", "household_id"])
    df = df[df["relate"] < 2]
    return df.groupby("household_id").agg({"age": "mean"})


@orca.column("households")
def hh_age_avg_bin2(households):
    df = households.to_frame(columns=["hh_age_avg"])["hh_age_avg"]
    return df.between(22, 35, inclusive="right") * 1


@orca.column("households")
def hh_age_avg_bin3(households):
    df = households.to_frame(columns=["hh_age_avg"])["hh_age_avg"]
    return df.between(35, 60, inclusive="right") * 1


@orca.column("households")
def hh_age_avg_bin4(households):
    df = households.to_frame(columns=["hh_age_avg"])
    return df.gt(60) * 1


# Minimum age of head and spouse
@orca.column("households")
def hh_age_min(persons):
    df = persons.to_frame(columns=["relate", "age", "household_id"])
    df = df[df["relate"] < 2]
    return df.groupby("household_id").agg({"age": "min"})


@orca.column("households")
def hh_age_min_bin2(households):
    df = households.to_frame(columns=["hh_age_min"])["hh_age_min"]
    return df.between(22, 35, inclusive="right") * 1


@orca.column("households")
def hh_age_min_bin3(households):
    df = households.to_frame(columns=["hh_age_min"])["hh_age_min"]
    return df.between(35, 60, inclusive="right") * 1


@orca.column("households")
def hh_age_min_bin4(households):
    df = households.to_frame(columns=["hh_age_min"])
    return df.gt(60) * 1


# Race of household head (White, race_id == 1)
@orca.column("households")
def hh_head_race_white(persons):
    df = persons.to_frame(columns=["relate", "race_white", "household_id"])
    df = df[df["relate"] == 0]
    df.sort_values("household_id", inplace=True)
    return df.set_index("household_id")["race_white"]


def simultaneous_calibration(
    sim_cal_config,
    persons,
    marriage_model,
    cohab_model,
    divorce_model,
    marriage_data,
    cohab_data,
    divorce_data,
):
    """
    Perform simultaneous calibration of marriage, divorce, and cohabitation models.

    Adjusts model parameters so that the combined outputs of the three estimated models
    (marriage, divorce, cohabitation) match observed aggregate statistics for married and divorced persons.
    Uses a simple gradient-based optimization with optional momentum.

    Parameters
    ----------
    sim_cal_config : SimultaneousCalibrationConfig
        Configuration for the calibration procedure (learning rate, tolerance, etc).
    persons : orca.Table
        The persons table.
    marriage_model, cohab_model, divorce_model : EstimatedModel
        Fitted estimated model objects for each sub-model.
    marriage_data, cohab_data, divorce_data : pandas.DataFrame
        Data for each model to make predictions on.

    Returns
    -------
    None

    Notes
    -----
    - Only runs if calibration data is provided in the configuration.
    - Updates model parameters in place.
    - See module-level docstring for caveats.
    """
    marital_status_table = orca.get_table("marital_status_output")

    def compute_error(n_married, n_divorced, target_married, target_divorced):
        married_rmse = (n_married - target_married) ** 2
        divorce_rmse = (n_divorced - target_divorced) ** 2
        return np.sqrt((married_rmse + divorce_rmse) / 2)

    # Load observed data
    observed_marrital = orca.get_table("observed_marrital_data").to_frame()
    target_data = observed_marrital[
        observed_marrital.index == orca.get_injectable("year")
    ]
    target_married_count = target_data[(target_data["MAR"] == 1)]["count"].values[0]
    target_divorced_count = target_data[(target_data["MAR"] == 3)]["count"].values[0]
    married_weight = target_married_count / (
        target_married_count + target_divorced_count
    )
    divorce_weight = target_divorced_count / (
        target_married_count + target_divorced_count
    )

    # Execute all the models
    marriage_list, divorce_list, cohabitate_x_list = run_models(
        marriage_model,
        cohab_model,
        divorce_model,
        marriage_data,
        cohab_data,
        divorce_data,
    )

    n_married, min_div, max_div = compute_expected_marital_status(
        persons.local, cohabitate_x_list, marriage_list, divorce_list
    )
    n_divorced = (max_div - min_div) / 2 + min_div

    logger.debug("Predicted Marital status after applying models BEFORE CALIBRATION")
    logger.debug(f"Predicted MAR == 1: {n_married:,}")
    logger.debug(f"Predicted MAR == 3: [{min_div:,}, {max_div:,}]")

    marital_status_table.local = pd.concat(
        [
            marital_status_table.local,
            pd.DataFrame(
                [
                    [orca.get_injectable("year"), "married", "before", n_married],
                    [orca.get_injectable("year"), "divorced_min", "before", min_div],
                    [orca.get_injectable("year"), "divorced_max", "before", max_div],
                ],
                columns=["year", "metric", "time", "value"],
            ),
        ],
        axis=0,
    )

    # Initialize optimization algorithm
    married_gradient = 0
    divorce_gradient = 0
    cohabitation_gradient = 0
    momentum_weight = sim_cal_config.momentum_weight
    total_iterations = 0
    error = compute_error(
        n_married, n_divorced, target_married_count, target_divorced_count
    )
    while (
        error > sim_cal_config.tolerance and total_iterations < sim_cal_config.max_iter
    ):
        logger.info(
            f"Simultaneous Calibration: Iteration {total_iterations} error: {error}"
        )
        lr = (
            sim_cal_config.learning_rate
            * ((sim_cal_config.max_iter - total_iterations) + 0.5)
            / sim_cal_config.max_iter
        )

        # Calculate updates with momentum
        ## TODO: Cohabitation gradient could be weighted according to the contribution of cohabitation models to marriages
        divorce_gradient = lr * (
            momentum_weight * divorce_gradient
            + (1 - momentum_weight)
            * divorce_weight
            * np.log(target_divorced_count / n_divorced)
        )
        married_gradient = lr * (
            momentum_weight * married_gradient
            + (1 - momentum_weight)
            * married_weight
            * np.log(target_married_count / n_married)
        )
        cohabitation_gradient = lr * (
            momentum_weight * cohabitation_gradient
            + (1 - momentum_weight)
            * married_weight
            * np.log(target_married_count / n_married)
        )

        # Apply updates
        divorce_model.fitted_parameters[0] += divorce_gradient
        marriage_model.coeffs.loc[0, "married"] += married_gradient
        cohab_model.coeffs.loc[0, "marriage"] += cohabitation_gradient

        # Re-run models
        marriage_list, divorce_list, cohabitate_x_list = run_models(
            marriage_model,
            cohab_model,
            divorce_model,
            marriage_data,
            cohab_data,
            divorce_data,
        )

        n_married, min_div, max_div = compute_expected_marital_status(
            persons.local, cohabitate_x_list, marriage_list, divorce_list
        )
        n_divorced = (max_div - min_div) / 2 + min_div

        error = compute_error(
            n_married, n_divorced, target_married_count, target_divorced_count
        )
        total_iterations += 1

    marital_status_table.local = pd.concat(
        [
            marital_status_table.local,
            pd.DataFrame(
                [
                    [orca.get_injectable("year"), "married", "after", n_married],
                    [orca.get_injectable("year"), "divorced_min", "after", min_div],
                    [orca.get_injectable("year"), "divorced_max", "after", max_div],
                ],
                columns=["year", "metric", "time", "value"],
            ),
        ],
        axis=0,
    )

    logger.info(f"Final error after Simultaneous calibration: {error}")


def run_models(
    marriage_model, cohab_model, divorce_model, marriage_data, cohab_data, divorce_data
):
    """
    Run all three estimated models and return their outputs.

    Parameters
    ----------
    marriage_model, cohab_model, divorce_model : EstimatedModel
        Fitted estimated model objects for each sub-model.
    marriage_data, cohab_data, divorce_data : pandas.DataFrame
        Data for each model to make predictions on.

    Returns
    -------
    tuple
        (marriage_list, divorce_list, cohabitate_x_list):
        - marriage_list: pd.Series, 0 = stay single, 1 = cohabitate, 2 = get married
        - divorce_list: pd.Series, 0 = stay married, 1 = divorce
        - cohabitate_x_list: pd.Series, 0 = stay cohabitating, 1 = break up, 2 = get married
    """
    marriage_list = marriage_model.predict(marriage_data)
    divorce_list = divorce_model.predict(divorce_data).astype(int)
    cohabitate_x_list = cohab_model.predict(cohab_data)
    return marriage_list, divorce_list, cohabitate_x_list


def print_marital_count(persons_df):
    """
    Print the number of people with each marital status (MAR=1 or MAR=3).

    Parameters
    ----------
    persons_df : pandas.DataFrame
        DataFrame of persons, must include 'MAR' column.

    Returns
    -------
    None
    """
    for i in [1, 3]:
        logger.debug(f"Number of people with MAR={i}: {(persons_df.MAR == i).sum():,}")


def compute_expected_marital_status(
    persons_df, cohabitate_x_list, marriage_list, divorce_list
):
    """
    Compute the expected number of married and divorced persons after applying model outputs.

    Parameters
    ----------
    persons_df : pandas.DataFrame
        DataFrame of persons, must include 'MAR', 'relate', 'person_sex', and 'household_id'.
    cohabitate_x_list : pd.Series
        Output of the cohabitation model (see module docstring for values).
    marriage_list : pd.Series
        Output of the marriage model (see module docstring for values).
    divorce_list : pd.Series
        Output of the divorce model (see module docstring for values).

    Returns
    -------
    tuple
        (n_married_after, min_div, max_div):
        - n_married_after: int, expected number of married persons after updates
        - min_div: int, minimum expected number of divorced persons
        - max_div: int, maximum expected number of divorced persons

    Notes
    -----
    - Used for calibration and reporting.
    - Logic is based on current and predicted statuses.
    """
    starting_married = persons_df.MAR == 1
    starting_n_married = starting_married.sum()
    starting_divorced = persons_df.MAR == 3
    starting_n_divorced = starting_divorced.sum()
    starting_head_idx = persons_df.relate == 0
    starting_partner_idx = persons_df.relate == 13
    starting_spouse_idx = persons_df.relate == 1

    # Cohabitations computations
    cohabitate_marriages = cohabitate_x_list == 2
    cohabitate_marriage_people = persons_df.household_id.isin(
        cohabitate_marriages[cohabitate_marriages].index.to_list()
    )

    # New married households * 2 (head and spouse)
    cohabitation_model_new_married_people = (cohabitate_x_list == 2).sum() * 2
    cohabitation_model_less_divorced = len(
        persons_df.loc[
            cohabitate_marriage_people
            & (starting_head_idx | starting_partner_idx)
            & (starting_divorced)
        ]
    )

    # Marriage computations
    male_filter = persons_df.person_sex == "male"
    female_filter = persons_df.person_sex == "female"
    marriage_reindexed = marriage_list.reindex(persons_df.index).fillna(0) == 2

    male_wed = persons_df.loc[(marriage_reindexed) & (male_filter)]
    female_wed = persons_df.loc[(marriage_reindexed) & (female_filter)]
    n_wed = min([len(male_wed), len(female_wed)])

    n_male_div = (male_wed.MAR == 3).sum()
    min_male_div_to_married = n_male_div - min([(len(male_wed) - n_wed), n_male_div])
    max_male_div_to_married = n_male_div - max([0, n_male_div - n_wed])

    n_female_div = (female_wed.MAR == 3).sum()
    min_female_div_to_married = n_female_div - min(
        [(len(female_wed) - n_wed), n_female_div]
    )
    max_female_div_to_married = n_female_div - max([0, n_female_div - n_wed])

    max_div_to_married = max_male_div_to_married + max_female_div_to_married
    min_div_to_married = min_male_div_to_married + min_female_div_to_married

    # Divorce computation
    divorced_household_ids = divorce_list[divorce_list.astype(bool)].index
    person_in_divorced_household_index = persons_df["household_id"].isin(
        divorced_household_ids
    )
    head_and_spose_index = (
        (starting_head_idx | starting_spouse_idx)
        & starting_married
        & person_in_divorced_household_index
    )
    n_divorced = head_and_spose_index.sum()

    div_after_cohabitation_and_divorce = (
        starting_n_divorced - cohabitation_model_less_divorced + n_divorced
    )

    n_married_after = (
        starting_n_married
        + cohabitation_model_new_married_people
        + n_wed * 2
        - n_divorced
    )
    min_div = div_after_cohabitation_and_divorce - max_div_to_married
    max_div = div_after_cohabitation_and_divorce - min_div_to_married
    return n_married_after, min_div, max_div


@orca.step("print_household_stats")
def print_household_stats():
    logger.debug(
        f"Households size from persons table: {orca.get_table('persons').local['household_id'].unique().shape[0]}"
    )
    logger.debug(
        f"Households size from households table: {orca.get_table('households').local.index.unique().shape[0]}"
    )
    logger.debug(
        f"Persons Size: {orca.get_table('persons').local.index.unique().shape[0]}"
    )
    logger.debug(
        f"Missing hh: {len(set(orca.get_table('persons').local['household_id'].unique()) - set(orca.get_table('households').local.index.unique()))}"
    )
    persons_df = orca.get_table("persons").local
    persons_df["relate_0"] = np.where(persons_df["relate"] == 0, 1, 0)
    persons_df["relate_1"] = np.where(persons_df["relate"] == 1, 1, 0)
    persons_df["relate_13"] = np.where(persons_df["relate"] == 13, 1, 0)
    persons_df_sum = persons_df.groupby("household_id").agg(
        relate_1=("relate_1", "sum"),
        relate_13=("relate_13", "sum"),
        relate_0=("relate_0", "sum"),
    )
    logger.debug(
        f"Households with multiple 0: {((persons_df_sum['relate_0'])>1).sum()}"
    )
    logger.debug(
        f"Households with multiple 1: {((persons_df_sum['relate_1'])>1).sum()}"
    )
    logger.debug(
        f"Households with multiple 13: {((persons_df_sum['relate_13'])>1).sum()}"
    )
    logger.debug(
        f"Households with 1 and 13: {((persons_df_sum['relate_1'] * persons_df_sum['relate_13'])>0).sum()}"
    )


def update_cohabitating_households(
    persons, households, cohabitate_list, get_new_households
):
    """
    Updating households and persons after cohabitation model.

    Args:
        persons (DataFrameWrapper): DataFrameWrapper of persons table
        households (DataFrameWrapper): DataFrameWrapper of households table
        cohabitate_list (pd.Series): Pandas Series of cohabitation model output

    Returns:
        None
    """

    # Load calibration config
    demos_config: DEMOSConfig = get_config()
    module_config: HHReorgModuleConfig = demos_config.hh_reorg_module_config

    # Precompute some indices
    married_hh = cohabitate_list.index[cohabitate_list == 2].to_list()
    breakup_hh = cohabitate_list.index[cohabitate_list == 1].to_list()
    newly_married_persons_index = persons["household_id"].isin(married_hh)
    newly_brokeup_persons_index = persons["household_id"].isin(breakup_hh)
    unmarried_partner_index = persons["relate"] == 13
    head_index = persons["relate"] == 0

    # Perform update for people that got married
    persons.local.loc[
        newly_married_persons_index & unmarried_partner_index,
        "relate",
    ] = 1
    persons.local.loc[
        newly_married_persons_index & (head_index | unmarried_partner_index), "MAR"
    ] = 1

    # Perform update for people that broke up
    leaving_person_index = newly_brokeup_persons_index & unmarried_partner_index

    # Get the old household_id for the leaving person to retrieve the county_id
    old_household_id = persons.local.loc[leaving_person_index, "household_id"].values

    ## Person leaving is now head of household
    persons.local.loc[leaving_person_index, "relate"] = 0

    ### Assign new household_id to people leaving
    new_households = get_new_households(leaving_person_index.sum())
    persons.local.loc[leaving_person_index, "household_id"] = new_households

    # If geoid_col is set, we copy the geoid from old households to new ones
    if module_config.geoid_col is not None:
        geoid_assignment = households.local.loc[
            old_household_id, module_config.geoid_col
        ].values
        households.local.loc[new_households, module_config.geoid_col] = geoid_assignment
    county_assignment = households.local.loc[old_household_id, "lcm_county_id"].values
    households.local.loc[new_households, "lcm_county_id"] = county_assignment


def fix_erroneous_households(persons):
    n_partners_df = (
        persons.local[(persons["relate"] == 1) | (persons["relate"] == 13)]
        .groupby("household_id")["relate"]
        .nunique()
        .reset_index()
    )
    households_to_drop = n_partners_df[n_partners_df["relate"] == 2][
        "household_id"
    ].to_list()

    # Drop the households
    if len(households_to_drop) > 0:
        persons.local = persons.local[
            ~persons.local["household_id"].isin(households_to_drop)
        ]
