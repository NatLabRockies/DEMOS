import orca
import numpy as np
import pandas as pd
from templates import estimated_models, modelmanager as mm
import time
from logging_logic import log_execution_time
from templates.utils.models import columns_in_formula

STEP_NAME = "education"
REQUIRED_COLUMNS = [
    "persons.edu",
    "persons.student",
]


@orca.step(STEP_NAME)
def education(
    persons, edu_highschool_proportion, edu_highschool_grads_proportion, year
):
    """
    Simulate educational attainment and student status transitions.

    This step applies the education model to eligible persons (age > 15 and currently students)
    to determine who drops out. It advances students through grades and degrees, maintains
    proportions of high school and GED graduates, and updates the persons table in place.

    Parameters
    ----------
    persons : orca.Table
        The persons table containing individual-level attributes.
    edu_highschool_proportion : pandas.Series
        Proportion of students in 11th and 12th grade.
    edu_highschool_grads_proportion : pandas.Series
        Proportion of students with GED or high school diploma.
    year : int
        The current simulation year.

    Returns
    -------
    None

    Notes
    -----
    - Modifies `persons.edu` and `persons.student` in place.
    - Only persons older than 15 and currently students are considered for dropout modeling.
    - Proportions for transitions (e.g., GED vs. diploma) are maintained using observed data.
    - Some transitions use random assignment based on empirical proportions.
    """
    start_time = time.time()

    # Run education model
    model = mm.get_step("education")
    model_variables = columns_in_formula(model.model_expression)
    model_filters = (persons.age > 15) & (persons.student == 1)
    model_data = persons.to_frame(model_variables)[model_filters]
    stop_student_list = model.predict(model_data).astype(int)

    reindexed_stop_student = stop_student_list.reindex(persons.local.index).fillna(-99)

    # Update education years
    ## Kids
    persons.local.loc[persons["age"] == 3, "edu"] = 2
    persons.local.loc[persons["age"].isin([4, 5]), "edu"] = 4

    ## Dropping out
    persons.local.loc[reindexed_stop_student == 1, "student"] = 0

    ## Update those that stayed in school
    stayed_index = reindexed_stop_student == 0

    ### Between 4 and 13, increase by one - Students go all the way to grade 10
    tenth_grade_or_below_index = persons["edu"].between(4, 13, inclusive="both")
    persons.local.loc[stayed_index & tenth_grade_or_below_index, "edu"] += 1

    # NOTE: We perform the following operations in reverse order to avoid skipping years
    ### Students with one year of college move to the next
    college_index = persons["edu"] == 18
    persons.local.loc[stayed_index & college_index, "edu"] = 19

    ### Students with GED or HS Degree move to college
    ged_or_hs_index = persons["edu"].isin([16, 17])
    persons.local.loc[stayed_index & ged_or_hs_index, "edu"] = 18

    ### Students in grade 12 move to either 16 or 17 based on weights
    ### Proportion of no diploma to GED students is roughly maintained
    twelveth_grade_index = persons["edu"] == 15
    twelveth_grade_transition = np.random.choice(
        [16, 17],
        size=(stayed_index & twelveth_grade_index).sum(),
        p=[edu_highschool_grads_proportion[16], edu_highschool_grads_proportion[17]],
    )
    persons.local.loc[stayed_index & twelveth_grade_index, "edu"] = (
        twelveth_grade_transition
    )

    ### Students in grade 11 move to either 15 or 16 based on weights
    ### Proportion of 12th grade students to diploma highschool students is roughly maintained
    eleventh_grade_index = persons["edu"] == 14
    eleventh_grade_transition = np.random.choice(
        [15, 16],
        size=(stayed_index & eleventh_grade_index).sum(),
        p=[edu_highschool_proportion[15], edu_highschool_proportion[16]],
    )
    persons.local.loc[stayed_index & eleventh_grade_index, "edu"] = (
        eleventh_grade_transition
    )

    log_execution_time(start_time, orca.get_injectable("year"), "education")


@orca.injectable(name="edu_highschool_proportion", cache_scope="forever", cache=True)
def edu_highschool_proportion(data="persons.edu"):
    """
    Calculate the proportion of students in 11th and 12th grade.

    Parameters
    ----------
    data : pandas.Series
        The `edu` column from the persons table.

    Returns
    -------
    pandas.Series
        Proportion of students in 11th (15) and 12th (16) grade.
    """
    return data[data.isin([15, 16])].value_counts(normalize=True)


@orca.injectable(name="edu_highschool_grads_proportion")
def edu_highschool_grads_proportion(data="persons.edu"):
    """
    Calculate the proportion of students with GED or high school diploma.

    Parameters
    ----------
    data : pandas.Series
        The `edu` column from the persons table.

    Returns
    -------
    pandas.Series
        Proportion of students with GED (16) or high school diploma (17).
    """
    return data[data.isin([16, 17])].value_counts(normalize=True)


@orca.column(table_name="persons")
def education_group(data="persons.edu"):
    """
    Assign each person to an education group.

    Categorizes persons into predefined education intervals for use in modeling and reporting.

    Parameters
    ----------
    data : pandas.Series
        The `edu` column from the persons table.

    Returns
    -------
    pandas.Series
        Categorical education group labels as strings.
    """
    education_intervals = [0, 18, 22, 200]
    education_labels = ["lte17", "18-21", "gte22"]
    return pd.cut(
        data, bins=education_intervals, labels=education_labels, include_lowest=True
    ).astype(str)
