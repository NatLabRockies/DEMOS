import orca
import numpy as np
from templates import estimated_models, modelmanager as mm

@orca.step("education_model")
def education_model(persons,
                    edu_highschool_proportion,
                    edu_highschool_grads_proportion,
                    year):
    """
    Run the education model and update the persons table

        Modifies State Variables:
        - persons.edu
        - persons.student

    Args:
        persons (DataFrameWrapper): DataFrameWrapper of the persons table

    Returns:
        None
    """
    # Run education model
    ## Add temporary variable
    persons_df = persons.local
    persons_df["stop"] = -99
    orca.add_table("persons", persons_df)

    stop_student_list = run_education_model()
    reindexed_stop_student = stop_student_list.reindex(persons.local.index).fillna(-99)

    # Update education years
    ## Kids
    persons.local.loc[persons["age"] == 3, "edu"] = 2
    persons.local.loc[persons["age"].isin([4, 5]), "edu"] = 4

    ## Dropping out
    persons.local.loc[reindexed_stop_student == 1, "student"] = 0
    # TODO: Check if this line is really necessary
    # persons.local.loc[reindexed_stop_student == 0, "student"] = 1

    ## Update those that stayed in school
    stayed_index = reindexed_stop_student == 0

    ### Between 4 and 13, increase by one - Students go all the way to grade 10
    tenth_grade_or_below_index = persons["edu"].between(4, 13, inclusive="both")
    persons.local.loc[stayed_index & tenth_grade_or_below_index, "edu"] += 1

    ### Students in grade 11 move to either 15 or 16 based on weights
    ### Proportion of 12th grade students to diploma highschool students is roughly maintained
    eleventh_grade_index = persons["edu"] == 14
    eleventh_grade_transition = np.random.choice([15, 16],
                                                 size=(stayed_index & eleventh_grade_index).sum(),
                                                 p=[edu_highschool_proportion[15],
                                                    edu_highschool_proportion[16]])
    persons.local.loc[stayed_index & eleventh_grade_index, "edu"] = eleventh_grade_transition

    ### Students in grade 12 move to either 15 or 16 based on weights
    ### Proportion of no diploma to GED students is roughly maintained
    twelveth_grade_index = persons["edu"] == 15
    twelveth_grade_transition = np.random.choice([16, 17],
                                                 size=(stayed_index & twelveth_grade_index).sum(),
                                                 p=[edu_highschool_proportion[16],
                                                    edu_highschool_proportion[17]])
    persons.local.loc[stayed_index & twelveth_grade_index, "edu"] = twelveth_grade_transition

    ### Students with GED or HS Degree move to college
    ged_or_hs_index = persons["edu"].isin([16, 17])
    persons.local.loc[stayed_index & ged_or_hs_index, "edu"] = 18

    ### Students with one year of college move to the next
    college_index = persons["edu"] == 18
    persons.local.loc[stayed_index & college_index, "edu"] = 19


def run_education_model():
    edu_model = mm.get_step("education")
    edu_model.run()
    return edu_model.choices.astype(int)


@orca.injectable(name="edu_highschool_proportion")
def edu_highschool_proportion(data="persons.edu"):
    return data[data.isin([15, 16])].value_counts(normalize=True)


@orca.injectable(name="edu_highschool_grads_proportion")
def edu_highschool_grads_proportion(data="persons.edu"):
    return data[data.isin([16, 17])].value_counts(normalize=True)
