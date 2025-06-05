import orca
import numpy as np
from templates import estimated_models, modelmanager as mm
import time
from datasources import log_execution_time

@orca.step("education_model")
def education_model(persons, year):
    """
    Run the education model and update the persons table

    Args:
        persons (DataFrameWrapper): DataFrameWrapper of the persons table

    Returns:
        None
    """
    start_time = time.time()
    # Add temporary variable
    persons_df = persons.local
    persons_df["stop"] = -99
    orca.add_table("persons", persons_df)

    # Run the education model
    # print("Running the education model...")
    edu_model = mm.get_step("education")
    edu_model.run()
    student_list = edu_model.choices.astype(int)

    # Update student status
    # print("Updating student status...")
    update_education_status(persons, student_list, year)
    log_execution_time(start_time, orca.get_injectable("year"), "education")

def update_education_status(persons, student_list, year):
    """
    Function to update the student status in persons table based
    on the

    Args:
        persons (DataFrameWrapper): DataFrameWrapper of the persons table
        student_list (pd.Series): Pandas Series containing the output of
        the education model

    Returns:
        None
    """
    # Pull Data
    persons_df = persons.to_frame(
        columns=["age", "household_id", "edu", "student", "stop"]
    )
    persons_df["stop"] = student_list
    persons_df["stop"].fillna(2, inplace=True)

    # Update education level for individuals staying in school
    weights = persons_df["edu"].value_counts(normalize=True)

    persons_df.loc[persons_df["age"] == 3, "edu"] = 2
    persons_df.loc[persons_df["age"].isin([4, 5]), "edu"] = 4

    dropping_out = persons_df.loc[persons_df["stop"] == 1].copy()
    staying_school = persons_df.loc[persons_df["stop"] == 0].copy()

    dropping_out.loc[:, "student"] = 0
    staying_school.loc[:, "student"] = 1

    # high school and high school graduates proportions
    hs_p = persons_df[persons_df["edu"].isin([15, 16])]["edu"].value_counts(
        normalize=True
    )
    hs_grad_p = persons_df[persons_df["edu"].isin([16, 17])]["edu"].value_counts(
        normalize=True
    )
    # Students all the way to grade 10
    staying_school.loc[:, "edu"] = np.where(
        staying_school["edu"].between(4, 13, inclusive="both"),
        staying_school["edu"] + 1,
        staying_school["edu"],
    )
    # Students in grade 11 move to either 15 or 16 based on weights
    staying_school.loc[:, "edu"] = np.where(
        staying_school["edu"] == 14,
        np.random.choice([15, 16], p=[hs_p[15], hs_p[16]]),
        staying_school["edu"],
    )
    # Students in grade 12 either get hs degree or GED
    staying_school.loc[:, "edu"] = np.where(
        staying_school["edu"] == 15,
        np.random.choice([16, 17], p=[hs_grad_p[16], hs_grad_p[17]]),
        staying_school["edu"],
    )
    # Students with GED or HS Degree move to college
    staying_school.loc[:, "edu"] = np.where(
        staying_school["edu"].isin([16, 17]), 18, staying_school["edu"]
    )
    # Students with one year of college move to the next
    staying_school.loc[:, "edu"] = np.where(
        staying_school["edu"] == 18, 19, staying_school["edu"]
    )
    # Others to be added here.

    # Update education levels
    persons_df.update(staying_school)
    persons_df.update(dropping_out)

    orca.get_table("persons").update_col("edu", persons_df["edu"])
    orca.get_table("persons").update_col("student", persons_df["student"])

    # compute mean age of students
    # print("Updating students metrics...")
    students = persons_df[persons_df["student"] == 1]
    edu_over_time = orca.get_table("edu_over_time").to_frame()
    # student_population = orca.get_table("student_population").to_frame()
    # if student_population.empty:
    #     student_population = pd.DataFrame(
    #         data={"year": [year], "count": [students.shape[0]]}
    #     )
    # else:
    #     student_population_new = pd.DataFrame(
    #         data={"year": [year], "count": [students.shape[0]]}
    #     )
    #     students = pd.concat([student_population, student_population_new])
    # if edu_over_time.empty:
    #     edu_over_time = pd.DataFrame(
    #         data={"year": [year], "mean_age_of_students": [students["age"].mean()]}
    #     )
    # else:
    #     edu_over_time = edu_over_time.append(
    #         pd.DataFrame(
    #             {"year": [year], "mean_age_of_students": [students["age"].mean()]}
    #         ),
    #         ignore_index=True,
    #     )

    # orca.add_table("edu_over_time", edu_over_time)
    # orca.add_table("student_population", student_population)
