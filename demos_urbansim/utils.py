

def update_married_households_random(persons, households, marriage_list):
    """
    Update the marriage status of individuals and create new households
    Args:
        persons (DataFrameWrapper): DataFrameWrapper of the persons table
        households (DataFrameWrapper): DataFrameWrapper of the households table
        marriage_list (pd.Series): Pandas Series of the married individuals
    Returns:
        None
    """
    # print("Updating persons and households...")
    p_df = persons.local
    household_cols = households.local_columns
    household_df = households.local
    persons_cols = persons.local_columns
    persons_local_cols = persons.local_columns
    hh_df = households.to_frame(columns=["lcm_county_id"])
    hh_df.reset_index(inplace=True)
    # print("Indices duplicated:",p_df.index.duplicated().sum())
    p_df["new_mar"] = marriage_list
    p_df["new_mar"].fillna(0, inplace=True)
    relevant = p_df[p_df["new_mar"] > 0].copy()
    print("New marriages:", (relevant["new_mar"] ==1).sum())
    print("New cohabs:", (relevant["new_mar"] ==2).sum())
    # breakpoint()
    # Ensure an even number of people get married
    if relevant[relevant["new_mar"]==1].shape[0] % 2 != 0:
        sampled = p_df[p_df["new_mar"]==1].sample(1)
        sampled.new_mar = 0
        p_df.update(sampled)
        relevant = p_df[p_df["new_mar"] > 0].copy()

    if relevant[relevant["new_mar"]==2].shape[0] % 2 != 0:
        sampled = p_df[p_df["new_mar"]==2].sample(1)
        sampled.new_mar = 0
        p_df.update(sampled)
        relevant = p_df[p_df["new_mar"] > 0].copy()

    relevant.sort_values("new_mar", inplace=True)

    relevant = relevant.reset_index().merge(hh_df, on=["household_id"]).set_index("person_id")
    p_df = p_df.reset_index().merge(hh_df, on=["household_id"]).set_index("person_id")


    def swap(arr):
        result = np.empty_like(arr)
        result[::2] = arr[1::2]
        result[1::2] = arr[::2]
        return result

    def first(size):
        result = np.zeros(size)
        result[::2] = result[::2] + 1
        return result

    def relate(size, marriage=True):
        result = np.zeros(size)
        if marriage:
            result[1::2] = result[1::2] + 1
        else:
            result[1::2] = result[1::2] + 13
        return result
    
    min_mar = relevant[relevant["new_mar"] == 1]["person_sex"].value_counts().min()
    min_cohab = relevant[relevant["new_mar"] == 2]["person_sex"].value_counts().min()
    min_mar = int(min_mar)
    min_cohab = int(min_cohab)

    female_mar = relevant[
        (relevant["new_mar"] == 1) & (relevant["person_sex"] == "female")
    ].sample(min_mar)
    male_mar = relevant[
        (relevant["new_mar"] == 1) & (relevant["person_sex"] == "male")
    ].sample(min_mar)
    female_coh = relevant[
        (relevant["new_mar"] == 2) & (relevant["person_sex"] == "female")
    ].sample(min_cohab)
    male_coh = relevant[
        (relevant["new_mar"] == 2) & (relevant["person_sex"] == "male")
    ].sample(min_cohab)
    
    
    # print("Printing family sizes:")
    # print(female_mar.shape[0])
    # print(male_mar.shape[0])
    # print(female_coh.shape[0])
    # print(male_coh.shape[0])
    
    female_mar["number"] = np.arange(female_mar.shape[0])
    male_mar["number"] = np.arange(male_mar.shape[0])
    female_coh["number"] = np.arange(female_coh.shape[0])
    male_coh["number"] = np.arange(male_coh.shape[0])
    married = pd.concat([male_mar, female_mar])
    cohabitate = pd.concat([male_coh, female_coh])
    married = married.sort_values(by=["number"])
    cohabitate = cohabitate.sort_values(by=["number"])

    married["household_group"] = np.repeat(np.arange(len(married.index) / 2), 2)
    cohabitate["household_group"] = np.repeat(np.arange(len(cohabitate.index) / 2), 2)

    married = married.sort_values(
        by=["household_group", "earning"], ascending=[True, False]
    )
    cohabitate = cohabitate.sort_values(
        by=["household_group", "earning"], ascending=[True, False]
    )

    cohabitate["household_group"] = (
        cohabitate["household_group"] + married["household_group"].max() + 1
    )

    married["new_relate"] = relate(married.shape[0])
    cohabitate["new_relate"] = relate(cohabitate.shape[0], False)

    final = pd.concat([married, cohabitate])

    final["first"] = first(final.shape[0])
    final["partner"] = swap(final.index)
    final["partner_house"] = swap(final["household_id"])
    final["partner_relate"] = swap(final["relate"])

    final["new_household_id"] = -99
    final["stay"] = -99

    final = final[~(final["household_id"] == final["partner_house"])]

    print("Pair people.")
    # Pair up the people and classify what type of marriage it is
    # TODO speed up this code by a lot
    # relevant.sort_values("new_mar", inplace=True)
    
    
    
    # Stay documentation
    # 0 - leaves household
    # 1 - stays
    # 2 - this persons household becomes a root household (a root household absorbs the leaf household)
    # 4 - this persons household becomes a leaf household
    # 3 - this person leaves their household and creates a new household with partner
    # Marriage
    CONDITION_1 = (
        (final["first"] == 1) & (final["relate"] == 0) & (final["partner_relate"] == 0)
    )
    final.loc[final[CONDITION_1].index, "stay"] = 1
    final.loc[final[CONDITION_1]["partner"].values, "stay"] = 0

    CONDITION_2 = (
        (final["first"] == 1) & (final["relate"] == 0) & (final["partner_relate"] != 0)
    )
    final.loc[final[CONDITION_2].index, "stay"] = 1
    final.loc[final[CONDITION_2]["partner"].values, "stay"] = 0

    CONDITION_3 = (
        (final["first"] == 1) & (final["relate"] != 0) & (final["partner_relate"] == 0)
    )
    final.loc[final[CONDITION_3].index, "stay"] = 0
    final.loc[final[CONDITION_3]["partner"].values, "stay"] = 1

    CONDITION_4 = (
        (final["first"] == 1) & (final["relate"] != 0) & (final["partner_relate"] != 0)
    )
    final.loc[final[CONDITION_4].index, "stay"] = 3
    final.loc[final[CONDITION_4]["partner"].values, "stay"] = 3

    new_household_ids = np.arange(final[CONDITION_4].index.shape[0])
    new_household_ids_max = new_household_ids.max() + 1
    final.loc[final[CONDITION_4].index, "new_household_id"] = new_household_ids
    final.loc[
        final[CONDITION_4]["partner"].values, "new_household_id"
    ] = new_household_ids

    print('Finished Pairing')
    # print("Updating households and persons table")
    # print(final.household_id.unique().shape[0])
    current_max_id = max(orca.get_injectable("max_hh_id"), household_df.index.max())

    final["hh_new_id"] = np.where(
        final["stay"].isin([1]),
        final["household_id"],
        np.where(
            final["stay"].isin([0]),
            final["partner_house"],
            final["new_household_id"] + current_max_id + 1,
        ),
    )

    # final["new_relate"] = relate(final.shape[0])
    ## NEED TO SEPARATE MARRIED FROM COHABITATE

    # Households where everyone left
    # household_matched = (p_df[p_df["household_id"].isin(final["household_id"].unique())].groupby("household_id").size() == final.groupby("household_id").size())
    # removed_hh_values = household_matched[household_matched==True].index.values

    # Households where head left
    household_ids_reorganized = final[(final["stay"] == 0) & (final["relate"] == 0)][
        "household_id"
    ].unique()

    p_df.loc[final.index, "household_id"] = final["hh_new_id"]
    p_df.loc[final.index, "relate"] = final["new_relate"]
    # print("HH SHAPE 1:", p_df["household_id"].unique().shape[0])

    households_restructuring = p_df.loc[
        p_df["household_id"].isin(household_ids_reorganized)
    ]

    households_restructuring = households_restructuring.sort_values(
        by=["household_id", "earning"], ascending=False
    )
    households_restructuring.loc[
        households_restructuring.groupby(["household_id"]).head(1).index, "relate"
    ] = 0

    household_df = household_df.loc[household_df.index.isin(p_df["household_id"])]

    # print("HH SHAPE 1:", p_df["household_id"].unique().shape[0])

    # leaf_hh = final.loc[final["stay"]==4, ["household_id", "partner_house"]]["household_id"].to_list()
    # root_hh = final.loc[final["stay"]==2, ["household_id", "partner_house"]]["household_id"].to_list()
    # new_hh = final.loc[final["stay"]==3, "hh_new_id"].to_list()

    # household_mapping_dict = {leaf_hh[i]: root_hh[i] for i in range(len(root_hh))}

    # household_df = household_df.reset_index()

    # class MyDict(dict):
    #     def __missing__(self, key):
    #         return key

    # recodes = MyDict(household_mapping_dict)

    # household_df["household_id"] = household_df["household_id"].map(recodes)
    # p_df["household_id"] = p_df["household_id"].map(recodes)

    p_df = p_df.sort_values("relate")

    p_df["person"] = 1
    p_df["is_head"] = np.where(p_df["relate"] == 0, 1, 0)
    p_df["race_head"] = p_df["is_head"] * p_df["race_id"]
    p_df["age_head"] = p_df["is_head"] * p_df["age"]
    p_df["hispanic_head"] = p_df["is_head"] * p_df["hispanic"]
    p_df["child"] = np.where(p_df["relate"].isin([2, 3, 4, 14]), 1, 0)
    p_df["senior"] = np.where(p_df["age"] >= 65, 1, 0)
    p_df["age_gt55"] = np.where(p_df["age"] >= 55, 1, 0)

    p_df = p_df.sort_values(by=["household_id", "relate"])
    household_agg = p_df.groupby("household_id").agg(
        income=("earning", "sum"),
        race_of_head=("race_id", "first"),
        age_of_head=("age", "first"),
        size=("person", "sum"),
        workers=("worker", "sum"),
        hispanic_head=("hispanic_head", "sum"),
        # lcm_county_id=("lcm_county_id", "first"),
        persons_age_gt55=("age_gt55", "sum"),
        seniors=("senior", "sum"),
        children=("child", "sum"),
        persons=("person", "sum"),
    )

    # household_agg["lcm_county_id"] = household_agg["lcm_county_id"]
    household_agg["gt55"] = np.where(household_agg["persons_age_gt55"] > 0, 1, 0)
    household_agg["gt2"] = np.where(household_agg["persons"] > 2, 1, 0)
    # household_agg["sf_detached"] = "unknown"
    # household_agg["serialno"] = "unknown"
    # household_agg["cars"] = np.random.ran
    # dom_integers(0, 2, size=household_agg.shape[0])
    household_agg["hh_workers"] = np.where(
        household_agg["workers"] == 0,
        "none",
        np.where(household_agg["workers"] == 1, "one", "two or more"),
    )
    household_agg["hh_age_of_head"] = np.where(
        household_agg["age_of_head"] < 35,
        "lt35",
        np.where(household_agg["age_of_head"] < 65, "gt35-lt65", "gt65"),
    )
    household_agg["hh_race_of_head"] = np.where(
        household_agg["race_of_head"] == 1,
        "white",
        np.where(
            household_agg["race_of_head"] == 2,
            "black",
            np.where(household_agg["race_of_head"].isin([6, 7]), "asian", "other"),
        ),
    )
    household_agg["hispanic_head"] = np.where(
        household_agg["hispanic_head"] == 1, "yes", "no"
    )
    household_agg["hh_size"] = np.where(
        household_agg["size"] == 1,
        "one",
        np.where(
            household_agg["size"] == 2,
            "two",
            np.where(household_agg["size"] == 3, "three", "four or more"),
        ),
    )
    household_agg["hh_children"] = np.where(household_agg["children"] >= 1, "yes", "no")
    household_agg["hh_seniors"] = np.where(household_agg["seniors"] >= 1, "yes", "no")
    household_agg["hh_income"] = np.where(
        household_agg["income"] < 30000,
        "lt30",
        np.where(
            household_agg["income"] < 60,
            "gt30-lt60",
            np.where(
                household_agg["income"] < 100,
                "gt60-lt100",
                np.where(household_agg["income"] < 150, "gt100-lt150", "gt150"),
            ),
        ),
    )

    # agg_households = household_df.groupby("household_id").agg(serialno = ("serialno", "first"), # change to min once you change the serial number for all
    #                                         cars = ("cars", "sum"),
    #                                         # income = ("income", "sum"),
    #                                         # workers = ("workers", "sum"),
    #                                         tenure = ("tenure", "first"),
    #                                         recent_mover = ("recent_mover", "first"),
    #                                         sf_detached = ("sf_detached", "first"),
    #                                         lcm_county_id = ("lcm_county_id", "first"),
    #                                         block_id=("block_id", "first")) # we need hhtype here

    # agg_households["hh_cars"] = np.where(agg_households["cars"] == 0, "none",
    #                                         np.where(agg_households["cars"] == 1, "one", "two or more"))

    # household_df = household_df.drop_duplicates(subset="household_id")

    # household_df = household_df.set_index("household_id")
    # household_df.update(agg_households)
    household_df.update(household_agg)

    final["MAR"] = np.where(final["new_mar"] == 1, 1, final["MAR"])
    p_df.update(final["MAR"])

    # print("HH SHAPE 2:", p_df["household_id"].unique().shape[0])

    new_hh = household_agg.loc[
        ~household_agg.index.isin(household_df.index.unique())
    ].copy()
    new_hh["serialno"] = "-1"
    new_hh["cars"] = np.random.choice([0, 1, 2], size=new_hh.shape[0])
    new_hh["hispanic_status_of_head"] = "-1"
    new_hh["tenure"] = "-1"
    new_hh["recent_mover"] = "-1"
    new_hh["sf_detached"] = "-1"
    new_hh["hh_cars"] = np.where(
        new_hh["cars"] == 0, "none", np.where(new_hh["cars"] == 1, "one", "two or more")
    )
    new_hh["tenure_mover"] = "-1"
    new_hh["block_id"] = "-1"
    new_hh["hh_type"] = -1
    household_df = pd.concat([household_df, new_hh])

    # p_df.update(relevant["household_id"])

    #
    # household_df = household_df.set_index("household_id")
    # new_households = household_agg.loc[household_agg.index.isin(new_hh)].copy()
    # new_households["serialno"] = "-1"
    # new_households["cars"] = np.random.choice([0, 1, 2], size=new_households.shape[0])
    # new_households["hispanic_status_of_head"] = -1
    # new_households["tenure"] = -1
    # new_households["recent_mover"] = "-1"
    # new_households["sf_detached"] = "-1"
    # new_households["hh_cars"] = np.where(new_households["cars"] == 0, "none",
    #                                      np.where(new_households["cars"] == 1, "one", "two or more"))
    # new_households["tenure_mover"] = "-1"
    # new_households["block_id"] = "-1"
    # new_households["hh_type"] = -1
    # household_df = pd.concat([household_df, new_households])

    # print('Time to run marriage', sp.duration)
    orca.add_table("households", household_df[household_cols])
    orca.add_table("persons", p_df[persons_cols])
    orca.add_injectable(
        "max_hh_id", max(orca.get_injectable("max_hh_id"), household_df.index.max())
    )

    # print("households size", household_df.shape[0])
    
    metadata = orca.get_table("metadata").to_frame()
    max_hh_id = metadata.loc["max_hh_id", "value"]
    max_p_id = metadata.loc["max_p_id", "value"]
    if household_df.index.max() > max_hh_id:
        metadata.loc["max_hh_id", "value"] = household_df.index.max()
    if p_df.index.max() > max_p_id:
        metadata.loc["max_p_id", "value"] = p_df.index.max()
    orca.add_table("metadata", metadata)
    
    married_table = orca.get_table("marriage_table").to_frame()
    if married_table.empty:
        married_table = pd.DataFrame(
            [[(marriage_list == 1).sum(), (marriage_list == 2).sum()]],
            columns=["married", "cohabitated"],
        )
    else:
        married_table = married_table.append(
            {
                "married": (marriage_list == 1).sum(),
                "cohabitated": (marriage_list == 2).sum(),
            },
            ignore_index=True,
        )
    orca.add_table("marriage_table", married_table)



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
    student_population = orca.get_table("student_population").to_frame()
    if student_population.empty:
        student_population = pd.DataFrame(
            data={"year": [year], "count": [students.shape[0]]}
        )
    else:
        student_population_new = pd.DataFrame(
            data={"year": [year], "count": [students.shape[0]]}
        )
        students = pd.concat([student_population, student_population_new])
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
    orca.add_table("student_population", student_population)




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
    highest_index = max(orca.get_injectable("max_p_id"), highest_index)

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
    babies["work_at_home"] = 0
    babies["worker"] = 0
    babies["person_age"] = "19 and under"
    babies["person_sex"] = babies["sex"].map({1: "male", 2: "female"})
    babies["child"] = 1
    babies["senior"] = 0
    babies["dead"] = -99
    babies["person"] = 1

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
    orca.add_table("persons", combined_result[persons_df.columns])
    orca.add_table("households", households_df[households_df.columns])
    metadata = orca.get_table("metadata").to_frame()
    max_hh_id = metadata.loc["max_hh_id", "value"]
    max_p_id = metadata.loc["max_p_id", "value"]
    if households_df.index.max() > max_hh_id:
        metadata.loc["max_hh_id", "value"] = households_df.index.max()
    if combined_result.index.max() > max_p_id:
        metadata.loc["max_p_id", "value"] = combined_result.index.max()
    
    orca.add_table("metadata", metadata)
    # orca.add_injectable("max_p_id", max(highest_index, orca.get_injectable("max_p_id")))




def update_households_after_kids(persons, households, kids_moving):
    """
    Add and update households after kids move out.

    Args:
        persons (DataFrameWrapper): DataFrameWrapper of persons table
        households (DataFrameWrapper): DataFrameWrapper of households table
        kids_moving (pd.Series): Pandas Series of kids moving out of household

    Returns:
        None
    """
    # print("Updating households...")
    persons_df = orca.get_table("persons").local

    persons_local_cols = orca.get_table("persons").local_columns

    households_df = orca.get_table("households").local
    households_local_cols = orca.get_table("households").local_columns
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
    current_max_household_id = max(orca.get_injectable("max_hh_id"), highest_index)

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
    orca.add_injectable(
        "max_hh_id", max(households_df.index.max(), orca.get_injectable("max_hh_id"))
    )

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
        kids_moving_table = kids_moving_table.append(
            {"kids_moving_out": kids_moving_table.sum()}, ignore_index=True
        )
    orca.add_table("kids_move_table", kids_moving_table)


def update_married_households(persons, households, marriage_list):
    """
    Update the marriage status of individuals and create new households

    Args:
        persons (DataFrameWrapper): DataFrameWrapper of the persons table
        households (DataFrameWrapper): DataFrameWrapper of the households table
        marriage_list (pd.Series): Pandas Series of the married individuals

    Returns:
        None
    """
    # print("Updating persons and households...")
    p_df = persons.local
    household_cols = households.local_columns
    household_df = households.local
    persons_cols = persons.local_columns
    persons_local_cols = persons.local_columns
    hh_df = households.to_frame(columns=["lcm_county_id"])
    hh_df.reset_index(inplace=True)
    p_df["new_mar"] = marriage_list
    p_df["new_mar"].fillna(0, inplace=True)
    relevant = p_df[p_df["new_mar"] > 0].copy()
    print("New marriages:", (relevant["new_mar"] ==1).sum())
    print("New cohabs:", (relevant["new_mar"] ==2).sum())
    # Ensure an even number of people get married
    if relevant[relevant["new_mar"] == 1].shape[0] % 2 != 0:
        sampled = p_df[p_df["new_mar"] == 1].sample(1)
        sampled.new_mar = 0
        p_df.update(sampled)
        relevant = p_df[p_df["new_mar"] > 0].copy()

    if relevant[relevant["new_mar"] == 2].shape[0] % 2 != 0:
        sampled = p_df[p_df["new_mar"] == 2].sample(1)
        sampled.new_mar = 0
        p_df.update(sampled)
        relevant = p_df[p_df["new_mar"] > 0].copy()

    relevant.sort_values("new_mar", inplace=True)

    relevant = (
        relevant.reset_index().merge(hh_df, on=["household_id"]).set_index("person_id")
    )
    p_df = p_df.reset_index().merge(hh_df, on=["household_id"]).set_index("person_id")

    print("Pair people.")
    min_mar = relevant[relevant["new_mar"] == 1]["person_sex"].value_counts().min()
    min_cohab = relevant[relevant["new_mar"] == 2]["person_sex"].value_counts().min()

    female_mar = relevant[
        (relevant["new_mar"] == 1) & (relevant["person_sex"] == "female")
    ].sample(min_mar)
    male_mar = relevant[
        (relevant["new_mar"] == 1) & (relevant["person_sex"] == "male")
    ].sample(min_mar)
    female_coh = relevant[
        (relevant["new_mar"] == 2) & (relevant["person_sex"] == "female")
    ].sample(min_cohab)
    male_coh = relevant[
        (relevant["new_mar"] == 2) & (relevant["person_sex"] == "male")
    ].sample(min_cohab)

    def brute_force_matching(male, female):
        """Function to run brute force marriage matching.

        TODO: Improve the matchmaking process
        TODO: Account for same-sex marriages

        Args:
            male (DataFrame): DataFrame of the male side
            female (DataFrame): DataFrame of the female side

        Returns:
            DataFrame: DataFrame of newly formed households
        """
        ordered_households = pd.DataFrame()
        male_mar = male.sample(frac=1)
        female_mar = female.sample(frac=1)
        for index in np.arange(female_mar.shape[0]):
            dist = cdist(
                female_mar.iloc[index][["age", "earning"]]
                .to_numpy()
                .reshape((1, 2))
                .astype(float),
                male_mar[["age", "earning"]].to_numpy(),
                "euclidean",
            )
            arg = dist.argmin()
            household_new = pd.DataFrame([female_mar.iloc[index], male_mar.iloc[arg]])
            # household_new["household_group"] = index + 1
            # household_new["new_household_id"] = -99
            # household_new["stay"] = -99
            male_mar = male_mar.drop(male_mar.iloc[arg].name)
            ordered_households = pd.concat([ordered_households, household_new])

        return ordered_households

    cohabitate = brute_force_matching(male_coh, female_coh)
    cohabitate.index.name = "person_id"

    married = brute_force_matching(male_mar, female_mar)
    married.index.name = "person_id"

    def relate(size, marriage=True):
        result = np.zeros(size)
        if marriage:
            result[1::2] = result[1::2] + 1
        else:
            result[1::2] = result[1::2] + 13
        return result

    def swap(arr):
        result = np.empty_like(arr)
        result[::2] = arr[1::2]
        result[1::2] = arr[::2]
        return result

    def first(size):
        result = np.zeros(size)
        result[::2] = result[::2] + 1
        return result

    married["household_group"] = np.repeat(np.arange(len(married.index) / 2), 2)
    cohabitate["household_group"] = np.repeat(np.arange(len(cohabitate.index) / 2), 2)

    married = married.sort_values(
        by=["household_group", "earning"], ascending=[True, False]
    )
    cohabitate = cohabitate.sort_values(
        by=["household_group", "earning"], ascending=[True, False]
    )

    cohabitate["household_group"] = (
        cohabitate["household_group"] + married["household_group"].max() + 1
    )

    married["new_relate"] = relate(married.shape[0])
    cohabitate["new_relate"] = relate(cohabitate.shape[0], False)

    final = pd.concat([married, cohabitate])

    final["first"] = first(final.shape[0])
    final["partner"] = swap(final.index)
    final["partner_house"] = swap(final["household_id"])
    final["partner_relate"] = swap(final["relate"])

    final["new_household_id"] = -99
    final["stay"] = -99

    final = final[~(final["household_id"] == final["partner_house"])]

    # Pair up the people and classify what type of marriage it is
    # TODO speed up this code by a lot
    # relevant.sort_values("new_mar", inplace=True)
    # married = final[final["new_mar"]==1].copy()
    # cohabitation = final[final["new_mar"]==2].copy()
    # Stay documentation
    # 0 - leaves household
    # 1 - stays
    # 2 - this persons household becomes a root household (a root household absorbs the leaf household)
    # 4 - this persons household becomes a leaf household
    # 3 - this person leaves their household and creates a new household with partner
    # Marriage
    CONDITION_1 = (
        (final["first"] == 1) & (final["relate"] == 0) & (final["partner_relate"] == 0)
    )
    final.loc[final[CONDITION_1].index, "stay"] = 1
    final.loc[final[CONDITION_1]["partner"].values, "stay"] = 0

    CONDITION_2 = (
        (final["first"] == 1) & (final["relate"] == 0) & (final["partner_relate"] != 0)
    )
    final.loc[final[CONDITION_2].index, "stay"] = 1
    final.loc[final[CONDITION_2]["partner"].values, "stay"] = 0

    CONDITION_3 = (
        (final["first"] == 1) & (final["relate"] != 0) & (final["partner_relate"] == 0)
    )
    final.loc[final[CONDITION_3].index, "stay"] = 0
    final.loc[final[CONDITION_3]["partner"].values, "stay"] = 1

    CONDITION_4 = (
        (final["first"] == 1) & (final["relate"] != 0) & (final["partner_relate"] != 0)
    )
    final.loc[final[CONDITION_4].index, "stay"] = 3
    final.loc[final[CONDITION_4]["partner"].values, "stay"] = 3

    new_household_ids = np.arange(final[CONDITION_4].index.shape[0])
    new_household_ids_max = new_household_ids.max() + 1
    final.loc[final[CONDITION_4].index, "new_household_id"] = new_household_ids
    final.loc[
        final[CONDITION_4]["partner"].values, "new_household_id"
    ] = new_household_ids

    print("Finished Pairing")
    # print("Updating households and persons table")
    # print(final.household_id.unique().shape[0])
    current_max_id = max(orca.get_injectable("max_hh_id"), household_df.index.max())

    final["hh_new_id"] = np.where(
        final["stay"].isin([1]),
        final["household_id"],
        np.where(
            final["stay"].isin([0]),
            final["partner_house"],
            final["new_household_id"] + current_max_id + 1,
        ),
    )

    # final["new_relate"] = relate(final.shape[0])
    ## NEED TO SEPARATE MARRIED FROM COHABITATE

    # Households where everyone left
    # household_matched = (p_df[p_df["household_id"].isin(final["household_id"].unique())].groupby("household_id").size() == final.groupby("household_id").size())
    # removed_hh_values = household_matched[household_matched==True].index.values

    # Households where head left
    household_ids_reorganized = final[(final["stay"] == 0) & (final["relate"] == 0)][
        "household_id"
    ].unique()

    p_df.loc[final.index, "household_id"] = final["hh_new_id"]
    p_df.loc[final.index, "relate"] = final["new_relate"]
    # print("HH SHAPE 1:", p_df["household_id"].unique().shape[0])

    households_restructuring = p_df.loc[
        p_df["household_id"].isin(household_ids_reorganized)
    ]

    households_restructuring = households_restructuring.sort_values(
        by=["household_id", "earning"], ascending=False
    )
    households_restructuring.loc[
        households_restructuring.groupby(["household_id"]).head(1).index, "relate"
    ] = 0

    household_df = household_df.loc[household_df.index.isin(p_df["household_id"])]

    # print("HH SHAPE 1:", p_df["household_id"].unique().shape[0])

    # leaf_hh = final.loc[final["stay"]==4, ["household_id", "partner_house"]]["household_id"].to_list()
    # root_hh = final.loc[final["stay"]==2, ["household_id", "partner_house"]]["household_id"].to_list()
    # new_hh = final.loc[final["stay"]==3, "hh_new_id"].to_list()

    # household_mapping_dict = {leaf_hh[i]: root_hh[i] for i in range(len(root_hh))}

    # household_df = household_df.reset_index()

    # class MyDict(dict):
    #     def __missing__(self, key):
    #         return key

    # recodes = MyDict(household_mapping_dict)

    # household_df["household_id"] = household_df["household_id"].map(recodes)
    # p_df["household_id"] = p_df["household_id"].map(recodes)

    p_df = p_df.sort_values("relate")

    p_df["person"] = 1
    p_df["is_head"] = np.where(p_df["relate"] == 0, 1, 0)
    p_df["race_head"] = p_df["is_head"] * p_df["race_id"]
    p_df["age_head"] = p_df["is_head"] * p_df["age"]
    p_df["hispanic_head"] = p_df["is_head"] * p_df["hispanic"]
    p_df["child"] = np.where(p_df["relate"].isin([2, 3, 4, 14]), 1, 0)
    p_df["senior"] = np.where(p_df["age"] >= 65, 1, 0)
    p_df["age_gt55"] = np.where(p_df["age"] >= 55, 1, 0)

    p_df = p_df.sort_values(by=["household_id", "relate"])
    household_agg = p_df.groupby("household_id").agg(
        income=("earning", "sum"),
        race_of_head=("race_id", "first"),
        age_of_head=("age", "first"),
        size=("person", "sum"),
        workers=("worker", "sum"),
        hispanic_head=("hispanic_head", "sum"),
        # lcm_county_id=("lcm_county_id", "first"),
        persons_age_gt55=("age_gt55", "sum"),
        seniors=("senior", "sum"),
        children=("child", "sum"),
        persons=("person", "sum"),
    )

    # household_agg["lcm_county_id"] = household_agg["lcm_county_id"]
    household_agg["gt55"] = np.where(household_agg["persons_age_gt55"] > 0, 1, 0)
    household_agg["gt2"] = np.where(household_agg["persons"] > 2, 1, 0)
    # household_agg["sf_detached"] = "unknown"
    # household_agg["serialno"] = "unknown"
    # household_agg["cars"] = np.random.ran
    # dom_integers(0, 2, size=household_agg.shape[0])
    household_agg["hh_workers"] = np.where(
        household_agg["workers"] == 0,
        "none",
        np.where(household_agg["workers"] == 1, "one", "two or more"),
    )
    household_agg["hh_age_of_head"] = np.where(
        household_agg["age_of_head"] < 35,
        "lt35",
        np.where(household_agg["age_of_head"] < 65, "gt35-lt65", "gt65"),
    )
    household_agg["hh_race_of_head"] = np.where(
        household_agg["race_of_head"] == 1,
        "white",
        np.where(
            household_agg["race_of_head"] == 2,
            "black",
            np.where(household_agg["race_of_head"].isin([6, 7]), "asian", "other"),
        ),
    )
    household_agg["hispanic_head"] = np.where(
        household_agg["hispanic_head"] == 1, "yes", "no"
    )
    household_agg["hh_size"] = np.where(
        household_agg["size"] == 1,
        "one",
        np.where(
            household_agg["size"] == 2,
            "two",
            np.where(household_agg["size"] == 3, "three", "four or more"),
        ),
    )
    household_agg["hh_children"] = np.where(household_agg["children"] >= 1, "yes", "no")
    household_agg["hh_seniors"] = np.where(household_agg["seniors"] >= 1, "yes", "no")
    household_agg["hh_income"] = np.where(
        household_agg["income"] < 30000,
        "lt30",
        np.where(
            household_agg["income"] < 60,
            "gt30-lt60",
            np.where(
                household_agg["income"] < 100,
                "gt60-lt100",
                np.where(household_agg["income"] < 150, "gt100-lt150", "gt150"),
            ),
        ),
    )

    # agg_households = household_df.groupby("household_id").agg(serialno = ("serialno", "first"), # change to min once you change the serial number for all
    #                                         cars = ("cars", "sum"),
    #                                         # income = ("income", "sum"),
    #                                         # workers = ("workers", "sum"),
    #                                         tenure = ("tenure", "first"),
    #                                         recent_mover = ("recent_mover", "first"),
    #                                         sf_detached = ("sf_detached", "first"),
    #                                         lcm_county_id = ("lcm_county_id", "first"),
    #                                         block_id=("block_id", "first")) # we need hhtype here

    # agg_households["hh_cars"] = np.where(agg_households["cars"] == 0, "none",
    #                                         np.where(agg_households["cars"] == 1, "one", "two or more"))

    # household_df = household_df.drop_duplicates(subset="household_id")

    # household_df = household_df.set_index("household_id")
    # household_df.update(agg_households)
    household_df.update(household_agg)

    final["MAR"] = np.where(final["new_mar"] == 1, 1, final["MAR"])
    p_df.update(final["MAR"])

    # print("HH SHAPE 2:", p_df["household_id"].unique().shape[0])

    new_hh = household_agg.loc[
        ~household_agg.index.isin(household_df.index.unique())
    ].copy()
    new_hh["serialno"] = "-1"
    new_hh["cars"] = np.random.choice([0, 1, 2], size=new_hh.shape[0])
    new_hh["hispanic_status_of_head"] = "-1"
    new_hh["tenure"] = "-1"
    new_hh["recent_mover"] = "-1"
    new_hh["sf_detached"] = "-1"
    new_hh["hh_cars"] = np.where(
        new_hh["cars"] == 0, "none", np.where(new_hh["cars"] == 1, "one", "two or more")
    )
    new_hh["tenure_mover"] = "-1"
    new_hh["block_id"] = "-1"
    new_hh["hh_type"] = -1
    household_df = pd.concat([household_df, new_hh])

    # p_df.update(relevant["household_id"])

    #
    # household_df = household_df.set_index("household_id")
    # new_households = household_agg.loc[household_agg.index.isin(new_hh)].copy()
    # new_households["serialno"] = "-1"
    # new_households["cars"] = np.random.choice([0, 1, 2], size=new_households.shape[0])
    # new_households["hispanic_status_of_head"] = -1
    # new_households["tenure"] = -1
    # new_households["recent_mover"] = "-1"
    # new_households["sf_detached"] = "-1"
    # new_households["hh_cars"] = np.where(new_households["cars"] == 0, "none",
    #                                      np.where(new_households["cars"] == 1, "one", "two or more"))
    # new_households["tenure_mover"] = "-1"
    # new_households["block_id"] = "-1"
    # new_households["hh_type"] = -1
    # household_df = pd.concat([household_df, new_households])

    # print('Time to run marriage', sp.duration)
    orca.add_table("households", household_df[household_cols])
    orca.add_table("persons", p_df[persons_cols])
    orca.add_injectable(
        "max_hh_id", max(orca.get_injectable("max_hh_id"), household_df.index.max())
    )

    # print("households size", household_df.shape[0])

    metadata = orca.get_table("metadata").to_frame()
    max_hh_id = metadata.loc["max_hh_id", "value"]
    max_p_id = metadata.loc["max_p_id", "value"]
    if household_df.index.max() > max_hh_id:
        metadata.loc["max_hh_id", "value"] = household_df.index.max()
    if p_df.index.max() > max_p_id:
        metadata.loc["max_p_id", "value"] = p_df.index.max()
    orca.add_table("metadata", metadata)

    married_table = orca.get_table("marriage_table").to_frame()
    if married_table.empty:
        married_table = pd.DataFrame(
            [[(marriage_list == 1).sum(), (marriage_list == 2).sum()]],
            columns=["married", "cohabitated"],
        )
    else:
        married_table = married_table.append(
            {
                "married": (marriage_list == 1).sum(),
                "cohabitated": (marriage_list == 2).sum(),
            },
            ignore_index=True,
        )
    orca.add_table("marriage_table", married_table)



def update_cohabitating_households(persons, households, cohabitate_list):
    """
    Updating households and persons after cohabitation model.

    Args:
        persons (DataFrameWrapper): DataFrameWrapper of persons table
        households (DataFrameWrapper): DataFrameWrapper of households table
        cohabitate_list (pd.Series): Pandas Series of cohabitation model output

    Returns:
        None
    """
    # print("Updating persons and households...")
    persons_df = persons.local
    persons_local_cols = persons.local_columns
    households_df = households.local
    hh_df = households.to_frame(columns=["lcm_county_id"])
    households_local_cols = households.local_columns
    married_hh = cohabitate_list.index[cohabitate_list == 1].to_list()
    breakup_hh = cohabitate_list.index[cohabitate_list == 2].to_list()

    persons_df.loc[
        (persons_df["household_id"].isin(married_hh)) & (persons_df["relate"] == 13),
        "relate",
    ] = 1
    persons_df.loc[
        (persons_df["household_id"].isin(married_hh))
        & (persons_df["relate"].isin([1, 0])),
        "MAR",
    ] = 1

    persons_df = (
        persons_df.reset_index()
        .merge(hh_df, on=["household_id"])
        .set_index("person_id")
    )

    leaving_person_index = persons_df.index[
        (persons_df["household_id"].isin(breakup_hh)) & (persons_df["relate"] == 13)
    ]

    leaving_house = persons_df.loc[leaving_person_index].copy()

    leaving_house["relate"] = 0

    persons_df = persons_df.drop(leaving_person_index)

    # Update characteristics for households staying
    persons_df["person"] = 1
    persons_df["is_head"] = np.where(persons_df["relate"] == 0, 1, 0)
    persons_df["race_head"] = persons_df["is_head"] * persons_df["race_id"]
    persons_df["age_head"] = persons_df["is_head"] * persons_df["age"]
    persons_df["hispanic_head"] = persons_df["is_head"] * persons_df["hispanic"]
    persons_df["child"] = np.where(persons_df["relate"].isin([2, 3, 4, 14]), 1, 0)
    persons_df["senior"] = np.where(persons_df["age"] >= 65, 1, 0)
    persons_df["age_gt55"] = np.where(persons_df["age"] >= 55, 1, 0)

    households_new = persons_df.groupby("household_id").agg(
        income=("earning", "sum"),
        race_of_head=("race_head", "sum"),
        age_of_head=("age_head", "sum"),
        workers=("worker", "sum"),
        hispanic_status_of_head=("hispanic", "sum"),
        persons=("person", "sum"),
        children=("child", "sum"),
        seniors=("senior", "sum"),
        gt55=("age_gt55", "sum"),
    )

    households_new["hh_age_of_head"] = np.where(
        households_new["age_of_head"] < 35,
        "lt35",
        np.where(households_new["age_of_head"] < 65, "gt35-lt65", "gt65"),
    )
    households_new["hispanic_head"] = np.where(
        households_new["hispanic_status_of_head"] == 1, "yes", "no"
    )
    households_new["hh_children"] = np.where(
        households_new["children"] >= 1, "yes", "no"
    )
    households_new["hh_seniors"] = np.where(households_new["seniors"] >= 1, "yes", "no")
    households_new["gt2"] = np.where(households_new["persons"] >= 2, 1, 0)
    households_new["gt55"] = np.where(households_new["gt55"] >= 1, 1, 0)
    households_new["hh_income"] = np.where(
        households_new["income"] < 30000,
        "lt30",
        np.where(
            households_new["income"] < 60,
            "gt30-lt60",
            np.where(
                households_new["income"] < 100,
                "gt60-lt100",
                np.where(households_new["income"] < 150, "gt100-lt150", "gt150"),
            ),
        ),
    )
    households_new["hh_workers"] = np.where(
        households_new["workers"] == 0,
        "none",
        np.where(households_new["workers"] == 1, "one", "two or more"),
    )

    households_new["hh_race_of_head"] = np.where(
        households_new["race_of_head"] == 1,
        "white",
        np.where(
            households_new["race_of_head"] == 2,
            "black",
            np.where(households_new["race_of_head"].isin([6, 7]), "asian", "other"),
        ),
    )

    households_new["hh_size"] = np.where(
        households_new["persons"] == 1,
        "one",
        np.where(
            households_new["persons"] == 2,
            "two",
            np.where(households_new["persons"] == 3, "three", "four or more"),
        ),
    )

    households_df.update(households_new)

    # Create household characteristics for new households formed
    leaving_house["household_id"] = (
        np.arange(len(breakup_hh))
        + max(orca.get_injectable("max_hh_id"), households_df.index.max())
        + 1
    )
    leaving_house["person"] = 1
    leaving_house["is_head"] = np.where(leaving_house["relate"] == 0, 1, 0)
    leaving_house["race_head"] = leaving_house["is_head"] * leaving_house["race_id"]
    leaving_house["age_head"] = leaving_house["is_head"] * leaving_house["age"]
    leaving_house["hispanic_head"] = (
        leaving_house["is_head"] * leaving_house["hispanic"]
    )
    leaving_house["child"] = np.where(leaving_house["relate"].isin([2, 3, 4, 14]), 1, 0)
    leaving_house["senior"] = np.where(leaving_house["age"] >= 65, 1, 0)
    leaving_house["age_gt55"] = np.where(leaving_house["age"] >= 55, 1, 0)

    households_new = leaving_house.groupby("household_id").agg(
        income=("earning", "sum"),
        race_of_head=("race_head", "sum"),
        age_of_head=("age_head", "sum"),
        workers=("worker", "sum"),
        hispanic_status_of_head=("hispanic", "sum"),
        persons=("person", "sum"),
        children=("child", "sum"),
        seniors=("senior", "sum"),
        gt55=("age_gt55", "sum"),
        lcm_county_id=("lcm_county_id", "first"),
    )

    households_new["hh_age_of_head"] = np.where(
        households_new["age_of_head"] < 35,
        "lt35",
        np.where(households_new["age_of_head"] < 65, "gt35-lt65", "gt65"),
    )
    households_new["hispanic_head"] = np.where(
        households_new["hispanic_status_of_head"] == 1, "yes", "no"
    )
    households_new["hh_children"] = np.where(
        households_new["children"] >= 1, "yes", "no"
    )
    households_new["hh_seniors"] = np.where(households_new["seniors"] >= 1, "yes", "no")
    households_new["gt2"] = np.where(households_new["persons"] >= 2, 1, 0)
    households_new["gt55"] = np.where(households_new["gt55"] >= 1, 1, 0)
    households_new["hh_income"] = np.where(
        households_new["income"] < 30000,
        "lt30",
        np.where(
            households_new["income"] < 60,
            "gt30-lt60",
            np.where(
                households_new["income"] < 100,
                "gt60-lt100",
                np.where(households_new["income"] < 150, "gt100-lt150", "gt150"),
            ),
        ),
    )
    households_new["hh_workers"] = np.where(
        households_new["workers"] == 0,
        "none",
        np.where(households_new["workers"] == 1, "one", "two or more"),
    )

    households_new["hh_race_of_head"] = np.where(
        households_new["race_of_head"] == 1,
        "white",
        np.where(
            households_new["race_of_head"] == 2,
            "black",
            np.where(households_new["race_of_head"].isin([6, 7]), "asian", "other"),
        ),
    )

    households_new["hh_size"] = np.where(
        households_new["persons"] == 1,
        "one",
        np.where(
            households_new["persons"] == 2,
            "two",
            np.where(households_new["persons"] == 3, "three", "four or more"),
        ),
    )

    households_new["cars"] = np.random.choice([0, 1], size=households_new.shape[0])
    households_new["hh_cars"] = np.where(
        households_new["cars"] == 0,
        "none",
        np.where(households_new["cars"] == 1, "one", "two or more"),
    )
    households_new["tenure"] = "unknown"
    households_new["recent_mover"] = "unknown"
    households_new["sf_detached"] = "unknown"
    households_new["tenure_mover"] = "unknown"
    households_new["block_id"] = "-1"
    households_new["hh_type"] = "-1"
    households_df = pd.concat([households_df, households_new])

    persons_df = pd.concat([persons_df, leaving_house])

    # add to orca
    orca.add_table("households", households_df[households_local_cols])
    orca.add_table("persons", persons_df[persons_local_cols])
    orca.add_injectable(
        "max_hh_id", max(orca.get_injectable("max_hh_id"), households_df.index.max())
    )
    metadata = orca.get_table("metadata").to_frame()
    max_hh_id = metadata.loc["max_hh_id", "value"]
    max_p_id = metadata.loc["max_p_id", "value"]
    if households_df.index.max() > max_hh_id:
        metadata.loc["max_hh_id", "value"] = households_df.index.max()
    if persons_df.index.max() > max_p_id:
        metadata.loc["max_p_id", "value"] = persons_df.index.max()
    orca.add_table("metadata", metadata)



def update_divorce(persons, households, divorce_list):
    """
    Updating stats for divorced households

    Args:
        persons (DataFrameWrapper): DataFrameWrapper of the persons table
        households (DataFrameWrapper): DataFrameWrapper of the households table
        divorce_list (pd.Series): pandas Series of the divorced households

    Returns:
        None
    """
    # print("Updating household stats...")
    households_local_cols = households.local_columns

    persons_local_cols = persons.local_columns

    households_df = households.local

    persons_df = persons.local

    households_df["divorced"] = divorce_list

    t0 = time.time()
    divorce_households = households_df[households_df["divorced"] == 1].copy()
    t1 = time.time()
    total = t1 - t0
    print("Copy divroce households:", total)
    t0 = time.time()
    persons_divorce = persons_df[
        persons_df["household_id"].isin(divorce_households.index)
    ].copy()

    divorced_parents = persons_divorce[
        (persons_divorce["relate"].isin([0, 1])) & (persons_divorce["MAR"] == 1)
    ].copy()

    leaving_house = divorced_parents.groupby("household_id").sample(n=1)

    staying_house = persons_divorce[
        ~persons_divorce.index.isin(leaving_house.index)
    ].copy()
    t1 = time.time()
    total = t1 - t0
    print("Household segmentation:", total)
    t0 = time.time()
    # give the people leaving a new household id, update their marriage status, and other variables
    leaving_house["relate"] = 0
    leaving_house["MAR"] = 3
    leaving_house["member_id"] = 1
    leaving_house["household_id"] = (
        np.arange(leaving_house.shape[0]) + households_df.index.max() + 1
    )
    t1 = time.time()
    total = t1 - t0
    print("Leaving households variables", total)
    t0 = time.time()
    # modify necessary variables for members staying in household
    staying_house["relate"] = np.where(
        staying_house["relate"].isin([1, 0]), 0, staying_house["relate"]
    )
    staying_house["member_id"] = np.where(
        staying_house["member_id"] != 1,
        staying_house["member_id"] - 1,
        staying_house["relate"],
    )
    staying_house["MAR"] = np.where(
        staying_house["MAR"] == 1, 3, staying_house["relate"]
    )

    # initiate new households with individuals leaving house
    # TODO: DISCUSS ALL THESE INITIALIZATION MEASURES
    staying_households = staying_house.copy()
    staying_households["person"] = 1
    staying_households["is_head"] = np.where(staying_households["relate"] == 0, 1, 0)
    staying_households["race_head"] = (
        staying_households["is_head"] * staying_households["race_id"]
    )
    staying_households["age_head"] = (
        staying_households["is_head"] * staying_households["age"]
    )
    staying_households["hispanic_head"] = (
        staying_households["is_head"] * staying_households["hispanic"]
    )
    staying_households["child"] = np.where(
        staying_households["relate"].isin([2, 3, 4, 14]), 1, 0
    )
    staying_households["senior"] = np.where(staying_households["age"] >= 65, 1, 0)
    staying_households["age_gt55"] = np.where(staying_households["age"] >= 55, 1, 0)

    staying_households = staying_households.sort_values(by=["household_id", "relate"])
    staying_household_agg = staying_households.groupby("household_id").agg(
        income=("earning", "sum"),
        race_of_head=("race_id", "first"),
        age_of_head=("age", "first"),
        size=("person", "sum"),
        workers=("worker", "sum"),
        hispanic_head=("hispanic_head", "sum"),
        persons_age_gt55=("age_gt55", "sum"),
        seniors=("senior", "sum"),
        children=("child", "sum"),
        persons=("person", "sum"),
    )
    
    t1 = time.time()
    total = t1-t0
    print("Staying Household aggregation:", total)
    t0 = time.time()
    
    # household_agg["lcm_county_id"] = household_agg["lcm_county_id"]
    staying_household_agg["gt55"] = np.where(
        staying_household_agg["persons_age_gt55"] > 0, 1, 0
    )
    staying_household_agg["gt2"] = np.where(staying_household_agg["persons"] > 2, 1, 0)
    # staying_household_agg["sf_detached"] = "unknown"
    # staying_household_agg["serialno"] = "unknown"
    # staying_household_agg["cars"] = households_df[households_df.index.isin(staying_house["household_id"].unique())]["cars"]

    staying_household_agg["hh_workers"] = np.where(
        staying_household_agg["workers"] == 0,
        "none",
        np.where(staying_household_agg["workers"] == 1, "one", "two or more"),
    )
    staying_household_agg["hh_age_of_head"] = np.where(
        staying_household_agg["age_of_head"] < 35,
        "lt35",
        np.where(staying_household_agg["age_of_head"] < 65, "gt35-lt65", "gt65"),
    )
    staying_household_agg["hh_race_of_head"] = np.where(
        staying_household_agg["race_of_head"] == 1,
        "white",
        np.where(
            staying_household_agg["race_of_head"] == 2,
            "black",
            np.where(
                staying_household_agg["race_of_head"].isin([6, 7]), "asian", "other"
            ),
        ),
    )
    staying_household_agg["hispanic_head"] = np.where(
        staying_household_agg["hispanic_head"] == 1, "yes", "no"
    )
    staying_household_agg["hh_size"] = np.where(
        staying_household_agg["size"] == 1,
        "one",
        np.where(
            staying_household_agg["size"] == 2,
            "two",
            np.where(staying_household_agg["size"] == 3, "three", "four or more"),
        ),
    )
    staying_household_agg["hh_children"] = np.where(
        staying_household_agg["children"] >= 1, "yes", "no"
    )
    staying_household_agg["hh_seniors"] = np.where(
        staying_household_agg["seniors"] >= 1, "yes", "no"
    )
    staying_household_agg["hh_income"] = np.where(
        staying_household_agg["income"] < 30000,
        "lt30",
        np.where(
            staying_household_agg["income"] < 60,
            "gt30-lt60",
            np.where(
                staying_household_agg["income"] < 100,
                "gt60-lt100",
                np.where(staying_household_agg["income"] < 150, "gt100-lt150", "gt150"),
            ),
        ),
    )
    t1 = time.time()
    total = t1-t0
    print("Staying Household variables:", total)
    # staying_household_agg["hh_type"] = 1
    # staying_household_agg["household_type"] = 1
    # staying_household_agg["serialno"] = -1
    # staying_household_agg["birth"] = -99
    # staying_household_agg["divorced"] = -99
    # staying_household_agg.set_index(staying_household_agg["household_id"], inplace=True)
    staying_household_agg.index.name = "household_id"

    # initiate new households with individuals leaving house
    # TODO: DISCUSS ALL THESE INITIALIZATION MEASURES
    t0 = time.time()
    new_households = leaving_house.copy()
    new_households["person"] = 1
    new_households["is_head"] = np.where(new_households["relate"] == 0, 1, 0)
    new_households["race_head"] = new_households["is_head"] * new_households["race_id"]
    new_households["age_head"] = new_households["is_head"] * new_households["age"]
    new_households["hispanic_head"] = (
        new_households["is_head"] * new_households["hispanic"]
    )
    new_households["child"] = np.where(
        new_households["relate"].isin([2, 3, 4, 14]), 1, 0
    )
    new_households["senior"] = np.where(new_households["age"] >= 65, 1, 0)
    new_households["age_gt55"] = np.where(new_households["age"] >= 55, 1, 0)
    t1 = time.time()
    total = t1-t0
    print("create newhouseholds:", total)

    new_households = new_households.sort_values(by=["household_id", "relate"])
    t0 = time.time()
    household_agg = new_households.groupby("household_id").agg(
        income=("earning", "sum"),
        race_of_head=("race_head", "sum"),
        age_of_head=("age_head", "sum"),
        size=("person", "sum"),
        workers=("worker", "sum"),
        hispanic_head=("hispanic_head", "sum"),
        persons_age_gt55=("age_gt55", "sum"),
        seniors=("senior", "sum"),
        children=("child", "sum"),
        persons=("person", "sum"),
    )
    t1 = time.time()
    total = t1-t0
    print("Household aggregation:", total)

    t0 = time.time()
    # household_agg["lcm_county_id"] = household_agg["lcm_county_id"]
    household_agg["gt55"] = np.where(household_agg["persons_age_gt55"] > 0, 1, 0)
    household_agg["gt2"] = np.where(household_agg["persons"] > 2, 1, 0)
    household_agg["sf_detached"] = "unknown"
    household_agg["serialno"] = "unknown"
    household_agg["tenure"] = "unknown"
    household_agg["tenure_mover"] = "unknown"
    household_agg["recent_mover"] = "unknown"
    household_agg["cars"] = np.random.choice([0, 1], size=household_agg.shape[0])

    household_agg["hh_workers"] = np.where(
        household_agg["workers"] == 0,
        "none",
        np.where(household_agg["workers"] == 1, "one", "two or more"),
    )
    household_agg["hh_age_of_head"] = np.where(
        household_agg["age_of_head"] < 35,
        "lt35",
        np.where(household_agg["age_of_head"] < 65, "gt35-lt65", "gt65"),
    )
    household_agg["hh_race_of_head"] = np.where(
        household_agg["race_of_head"] == 1,
        "white",
        np.where(
            household_agg["race_of_head"] == 2,
            "black",
            np.where(household_agg["race_of_head"].isin([6, 7]), "asian", "other"),
        ),
    )
    household_agg["hispanic_head"] = np.where(
        household_agg["hispanic_head"] == 1, "yes", "no"
    )
    household_agg["hispanic_status_of_head"] = np.where(
        household_agg["hispanic_head"] == "yes", 1, 0
    )
    household_agg["hh_size"] = np.where(
        household_agg["size"] == 1,
        "one",
        np.where(
            household_agg["size"] == 2,
            "two",
            np.where(household_agg["size"] == 3, "three", "four or more"),
        ),
    )
    household_agg["hh_children"] = np.where(household_agg["children"] >= 1, "yes", "no")
    household_agg["hh_seniors"] = np.where(household_agg["seniors"] >= 1, "yes", "no")
    household_agg["hh_income"] = np.where(
        household_agg["income"] < 30000,
        "lt30",
        np.where(
            household_agg["income"] < 60,
            "gt30-lt60",
            np.where(
                household_agg["income"] < 100,
                "gt60-lt100",
                np.where(household_agg["income"] < 150, "gt100-lt150", "gt150"),
            ),
        ),
    )

    household_agg["hh_cars"] = np.where(
        household_agg["cars"] == 0,
        "none",
        np.where(household_agg["cars"] == 1, "one", "two or more"),
    )
    household_agg["block_id"] = "-1"
    household_agg["lcm_county_id"] = "-1"
    household_agg["hh_type"] = 1
    household_agg["household_type"] = 1
    household_agg["serialno"] = "-1"
    household_agg["birth"] = -99
    household_agg["divorced"] = -99
    # household_agg.set_index(household_agg["household_id"], inplace=True)
    # household_agg.index.name = "household_id"

    t1 = time.time()
    total = t1-t0
    print("Household aggregation variables:", total)
    
    households_df.update(staying_household_agg)

    t0 = time.time()
    # merge all in one persons and households table
    new_households = pd.concat(
        [households_df[households_local_cols], household_agg[households_local_cols]]
    )
    persons_df.update(staying_house[persons_local_cols])
    persons_df.update(leaving_house[persons_local_cols])
    t1 = time.time()
    total = t1-t0
    print("Updating dataframes:", total)
    # add to orca
    t0 = time.time()
    orca.add_table("households", new_households[households_local_cols])
    orca.add_table("persons", persons_df[persons_local_cols])
    orca.add_injectable(
        "max_hh_id", max(orca.get_injectable("max_hh_id"), new_households.index.max())
    )
    
    metadata = orca.get_table("metadata").to_frame()
    max_hh_id = metadata.loc["max_hh_id", "value"]
    max_p_id = metadata.loc["max_p_id", "value"]
    if new_households.index.max() > max_hh_id:
        metadata.loc["max_hh_id", "value"] = new_households.index.max()
    if persons_df.index.max() > max_p_id:
        metadata.loc["max_p_id", "value"] = persons_df.index.max()
    orca.add_table("metadata", metadata)

    print("Updating divorce metrics...")
    divorce_table = orca.get_table("divorce_table").to_frame()
    if divorce_table.empty:
        divorce_table = pd.DataFrame([divorce_list.sum()], columns=["divorced"])
    else:
        divorce_table = divorce_table.append(
            {"divorced": divorce_list.sum()}, ignore_index=True
        )
    orca.add_table("divorce_table", divorce_table)
    t1 = time.time()
    total = t1-t0
    print("Adding to orca:", total)