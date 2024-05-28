import os
from collections import OrderedDict

import mode_choice
import numpy as np
import openmatrix as omx
import orca
import pandas as pd
import yaml
from urbansim.utils import misc

print('importing variables for region', orca.get_injectable('region_code'))


# -----------------------------------------------------------------------------------------
# WORK LOCATION CHOICE VARIABLES
# -----------------------------------------------------------------------------------------
@orca.column('zones')
def mean_income(households):
    h = households.to_frame(columns = ['income', 'home_taz'])
    h = h.groupby('home_taz')['income'].mean()
    return h.fillna(0)

@orca.column('zones')
def pct_hh_inc_under_25k(households):
    h = households.to_frame(columns = ['home_taz', 'hh_inc_under_25k'])
    h = h.groupby('home_taz')['hh_inc_under_25k'].mean()
    return h.fillna(0)

@orca.column('zones')
def pct_hh_inc_25_to_75k(households):
    h = households.to_frame(columns = ['home_taz', 'hh_inc_25_to_75k'])
    h = h.groupby('home_taz')['hh_inc_25_to_75k'].mean()
    return h.fillna(0)

@orca.column('zones')
def pct_hh_inc_75_to_200k(households):
    h = households.to_frame(columns = ['home_taz', 'hh_inc_75_to_200k'])
    h = h.groupby('home_taz')['hh_inc_75_to_200k'].mean()
    return h.fillna(0)

@orca.column('zones')
def pct_no_higher_ed(persons):
    p = persons.to_frame(columns = ['home_taz', 'no_higher_ed'])
    p = p.groupby('home_taz')['no_higher_ed'].mean()
    return p.fillna(0)

@orca.column('zones')
def pct_sector_tech(jobs):
    j = jobs.to_frame(columns = ['taz', 'sector_tech'])
    j = j.groupby('taz')['sector_tech'].mean()
    return j.fillna(0)

@orca.column('zones')
def pct_sector_retail(jobs):
    j = jobs.to_frame(columns = ['taz', 'sector_retail'])
    j = j.groupby('taz')['sector_retail'].mean()
    return j.fillna(0)

@orca.column('zones', cache = True)
def pct_sector_healthcare(jobs):
    j = jobs.to_frame(columns = ['taz', 'sector_healthcare'])
    j = j.groupby('taz')['sector_healthcare'].mean()
    return j.fillna(0)

@orca.column('households')
def hh_inc_under_25k(households):
    i = households.income
    return i.between(-np.inf, 25000).astype(int)

@orca.column('households')
def hh_inc_25_to_75k(households):
    i = households.income
    return i.between(25000, 75000).astype(int)

@orca.column('households')
def hh_inc_75_to_200k(households):
    i = households.income
    return i.between(75000, 200000).astype(int)

@orca.column('persons')
def no_higher_ed(persons):
    education = persons.edu
    return education.between(-np.inf, 17).astype(int)

@orca.column('jobs')
def sector_retail(jobs):
    return jobs.sector_id.isin(['44-45']).astype(int)

@orca.column('jobs')
def sector_healthcare(jobs):
    return jobs.sector_id.isin(['62']).astype(int)

@orca.column('jobs')
def sector_tech(jobs):
    return jobs.sector_id.isin(['51', '54']).astype(int)

@orca.column('zones')
def jobs_capacity(jobs):
    capacity = jobs.taz.value_counts()
#     capacity.index = capacity.index.astype(int)
    return capacity

@orca.column('persons')
def taz_pct_no_higher_ed(persons, zones):
    return misc.reindex(zones.pct_no_higher_ed, persons.home_taz)

@orca.column('persons')
def taz_pct_hh_inc_under_25k(persons, zones):
    return misc.reindex(zones.pct_hh_inc_under_25k, persons.home_taz)

@orca.column('blocks')
def pct_sector_tech(jobs):
    j = jobs.to_frame(columns = ['block_id', 'sector_tech'])
    return j.groupby('block_id').agg({'sector_tech': 'mean'}).fillna(0)

@orca.column('blocks')
def pct_sector_retail(jobs):
    j = jobs.to_frame(columns = ['block_id', 'sector_retail'])
    return j.groupby('block_id').agg({'sector_retail': 'mean'}).fillna(0)

@orca.column('travel_data')
def dist_0_5(travel_data):
    return travel_data.tour_dist.clip(0,5)

@orca.column('travel_data')
def dist_5_15(travel_data):
    return (travel_data.tour_dist - 5).clip(0,10)

@orca.column('travel_data')
def dist_15plus(travel_data):
    return (travel_data.tour_dist -  15).clip(0)
# -----------------------------------------------------------------------------------------
# DEMOGRAPHIC VARIABLES
# -----------------------------------------------------------------------------------------

@orca.column('persons')
def intercept(persons):
    size = persons.to_frame(columns=["age"]).shape[0]
    return np.ones(size)

# @orca.column('persons')
# def dead(persons):
#     size = persons.to_frame(columns=["age"]).shape[0]
#     return np.zeros(size) - 99

# @orca.column('persons')
# def stop(persons):
#     size = persons.to_frame(columns=["age"]).shape[0]
#     return np.zeros(size) - 99

# @orca.column('persons')
# def kids_move(persons):
#     size = persons.to_frame(columns=["age"]).shape[0]
#     return np.zeros(size) - 99

@orca.column('persons')
def agebin1_labor(persons):
    p = persons.to_frame(columns=['age'])['age']
    return p.between(20, 30, inclusive='both') * 1

@orca.column('persons')
def agebin2_labor(persons):
    p = persons.to_frame(columns=['age'])['age']
    return p.between(31, 40, inclusive='both') * 1

@orca.column('persons')
def agebin3_labor(persons):
    p = persons.to_frame(columns=['age'])['age']
    return p.between(41, 50, inclusive='both') * 1

@orca.column('persons')
def agebin4_labor(persons):
    p = persons.to_frame(columns=['age'])['age']
    return p.between(51, 70, inclusive='both') * 1

@orca.column('persons')
def agebin5_labor(persons):
    p = persons.to_frame(columns=['age'])['age']
    return (p>70) * 1

@orca.column('persons')
def agebin1(persons):
    p = persons.to_frame(columns=['age'])['age']
    return p.between(16, 22, inclusive='both') * 1


@orca.column('persons')
def agebin2(persons):
    p = persons.to_frame(columns=['age'])['age']
    return p.between(23, 35, inclusive='both') * 1


@orca.column('persons')
def agebin3(persons):
    p = persons.to_frame(columns=['age'])['age']
    return p.between(36, 60, inclusive='both') * 1


@orca.column('persons')
def agebin4(persons):
    p = persons.to_frame(columns=['age'])
    return p.gt(60) * 1

@orca.column('persons')
def agebin1_mo(persons):
    p = persons.to_frame(columns=['age'])['age']
    return p.between(16, 18, inclusive='both') * 1

@orca.column('persons')
def agebin2_mo(persons):
    p = persons.to_frame(columns=['age'])['age']
    return p.between(19, 20, inclusive='both') * 1

@orca.column('persons')
def agebin3_mo(persons):
    p = persons.to_frame(columns=['age'])['age']
    return p.between(21, 25, inclusive='both') * 1

@orca.column('persons')
def agebin4_mo(persons):
    p = persons.to_frame(columns=['age'])['age']
    return p.between(26, 30, inclusive='both') * 1

@orca.column('persons')
def agebin5_mo(persons):
    p = persons.to_frame(columns=['age'])['age']
    return p.gt(30) * 1

# Male
@orca.column('persons')
def gender1(persons):
    p = persons.to_frame(columns=['sex'])
    return p.eq(1).astype(int)


# Female
@orca.column('persons')
def gender2(persons):
    p = persons.to_frame(columns=['sex'])
    return p.eq(2).astype(int)


@orca.column('persons')
def employbin1(persons):
    p = persons.to_frame(columns=['worker'])
    return p.eq(1) * 1


@orca.column('persons')
def employbin2(persons):
    p = persons.to_frame(columns=['worker', 'age'])
    return (p['worker'].eq(0) & p['age'].lt(60)).astype(int)


@orca.column('persons')
def employbin3(persons):
    p = persons.to_frame(columns=['worker', 'age'])
    return (p['worker'].eq(0) & p['age'].ge(60)).astype(int)

@orca.column('persons')
def employ2_agebin2_mo(persons):
    p = persons.to_frame(columns=["employbin2", "agebin2_mo"])
    return p["employbin2"] * p["agebin2_mo"]

@orca.column('persons')
def employ2_agebin3_mo(persons):
    p = persons.to_frame(columns=["employbin2", "agebin3_mo"])
    return p["employbin2"] * p["agebin3_mo"]

@orca.column('persons')
def employ2_agebin4_mo(persons):
    p = persons.to_frame(columns=["employbin2", "agebin4_mo"])
    return p["employbin2"] * p["agebin4_mo"]

@orca.column('persons')
def employ2_agebin5_mo(persons):
    p = persons.to_frame(columns=["employbin2", "agebin5_mo"])
    return p["employbin2"] * p["agebin5_mo"]

###

@orca.column('persons')
def employ3_agebin2_mo(persons):
    p = persons.to_frame(columns=["employbin3", "agebin2_mo"])
    return p["employbin3"] * p["agebin2_mo"]

@orca.column('persons')
def employ3_agebin3_mo(persons):
    p = persons.to_frame(columns=["employbin3", "agebin3_mo"])
    return p["employbin3"] * p["agebin3_mo"]

@orca.column('persons')
def employ3_agebin4_mo(persons):
    p = persons.to_frame(columns=["employbin3", "agebin4_mo"])
    return p["employbin3"] * p["agebin4_mo"]

@orca.column('persons')
def employ3_agebin5_mo(persons):
    p = persons.to_frame(columns=["employbin3", "agebin5_mo"])
    return p["employbin3"] * p["agebin5_mo"]

# Less than high school
@orca.column('persons')
def edubin1(persons):
    p = persons.to_frame(columns=['edu'])
    return p.lt(16) * 1


# High School or GED
@orca.column('persons')
def edubin2(persons):
    p = persons.to_frame(columns=['edu'])['edu']
    return p.between(16, 17, inclusive='both') * 1


# Some college or more
@orca.column('persons')
def edubin3(persons):
    p = persons.to_frame(columns=['edu'])
    return p.gt(17) * 1


@orca.column('persons')
def race_wht(persons):
    p = persons.to_frame(columns=['race_id'])
    return p.eq(1) * 1


@orca.column('persons')
def race_blk(persons):
    p = persons.to_frame(columns=['race_id'])
    return p.eq(2) * 1


@orca.column('persons')
def race_asn(persons):
    p = persons.to_frame(columns=['race_id'])['race_id']
    return p.between(6, 7) * 1


list_of_other_races = [3, 4, 5, 8, 9]


# Other Races
@orca.column('persons')
def race_other(persons):
    p = persons.to_frame(columns=['race_id'])
    return p.isin(list_of_other_races) * 1


# White
@orca.column('persons')
def race1(persons):
    p = persons.to_frame(columns=['race_wht'])
    return p['race_wht']


# Black
@orca.column('persons')
def race2(persons):
    p = persons.to_frame(columns=['race_blk'])
    return p['race_blk']


native_american_list = [3, 4, 5]


# Native American
@orca.column('persons')
def race3(persons):
    p = persons.to_frame(columns=['race_id'])
    return p.isin(native_american_list) * 1


# Asian or Pacific Islander
@orca.column('persons')
def race4(persons):
    p = persons.to_frame(columns=['race_asn'])
    return p['race_asn']


# Other
@orca.column('persons')
def race5(persons):
    p = persons.to_frame(columns=['race_id'])
    return p.eq(8) * 1


# Multiracial
@orca.column('persons')
def race6(persons):
    p = persons.to_frame(columns=['race_id'])
    return p.eq(9) * 1


@orca.column('persons')
def age2_edu2(persons):
    p = persons.to_frame(columns=['agebin2', 'edubin2'])
    p['age2_edu2'] = p['agebin2'] * p['edubin2']
    return p['age2_edu2']


@orca.column('persons')
def age3_edu2(persons):
    p = persons.to_frame(columns=['agebin3', 'edubin2'])
    p['age3_edu2'] = p['agebin3'] * p['edubin2']
    return p['age3_edu2']


@orca.column('persons')
def age4_edu2(persons):
    p = persons.to_frame(columns=['agebin4', 'edubin2'])
    p['age4_edu2'] = p['agebin4'] * p['edubin2']
    return p['age4_edu2']


@orca.column('persons')
def age2_edu3(persons):
    p = persons.to_frame(columns=['agebin2', 'edubin3'])
    p['age2_edu3'] = p['agebin2'] * p['edubin3']
    return p['age2_edu3']


@orca.column('persons')
def age3_edu3(persons):
    p = persons.to_frame(columns=['agebin3', 'edubin3'])
    p['age3_edu3'] = p['agebin3'] * p['edubin3']
    return p['age3_edu3']


@orca.column('persons')
def age4_edu3(persons):
    p = persons.to_frame(columns=['agebin4', 'edubin3'])
    p['age4_edu3'] = p['agebin4'] * p['edubin3']
    return p['age4_edu3']

@orca.column('persons')
def edu2_agebin2_mo(persons):
    p = persons.to_frame(columns=['agebin2_mo', 'edubin2'])
    p['edu2_agebin2_mo'] = p['agebin2_mo'] * p['edubin2']
    return p['edu2_agebin2_mo']


@orca.column('persons')
def edu2_agebin3_mo(persons):
    p = persons.to_frame(columns=['agebin3_mo', 'edubin2'])
    p['edu2_agebin3_mo'] = p['agebin3_mo'] * p['edubin2']
    return p['edu2_agebin3_mo']


@orca.column('persons')
def edu2_agebin4_mo(persons):
    p = persons.to_frame(columns=['agebin4_mo', 'edubin2'])
    p['edu2_agebin4_mo'] = p['agebin4_mo'] * p['edubin2']
    return p['edu2_agebin4_mo']

@orca.column('persons')
def edu2_agebin5_mo(persons):
    p = persons.to_frame(columns=['agebin5_mo', 'edubin2'])
    p['edu2_agebin5_mo'] = p['agebin5_mo'] * p['edubin2']
    return p['edu2_agebin5_mo']

@orca.column('persons')
def edu3_agebin2_mo(persons):
    p = persons.to_frame(columns=['agebin2_mo', 'edubin3'])
    p['edu3_agebin2_mo'] = p['agebin2_mo'] * p['edubin3']
    return p['edu3_agebin2_mo']


@orca.column('persons')
def edu3_agebin3_mo(persons):
    p = persons.to_frame(columns=['agebin3_mo', 'edubin3'])
    p['edu3_agebin3_mo'] = p['agebin3_mo'] * p['edubin3']
    return p['edu3_agebin3_mo']


@orca.column('persons')
def edu3_agebin4_mo(persons):
    p = persons.to_frame(columns=['agebin4_mo', 'edubin3'])
    p['edu3_agebin4_mo'] = p['agebin4_mo'] * p['edubin3']
    return p['edu3_agebin4_mo']


@orca.column('persons')
def edu3_agebin5_mo(persons):
    p = persons.to_frame(columns=['agebin5_mo', 'edubin3'])
    p['edu3_agebin5_mo'] = p['agebin5_mo'] * p['edubin3']
    return p['edu3_agebin5_mo']


@orca.column('persons')
def marital1(persons):
    p = persons.to_frame(columns=['MAR'])
    return p.eq(1).astype(int)


@orca.column('persons')
def marital2(persons):
    p = persons.to_frame(columns=['MAR'])
    return p.eq(2).astype(int)


@orca.column('persons')
def marital3(persons):
    p = persons.to_frame(columns=['MAR'])
    # print(persons.local.columns)
    return p.eq(3).astype(int)


@orca.column('persons')
def marital4(persons):
    p = persons.to_frame(columns=['MAR'])
    return p.gt(3).astype(int)


@orca.column('persons')
def married_before(persons):
    p = persons.to_frame(columns=['MAR'])
    return p['MAR'].between(2, 4) * 1


# PERSON VARIABLES
# -----------------------------------------------------------------------------------------


@orca.column('persons', cache=True)
def mandatory_work_zone_id(persons):
    return persons.work_zone_id.where(persons.age > 17, "-1").astype(str)


@orca.column('persons', cache=True)
def mandatory_school_zone_id(persons):
    return persons.school_zone_id.where(persons.age > 17, "-1").astype(str)


@orca.column('persons', cache=True)
def mandatory_work_dummy(persons):
    has_work = persons.mandatory_work_zone_id != "-1"
    return has_work.astype(int)


@orca.column('persons', cache=True)
def mandatory_school_dummy(persons):
    has_school = persons.mandatory_school_zone_id != "-1"
    return has_school.astype(int)


@orca.column('persons', cache=True)
def mandatory_activity_dummy(persons):
    school_or_work = ((
        persons.mandatory_school_dummy.astype(bool)) | (
        persons.mandatory_work_dummy.astype(bool)))
    return school_or_work.astype(int)
# -----------------------------------------------------------------------------------------
# HOUSEHOLD VARIABLES
# -----------------------------------------------------------------------------------------


@orca.column('households', cache=True, cache_scope='step')
def intercept(households):
    return np.ones(households.local.shape[0])

# @orca.column('households')
# def birth(households):
#     return np.zeros(households.local.shape[0]) - 99

# @orca.column('households')
# def divorced(households):
#     return np.zeros(households.local.shape[0]) - 99

# @orca.column('households')
# def birth(households):
#     return np.zeros(households.local.shape[0]) - 99


@orca.column('households', cache=True, cache_scope='step')
def county_id(households):
    hh = households.to_frame('block_id')
    return hh['block_id'].str.slice(0,5)


@orca.column('households', cache=True, cache_scope='step')
def tract_id(households):
    hh = households.to_frame('block_id')
    return hh['block_id'].str.slice(0,11)


@orca.column('households', cache=True)
def hh_type():
    hh = orca.get_table('households').to_frame(['persons', 'tenure', 'age_of_head'])
    hh['gt55'] = (hh.age_of_head >= 55).astype('int')
    hh['gt2'] = (hh.persons >= 2).astype('int')
    hh['hh_type'] = 0
    hh.loc[(hh.tenure == 1) & (hh.gt2 == 0) & (hh.gt55 == 0), 'hh_type'] = 1
    hh.loc[(hh.tenure == 1) & (hh.gt2 == 0) & (hh.gt55 == 1), 'hh_type'] = 2
    hh.loc[(hh.tenure == 1) & (hh.gt2 == 1) & (hh.gt55 == 0), 'hh_type'] = 3
    hh.loc[(hh.tenure == 1) & (hh.gt2 == 1) & (hh.gt55 == 1), 'hh_type'] = 4
    hh.loc[(hh.tenure == 2) & (hh.gt2 == 0) & (hh.gt55 == 0), 'hh_type'] = 5
    hh.loc[(hh.tenure == 2) & (hh.gt2 == 0) & (hh.gt55 == 1), 'hh_type'] = 6
    hh.loc[(hh.tenure == 2) & (hh.gt2 == 1) & (hh.gt55 == 0), 'hh_type'] = 7
    hh.loc[(hh.tenure == 2) & (hh.gt2 == 1) & (hh.gt55 == 1), 'hh_type'] = 8
    for col in ['gt55', 'gt2']: del hh[col]
    return hh.hh_type


@orca.column('households')
def persons(households, persons):
    hh = households.local.copy()
    persons = pd.DataFrame(persons.local.groupby('household_id').size())
    persons.columns = ['total_persons']
    hh = hh.join(persons)
    return hh['total_persons']


@orca.column('households')
def children(households, persons):
    hh = households.local.copy()
    persons = persons.local.copy()
    children = persons[persons['age'] <= 17].groupby('household_id').count()
    hh = hh.join(children[['age']]).fillna(0)
    return hh['age']


@orca.column('households')
def adult_count(households, persons):
    hh = households.local.copy()
    p = persons.local.copy()
    p = p[p['age'] >= 18].groupby('household_id').count()
    return hh.join(p)["age"].fillna(0)

@orca.column('households')
def adult_count2(households, persons):
    hh = households.local.copy()
    p = persons.local.copy()
    p = (p[p['age'] >= 18].groupby('household_id').count() == 2)
    return hh.join(p)["age"].fillna(False) * 1

@orca.column('households')
def adult_count_gt2(households, persons):
    hh = households.local.copy()
    p = persons.local.copy()
    p = (p[p['age'] >= 18].groupby('household_id').count() > 2)
    return hh.join(p)["age"].fillna(False) * 1


@orca.column('households')
def persons_65plus(households, persons):
    hh = households.local.copy()
    persons = persons.local.copy()
    persons = persons[persons['age'] >= 65].groupby('household_id').count()
    hh = hh.join(persons[['age']]).fillna(0)
    return hh['age']


@orca.column('households')
def persons_black(households, persons):
    hh = households.local.copy()
    persons = persons.local.copy()
    persons = persons[persons['race'] == 'black'].groupby('household_id').count()
    hh = hh.join(persons[['race']]).fillna(0)
    return hh['race']


@orca.column('households')
def persons_hispanic(households, persons):
    hh = households.local.copy()
    persons = persons.local.copy()
    persons = persons[persons['hispanic'] == 1].groupby('household_id').count()
    hh = hh.join(persons[['hispanic']]).fillna(0)
    return hh['hispanic']


@orca.column('households')
def persons_asian(households, persons):
    hh = households.local.copy()
    persons = persons.local.copy()
    persons = persons[persons['race'] == 'asian'].groupby('household_id').count()
    hh = hh.join(persons[['race']]).fillna(0)
    return hh['race']


@orca.column('households')
def income_segment(households):
    s = pd.Series(households.income.transform(
        lambda x: pd.qcut(x.rank(method='first'),
                          q = [0., .1, .25, .5, .75, .9, 1.], labels = False)), index=households.index)
    s = s.add(1)
    return s

@orca.column('households')
def income_bin1(households):
    df = households.to_frame(columns=["income"])
    return df.lt(250000) * 1

@orca.column('households')
def income_bin2(households):
    df = households.to_frame(columns=["income"])["income"]
    return df.between(25000, 50000, inclusive='left') * 1

@orca.column('households')
def income_bin3(households):
    df = households.to_frame(columns=["income"])["income"]
    return df.between(50000, 75000, inclusive='left') * 1

@orca.column('households')
def income_bin4(households):
    df = households.to_frame(columns=["income"])["income"]
    return df.between(75000, 150000, inclusive='left') * 1

@orca.column('households')
def income_bin5(households):
    df = households.to_frame(columns=["income"])
    return df.ge(150000) * 1


# max edu year
@orca.column('households')
def top_edu(persons):
    df = persons.to_frame(columns=['household_id', 'edu', 'relate'])
    df = df[df['relate'] < 2][["household_id", "edu"]]
    return df.groupby('household_id').agg({'edu': 'max'})

# less than high school
@orca.column('households')
def top_edu_bin1(households):
    df = households.to_frame(columns=["top_edu"])
    return df.lt(16) * 1

# high school or equivalent
@orca.column('households')
def top_edu_bin2(households):
    df = households.to_frame(columns=["top_edu"])["top_edu"]
    return df.between(16, 17, inclusive='both') * 1

# Some college, college, or more than college
@orca.column('households')
def top_edu_bin3(households):
    df = households.to_frame(columns=["top_edu"])
    return df.gt(17) * 1


@orca.column('households')
def have_spouse(persons, households):
    df = persons.to_frame(columns=['household_id', 'relate'])
    households_df = households.to_frame(columns=['serialno'])
    household_id = households_df.index.to_series()
    filtered_houses = df[df['relate'] == 1]['household_id']
    return household_id.isin(filtered_houses) * 1

@orca.column('households')
def hh_birth_agebin1(persons, households):
    df = persons.to_frame(columns=['household_id', 'relate', 'sex', 'age'])
    households_df = households.to_frame(columns=["have_spouse"]).reset_index()
    df = df.merge(households_df, on="household_id")
    # subset = df[df["relate"].isin([0, 1])]
    # print("DF shape:", df.shape[0])
    df.loc[:,"is_head"] = np.where(df["relate"]==0, 1, 0)
    df.loc[:,"is_female"] = np.where(df["sex"]==2, 1, 0)
    df.loc[:,"female_head"] = df["is_head"] * df["is_female"]
    df.loc[:,"is_head_or_spouse"] = np.where(df["relate"].isin([0, 1, 13]), 1, 0)
    df.loc[:,"age_head"] = df["age"] * df["is_head"]
    df.loc[:,"age_female"] = df["age"] * df["is_female"] * df["is_head_or_spouse"]
    df.loc[:, "is_spouse"] = np.where(df["relate"].isin([1, 13]), 1, 0)
    df.loc[:,"head_spouse"] = df["is_head"] + df["is_spouse"]
    df = df.groupby("household_id").agg(
        # size = ("person", "sum"),
        age_head = ("age_head", "sum"),
        age_female = ("age_female", "sum"),
        head_spouse = ("head_spouse", "sum")
    ).reset_index()
    df.loc[:, "age_final"] = np.where(df["head_spouse"]>=2, df["age_female"], df["age_head"])
    # print("NEW DF", df.shape[0])
    return np.where(df["age_final"]<=27, 1, 0)

@orca.column('households')
def hh_birth_agebin2(persons, households):
    df = persons.to_frame(columns=['household_id', 'relate', 'sex', 'age'])
    households_df = households.to_frame(columns=["have_spouse"]).reset_index()
    df = df.merge(households_df, on="household_id")
    # subset = df[df["relate"].isin([0, 1])]
    df.loc[:, "is_head"] = np.where(df["relate"]==0, 1, 0)
    df.loc[:, "is_female"] = np.where(df["sex"]==2, 1, 0)
    df.loc[:, "female_head"] = df["is_head"] * df["is_female"]
    df.loc[:, "is_head_or_spouse"] = np.where(df["relate"].isin([0, 1, 13]), 1, 0)
    df.loc[:, "age_head"] = df["age"] * df["is_head"]
    df.loc[:, "age_female"] = df["age"] * df["is_female"] * df["is_head_or_spouse"]
    df.loc[:, "is_spouse"] = np.where(df["relate"].isin([1, 13]), 1, 0)
    df.loc[:, "head_spouse"] = df["is_head"] + df["is_spouse"]
    df = df.groupby("household_id").agg(
        # size = ("person", "sum"),
        age_head = ("age_head", "sum"),
        age_female = ("age_female", "sum"),
        head_spouse = ("head_spouse", "sum")
    ).reset_index()
    df.loc[:, "age_final"] = np.where(df["head_spouse"]>=2, df["age_female"], df["age_head"])
    # print(df.shape[0])
    return np.where(df["age_final"].between(27, 35, inclusive='right'), 1, 0)

# @orca.column('households')
# def use_agebin3(persons, households):
#     df = persons.to_frame(columns=['household_id', 'relate'])
#     households_df = households.to_frame(columns=["have_spouse"]).reset_index()
#     df.merge(households_df, on="household_id")
#     subset = df[df["relate"].isin([0, 1])]
    
#     subset["is_head"] = np.where(df["relate"]==0, 1, 0)
#     subset["is_female"] = np.where(df["sex"]==2, 1, 0)
#     subset["female_head"] = df["is_head"] * df["is_female"]
#     subset["person"] = 1
#     subset["age_head"] = subset["age"] * subset["is_head"]
#     subset["age_female"] = subset["age"] * subset["is_female"]
#     subset = subset.groupby("household_id").agg(
#         size = ("person", "sum"),
#         age_head = ("age_head", "sum"),
#         age_female = ("age_female", "sum")
#     ).reset_index()
#     subset["age_final"] = np.where(subset["size"]==2, subset["age_female"], subset["age_head"])
    
    # return np.where(subset["age_final"]>35, 1, 0)


@orca.column('households')
def fam_work(persons):
    df = persons.to_frame(columns=['relate', 'worker', 'household_id'])
    df = df[df['relate'] < 2]
    return df.groupby('household_id').agg({'worker': 'sum'})


@orca.column('households')
def fam_work0(households):
    df = households.to_frame(columns=['fam_work'])
    return df.eq(0) * 1


@orca.column('households')
def fam_work1(households):
    df = households.to_frame(columns=['fam_work'])
    return df.eq(1) * 1


@orca.column('households')
def fam_work2(households):
    df = households.to_frame(columns=['fam_work'])
    return df.eq(2) * 1


@orca.column('households')
def fsize_bin1(households):
    df = households.to_frame(columns=['persons'])
    return df.eq(1) * 1


@orca.column('households')
def fsize_bin2(households):
    df = households.to_frame(columns=['persons'])
    return df.eq(2) * 1


@orca.column('households')
def fsize_bin3(households):
    df = households.to_frame(columns=['persons'])
    return df.eq(3) * 1


@orca.column('households')
def fsize_bin35(households):
    df = households.to_frame(columns=['persons'])
    return df['persons'].between(4, 5) * 1


@orca.column('households')
def fsize_bin5(households):
    df = households.to_frame(columns=['persons'])
    return df.gt(5) * 1


@orca.column('households')
def avg_age(persons):
    df = persons.to_frame(columns=['relate', 'age', 'household_id'])
    df = df[df['relate'] < 2]
    return df.groupby('household_id').agg({'age': 'mean'})

@orca.column('households')
def avg_agebin1(households):
    df = households.to_frame(columns=['avg_age'])["avg_age"]
    return df.between(16, 22, inclusive='both') * 1

@orca.column('households')
def avg_agebin2(households):
    df = households.to_frame(columns=['avg_age'])["avg_age"]
    return df.between(22, 35, inclusive='right') * 1

@orca.column('households')
def avg_agebin3(households):
    df = households.to_frame(columns=['avg_age'])["avg_age"]
    return df.between(35, 60, inclusive='right') * 1

@orca.column('households')
def avg_agebin4(households):
    df = households.to_frame(columns=['avg_age'])
    return df.gt(60) * 1


@orca.column('households')
def min_age(persons):
    df = persons.to_frame(columns=['relate', 'age', 'household_id'])
    df = df[df['relate'] < 2]
    return df.groupby('household_id').agg({'age': 'min'})

@orca.column('households')
def min_agebin1(households):
    df = households.to_frame(columns=['min_age'])["min_age"]
    return df.between(16, 22, inclusive='both') * 1

@orca.column('households')
def min_agebin2(households):
    df = households.to_frame(columns=['min_age'])["min_age"]
    return df.between(22, 35, inclusive='right') * 1

@orca.column('households')
def min_agebin3(households):
    df = households.to_frame(columns=['min_age'])["min_age"]
    return df.between(35, 60, inclusive='right') * 1

@orca.column('households')
def min_agebin4(households):
    df = households.to_frame(columns=['min_age'])
    return df.gt(60) * 1

@orca.column('households')
def kidsn0(households):
    df = households.to_frame(columns=['children'])
    return df.eq(0) * 1


@orca.column('households')
def kidsn1(households):
    df = households.to_frame(columns=['children'])
    return df.eq(1) * 1


@orca.column('households')
def kidsn2(households):
    df = households.to_frame(columns=['children'])
    return df.eq(2) * 1


@orca.column('households')
def kidsn3(households):
    df = households.to_frame(columns=['children'])
    return df.gt(2) * 1


@orca.column('households')
def hd_race_wht(persons):
    df = persons.to_frame(columns=['relate', 'race_wht', 'household_id'])
    df = df[df['relate'] == 0]
    df.sort_values('household_id',inplace=True)
    return df.set_index('household_id')['race_wht']

@orca.column('households')
def hd_agebin1(households):
    df = households.to_frame(columns=['age_of_head'])['age_of_head']
    return df.between(16, 22, inclusive='right') * 1

@orca.column('households')
def hd_agebin2(households):
    df = households.to_frame(columns=['age_of_head'])['age_of_head']
    return df.between(22, 35, inclusive='right') * 1

@orca.column('households')
def hd_agebin3(households):
    df = households.to_frame(columns=['age_of_head'])['age_of_head']
    return df.between(35, 60, inclusive='right') * 1

@orca.column('households')
def hd_agebin4(households):
    df = households.to_frame(columns=['age_of_head'])['age_of_head']
    return df.gt(60) * 1

@orca.column('households')
def husband_works(households, persons):
    df = persons.to_frame(columns=['relate', 'household_id', 'worker', 'sex'])
    households = households.to_frame(columns=['cars'])
    df = df[(df['relate']<2) & (df['sex']==1)]
    households['husband_works'] = df.groupby('household_id').agg({'worker': 'first'})
    households['husband_works'] = households['husband_works'].fillna(0).astype(int)
    return households['husband_works']

@orca.column('households')
def sex_of_head(persons):
    df = persons.to_frame(columns=['relate', 'household_id', 'sex'])
    df.sort_values('relate', inplace=True)
    return df.groupby('household_id').agg({'sex':'first'})


@orca.column('households')
def fes(households):
    df = households.to_frame(columns=['fam_work1', 'fam_work2', 'have_spouse', 'husband_works', 'sex_of_head', 'fam_work0'])
    df['fes'] = 0
    df.loc[(df.fam_work2 == 1), 'fes'] = 1
    df.loc[(df.have_spouse == 1) & (df.fam_work1 == 1) & (df.husband_works == 1), 'fes'] = 2
    df.loc[(df.have_spouse == 1) & (df.fam_work1 == 1) & (df.husband_works == 0), 'fes'] = 3
    df.loc[(df.have_spouse == 1) & (df.fam_work0 == 1), 'fes'] = 4
    df.loc[(df.have_spouse == 0) & (df.fam_work0 == 1) & (df.sex_of_head == 1), 'fes'] = 6
    df.loc[(df.have_spouse == 0) & (df.fam_work0 == 1) & (df.sex_of_head == 2), 'fes'] = 8
    df.loc[(df.have_spouse == 0) & (df.fam_work1 == 1) & (df.sex_of_head == 1), 'fes'] = 5
    df.loc[(df.have_spouse == 0) & (df.fam_work1 == 1) & (df.sex_of_head == 2), 'fes'] = 7
    return df['fes']


# -----------------------------------------------------------------------------------------
# JOB VARIABLES
# -----------------------------------------------------------------------------------------


@orca.column('jobs', cache=True, cache_scope='step')
def county_id(jobs):
    jobs = jobs.to_frame('block_id')
    return jobs['block_id'].str.slice(0,5)


@orca.column('jobs',  cache=True, cache_scope='step')
def tract_id(jobs):
    jobs = jobs.to_frame('block_id')
    return jobs['block_id'].str.slice(0,11)


# -----------------------------------------------------------------------------------------
# UNIT VARIABLES
# -----------------------------------------------------------------------------------------


@orca.column('residential_units', cache=True, cache_scope='step')
def county_id(residential_units):
    units = residential_units.to_frame('block_id')
    return units['block_id'].str.slice(0,5)


@orca.column('residential_units',  cache=True, cache_scope='step')
def tract_id(residential_units):
    units = residential_units.to_frame('block_id')
    return units['block_id'].str.slice(0,11)


@orca.column('residential_units', cache=True)
def building_type(residential_units, btypes_dict):
    units = residential_units.local.copy()
    building_type_dict = {btypes_dict[key]: key for key in btypes_dict.keys()}
    units['building_type'] = units['building_type_id'].map(building_type_dict)
    return units['building_type']


# -----------------------------------------------------------------------------------------
# BLOCK VARIABLES
# -----------------------------------------------------------------------------------------


@orca.column('blocks', cache=True, cache_scope='step')
def total_jobs(blocks, jobs):
    jobs = jobs.local.groupby('block_id').count()
    jobs = pd.DataFrame(jobs.iloc[:,1])
    jobs.columns = ['total_jobs']
    blocks = blocks.local.join(jobs).fillna(0)
    return blocks['total_jobs']


@orca.column('blocks', cache=True, cache_scope='step')
def density_jobs(blocks):
    blocks = blocks.to_frame(['total_jobs', 'sum_acres'])
    series = (blocks.total_jobs) * 1.0 / (blocks.sum_acres+ 1.0)
    return series.fillna(0)


@orca.column('blocks', cache=True, cache_scope='step')
def density_jobs_90pct_plus(blocks):
    blocks = blocks.to_frame('density_jobs')
    percentile = blocks['density_jobs'].quantile(.9)
    blocks['percentile'] = 0
    blocks.loc[blocks['density_jobs'] >= percentile, 'percentile'] = 1
    return blocks['percentile']


@orca.column('blocks', cache=True, cache_scope='step')
def density_jobs_10pct_low(blocks):
    blocks = blocks.to_frame('density_jobs')
    percentile = blocks['density_jobs'].quantile(.1)
    blocks['percentile'] = 0
    blocks.loc[blocks['density_jobs'] <= percentile, 'percentile'] = 1
    return blocks['percentile']


@orca.column('blocks', 'job_spaces', cache=True)
def job_spaces(blocks):
    blocks_df = blocks.to_frame(['employment_capacity', 'sum_acres'])
    blocks_df.loc[blocks_df['sum_acres']==0, 'employment_capacity'] = 0
    employment_capacity = blocks_df.employment_capacity
    employment_capacity = employment_capacity*orca.get_injectable('capacity_boost')
    if 'proportion_undevelopable' in blocks.local_columns:
        print('Adjusting employment capacities based on proportion undevelopable.')
        return (employment_capacity * (1 - blocks.proportion_undevelopable)).astype('int')
    else:
        return employment_capacity


@orca.column('blocks', 'job_spaces_acre', cache=True)
def job_spaces_acre(blocks):
    df = blocks.to_frame(['job_spaces', 'sum_acres'])
    df['job_spaces_acre'] = df['job_spaces']/df['sum_acres']
    return df['job_spaces_acre'].fillna(0)


@orca.column('blocks', 'vacant_job_spaces', cache=False)
def vacant_job_spaces(blocks, jobs):
    return blocks.job_spaces.sub(jobs.block_id.value_counts(), fill_value=0)


@orca.column('blocks', cache=True, cache_scope='step')
def total_hh(blocks, households):
    hh = households.local.groupby('block_id').count()
    blocks = blocks.local.join(hh[['serialno']]).fillna(0)
    return blocks['serialno']


@orca.column('blocks', cache=True, cache_scope='step')
def density_hh(blocks):
    blocks = blocks.to_frame(['total_hh', 'sum_acres'])
    series = blocks.total_hh * 1.0 / (blocks.sum_acres + 1.0)
    return series.fillna(0)


@orca.column('blocks', cache=True, cache_scope='step')
def hh_size_1(blocks, households):
    hh = households.to_frame(['persons', 'block_id'])
    hh_size_1 = hh[hh['persons']==1].groupby('block_id').size()
    return pd.Series(index=blocks.index, data=hh_size_1).fillna(0)


@orca.column('blocks', cache=True, cache_scope='step')
def hh_size_5plus(blocks, households):
    hh = households.to_frame(['persons', 'block_id'])
    hh_size_5plus = hh[hh['persons']>=5].groupby('block_id').size()
    return pd.Series(index=blocks.index, data=hh_size_5plus).fillna(0)


# TODO: revert var names to hh_income_segment_1 for consistency. Need to fix calibrated configs also
@orca.column('blocks', cache=True, cache_scope='step')
def income_segment_1_hh(blocks, households):
    households = households.to_frame(['income_segment', 'block_id'])
    hh_by_block = households[households.income_segment == 1].groupby('block_id').size()
    return pd.Series(index=blocks.index, data=hh_by_block).fillna(0)


@orca.column('blocks', cache=True, cache_scope='step')
def income_segment_6_hh(blocks, households):
    households = households.to_frame(['income_segment', 'block_id'])
    hh_by_block = households[households.income_segment == 6].groupby('block_id').size()
    return pd.Series(index=blocks.index, data=hh_by_block).fillna(0)


@orca.column('blocks', cache=True)
def mean_income(blocks, households):
    income = households.to_frame(['income', 'block_id']).groupby('block_id').mean()
    blocks = blocks.local.join(income)
    return blocks['income'].fillna(blocks['income'].median())


@orca.column('blocks', cache=True)
def mean_hh_size(blocks, households):
    size = households.to_frame(['persons', 'block_id']).groupby('block_id').mean()
    blocks = blocks.local.join(size)
    return blocks['persons'].fillna(blocks['persons'].median())


@orca.column('blocks', cache=True, cache_scope='step')
def hh_own(blocks, households):
    households = households.to_frame(['tenure', 'block_id'])
    households['tenure_1'] = 0
    households.loc[households['tenure'] == 1, 'tenure_1'] = 1
    tenure_1 = households.groupby('block_id').sum()
    blocks = blocks.local.copy()
    blocks = blocks.join(tenure_1).fillna(0)
    return blocks.tenure_1


@orca.column('blocks', cache=True, cache_scope='step')
def hh_rent(blocks, households):
    households = households.to_frame(['tenure', 'block_id'])
    households['tenure_2'] = 0
    households.loc[households['tenure'] == 2, 'tenure_2'] = 1
    tenure_2 = households.groupby('block_id').sum()
    blocks = blocks.local.copy()
    blocks = blocks.join(tenure_2).fillna(0)
    return blocks.tenure_2


@orca.column('blocks', cache=True, cache_scope='step')
def total_persons(blocks, households):
    persons = households.to_frame(['persons', 'block_id']).groupby('block_id').sum()
    blocks = blocks.local.join(persons).fillna(0)
    return blocks['persons']


@orca.column('blocks', cache=True, cache_scope='step')
def children(blocks, households):
    children = households.to_frame(['children', 'block_id']).groupby('block_id').sum()
    blocks = blocks.local.join(children).fillna(0)
    return blocks['children']


@orca.column('blocks', cache=True, cache_scope='step')
def persons_65plus(blocks, households):
    persons_65plus = households.to_frame(['persons_65plus', 'block_id']).groupby('block_id').sum()
    blocks = blocks.local.join(persons_65plus).fillna(0)
    return blocks['persons_65plus']


@orca.column('blocks', cache=True, cache_scope='step')
def persons_black(blocks, households):
    persons = households.to_frame(['persons_black', 'block_id']).groupby('block_id').sum()
    blocks = blocks.local.join(persons).fillna(0)
    return blocks['persons_black']


@orca.column('blocks', cache=True, cache_scope='step')
def persons_hispanic(blocks, households):
    persons = households.to_frame(['persons_hispanic', 'block_id']).groupby('block_id').sum()
    blocks = blocks.local.join(persons).fillna(0)
    return blocks['persons_hispanic']


@orca.column('blocks', cache=True, cache_scope='step')
def persons_asian(blocks, households):
    persons = households.to_frame(['persons_asian', 'block_id']).groupby('block_id').sum()
    blocks = blocks.local.join(persons).fillna(0)
    return blocks['persons_asian']


@orca.column('blocks', cache=True, cache_scope='step')
def total_units(blocks, residential_units):
    units = residential_units.local.groupby('block_id').count()
    units = pd.DataFrame(units.iloc[:,1])
    units.columns = ['total_units']
    blocks = blocks.local.join(units).fillna(0)
    return blocks['total_units']


@orca.column('blocks', 'vacant_residential_units', cache=False)
def vacant_residential_units(blocks, households):
    return blocks.total_units.sub(households.block_id.value_counts(), fill_value=0)


@orca.column('blocks', cache=True, cache_scope='step')
def density_units(blocks):
    blocks = blocks.to_frame(['total_units', 'sum_acres'])
    series = blocks.total_units * 1.0 / (blocks.sum_acres + 1.0)
    return series.fillna(0)


@orca.column('blocks', cache=True, cache_scope='step')
def density_units_90pct_plus(blocks):
    blocks = blocks.to_frame('density_units')
    percentile = blocks['density_units'].quantile(.9)
    blocks['percentile'] = 0
    blocks.loc[blocks['density_units'] >= percentile, 'percentile'] = 1
    return blocks['percentile']


@orca.column('blocks', cache=True, cache_scope='step')
def density_units_10pct_low(blocks):
    blocks = blocks.to_frame('density_units')
    percentile = blocks['density_units'].quantile(.1)
    blocks['percentile'] = 0
    blocks.loc[blocks['density_units'] <= percentile, 'percentile'] = 1
    return blocks['percentile']


@orca.column('blocks')
def units_own(blocks, residential_units, btypes_dict):
    own_btypes = [btypes_dict['sf_own'], btypes_dict['mf_own']]
    units = residential_units.to_frame(['block_id', 'building_type_id'])
    units = units.loc[units['building_type_id'].isin(own_btypes)]
    units = pd.DataFrame(units.groupby('block_id').size())
    units.columns = ['units_own']
    blocks = blocks.local.join(units).fillna(0)
    return blocks['units_own']


@orca.column('blocks')
def units_rent(blocks, residential_units, btypes_dict):
    rent_btypes = [btypes_dict['sf_rent'], btypes_dict['mf_rent']]
    units = residential_units.to_frame(['block_id', 'building_type_id'])
    units = units.loc[units['building_type_id'].isin(rent_btypes)]
    units = pd.DataFrame(units.groupby('block_id').size())
    units.columns = ['units_rent']
    blocks = blocks.local.join(units).fillna(0)
    return blocks['units_rent']


@orca.column('blocks')
def units_sf(blocks, residential_units, btypes_dict):
    sf_btypes = [btypes_dict['sf_own'], btypes_dict['sf_rent']]
    units = residential_units.to_frame(['building_type_id', 'block_id'])
    units = units.loc[units['building_type_id'].isin(sf_btypes)]
    units = pd.DataFrame(units.groupby('block_id').size())
    units.columns = ['units_sf']
    blocks = blocks.local.join(units).fillna(0)
    return blocks['units_sf']


@orca.column('blocks')
def units_mf(blocks, residential_units, btypes_dict):
    mf_btypes = [btypes_dict['mf_own'], btypes_dict['mf_rent']]
    units = residential_units.to_frame(['building_type_id', 'block_id'])
    units = units.loc[units['building_type_id'].isin(mf_btypes)]
    units = pd.DataFrame(units.groupby('block_id').size())
    units.columns = ['units_mf']
    blocks = blocks.local.join(units).fillna(0)
    return blocks['units_mf']


@orca.column('blocks')
def units_before_1930(blocks, residential_units):
    units = residential_units.to_frame(['year_built', 'block_id'])
    units = units.loc[units['year_built'] == 1930]
    units = pd.DataFrame(units.groupby('block_id').size())
    units.columns = ['units_old']
    blocks = blocks.local.join(units).fillna(0)
    return blocks['units_old']


@orca.column('blocks')
def units_after_2000(blocks, residential_units):
    units = residential_units.to_frame(['year_built', 'block_id'])
    units = units.loc[(units['year_built'] >= 2010)]
    units = pd.DataFrame(units.groupby('block_id').size())
    units.columns = ['units_new']
    blocks = blocks.local.join(units).fillna(0)
    return blocks['units_new']


@orca.column('blocks', 'du_spaces', cache=True)
def du_spaces(blocks):
    blocks_df = blocks.to_frame(['residential_unit_capacity', 'sum_acres'])
    blocks_df.loc[blocks_df['sum_acres']==0, 'residential_unit_capacity'] = 0
    du_capacity = blocks_df.residential_unit_capacity
    du_capacity = du_capacity*orca.get_injectable('capacity_boost')
    if 'proportion_undevelopable' in blocks.local_columns:
        print('Adjusting DU capacities based on proportion undevelopable.')
        return (du_capacity * (1 - blocks.proportion_undevelopable)).astype('int')
    else:
        return du_capacity


@orca.column('blocks', 'du_spaces_acre', cache=True)
def du_spaces_acre(blocks):
    df = blocks.to_frame(['du_spaces', 'sum_acres'])
    df['max_dua'] = df['du_spaces']/df['sum_acres']
    return df['max_dua'].fillna(0)


@orca.column('blocks', 'vacant_du_spaces', cache=False)
def vacant_du_spaces(blocks, residential_units):
    return blocks.du_spaces.sub(residential_units.block_id.value_counts(), fill_value=0)


@orca.column('blocks', cache=True, cache_scope='step')
def pred_rich_owned_antique(blocks):
    blocks = blocks.to_frame(['prop_income_segment_6_hh', 'prop_units_own', 'prop_units_before_1930'])
    blocks['rich_owned_antique'] = 0
    blocks_rich = blocks['prop_income_segment_6_hh'] >= blocks['prop_income_segment_6_hh'].quantile(.8)
    blocks_own = blocks['prop_units_own'] >= blocks['prop_units_own'].quantile(.8)
    blocks_antique = blocks['prop_units_before_1930'] >= blocks['prop_units_before_1930'].quantile(.8)
    blocks.loc[blocks_rich & blocks_own & blocks_antique, 'rich_owned_antique'] = 1
    return blocks['rich_owned_antique']


@orca.column('blocks', cache=True, cache_scope='step')
def pred_built_out_sf(blocks):
    blocks = blocks.to_frame(['prop_units_sf', 'density_units'])
    blocks['built_out'] = 0
    blocks_sf = blocks['prop_units_sf'] >= blocks['prop_units_sf'].quantile(.8)
    blocks_dense = blocks['density_units'] >= blocks['density_units'].quantile(.8)
    blocks.loc[blocks_sf & blocks_dense, 'built_out'] = 1
    return blocks['built_out']


@orca.column('blocks', cache=True, cache_scope='step')
def pred_built_out_mf(blocks):
    blocks = blocks.to_frame(['prop_units_mf', 'density_units'])
    blocks['built_out'] = 0
    blocks_mf = blocks['prop_units_mf'] >= blocks['prop_units_mf'].quantile(.8)
    blocks_dense = blocks['density_units'] >= blocks['density_units'].quantile(.8)
    blocks.loc[blocks_mf & blocks_dense, 'built_out'] = 1
    return blocks['built_out']


@orca.column('blocks', cache=True, cache_scope='step')
def low_density_near_dense(blocks):
    blocks = blocks.to_frame(['bg_total_units_sum_5_km_pandana', 'density_units_10pct_low', 'vacant_du_spaces'])
    blocks['near_dense'] = 0
    blocks['low_density_near_dense'] = 0
    dense_threshold = blocks['bg_total_units_sum_5_km_pandana'].quantile(.9)
    blocks.loc[blocks['bg_total_units_sum_5_km_pandana'] >= dense_threshold, 'near_dense'] = 1
    vacant = (blocks['density_units_10pct_low']==1) & (blocks['near_dense']==1) & (blocks['vacant_du_spaces']>0)
    blocks.loc[vacant, 'low_density_near_dense'] = 1
    return blocks['low_density_near_dense']


@orca.column('blocks', cache=True, cache_scope='step')
def ratio_households_to_units(blocks):
    series = (blocks.total_hh) * 1.0/(blocks.total_units + 1.0)
    return series.fillna(0)


@orca.column('blocks', cache=True, cache_scope='step')
def ratio_jobs_to_units(blocks):
    series = (blocks.total_jobs) * 1.0/(blocks.total_units + 1.0)
    return series.fillna(0)


@orca.column('blocks', cache=True, cache_scope='step')
def home_value(blocks):
    if 'pred_home_value' in blocks.columns:
        return blocks.pred_home_value
    else:
        return blocks.bg_median_value_13_acs


@orca.column('blocks', cache=True, cache_scope='step')
def home_rent(blocks):
    if 'pred_home_rent' in blocks.columns:
        return blocks.pred_home_rent
    else:
        return blocks.bg_median_rent_13_acs


# -----------------------------------------------------------------------------------------
# BLOCK GROUP VARIABLES
# -----------------------------------------------------------------------------------------


@orca.column('block_groups')
def x(block_groups, blocks):
    coords = blocks.local.groupby('block_group_id').mean().reset_index()
    coords = coords.set_index('block_group_id')[['x']]
    block_groups = block_groups.local.join(coords)
    return block_groups['x']


@orca.column('block_groups')
def y(block_groups, blocks):
    coords = blocks.local.groupby('block_group_id').mean().reset_index()
    coords = coords.set_index('block_group_id')[['y']]
    block_groups = block_groups.local.join(coords)
    return block_groups['y']


@orca.column('block_groups', cache=True, cache_scope='step')
def density_hh(block_groups):
    block_groups = block_groups.to_frame(['total_hh', 'sum_acres'])
    series = block_groups.total_hh * 1.0 / (block_groups.sum_acres + 1.0)
    return series.fillna(0)


@orca.column('block_groups', cache=True, cache_scope='step')
def density_jobs(block_groups):
    block_groups = block_groups.to_frame(['total_jobs', 'sum_acres'])
    series = block_groups.total_jobs * 1.0 / (block_groups.sum_acres + 1.0)
    return series.fillna(0)


@orca.column('block_groups', cache=True, cache_scope='step')
def density_units(block_groups):
    block_groups = block_groups.to_frame(['total_units', 'sum_acres'])
    series = block_groups.total_units * 1.0 / (block_groups.sum_acres + 1.0)
    return series.fillna(0)


@orca.column('block_groups', cache=True, cache_scope='step')
def ratio_households_to_units(block_groups):
    series = (block_groups.total_hh) * 1.0/(block_groups.total_units + 1.0)
    return series.fillna(0)


@orca.column('block_groups', cache=True, cache_scope='step')
def ratio_jobs_to_units(block_groups):
    series = (block_groups.total_jobs) * 1.0/(block_groups.total_units + 1.0)
    return series.fillna(0)


@orca.column('block_groups', cache=True, cache_scope='step')
def predominant_building_type(blocks, block_groups, residential_units):
    block_groups = block_groups.local.copy()
    blocks = blocks.to_frame(['block_group_id']).reset_index()
    units = residential_units.to_frame(['block_id', 'building_type'])
    units = units.merge(blocks, on='block_id', how='left')
    for building_type in units.building_type.unique():
        units_type = units[units['building_type']==building_type].groupby('block_group_id').size()
        block_groups[building_type] = units_type
        block_groups[building_type] = block_groups[building_type].fillna(0)
    block_groups['max_col'] = block_groups[units.building_type.unique()].idxmax(axis=1)
    return block_groups['max_col']


@orca.column('block_groups', cache=True, cache_scope='step')
def mean_income(block_groups, blocks, households):
    blocks = blocks.to_frame(['block_group_id']).reset_index()
    hh = households.to_frame(['income', 'block_id'])
    hh = hh.merge(blocks, on='block_id', how='left')
    mean_income = hh.groupby('block_group_id').mean()
    block_groups = block_groups.local.copy()
    block_groups['mean_income'] = mean_income
    return block_groups['mean_income'].fillna(block_groups['mean_income'].median())


@orca.column('block_groups', cache=True, cache_scope='step')
def mean_hh_size(block_groups, blocks, households):
    blocks = blocks.to_frame(['block_group_id']).reset_index()
    hh = households.to_frame(['persons', 'block_id'])
    hh = hh.merge(blocks, on='block_id', how='left')
    mean_size = hh.groupby('block_group_id').mean()
    block_groups = block_groups.local.copy()
    block_groups['mean_size'] = mean_size
    return block_groups['mean_size'].fillna(block_groups['mean_size'].median())


@orca.column('block_groups', cache=True, cache_scope='step')
def median_value_13_acs(block_groups, values):
    values = values.local.set_index('block_group_id')
    block_groups = block_groups.local.copy()
    block_groups = block_groups.join(values)
    return block_groups['ACS_13_value']


@orca.column('block_groups', cache=True, cache_scope='step')
def median_rent_13_acs(block_groups, values):
    values = values.local.set_index('block_group_id')
    block_groups = block_groups.local.copy()
    block_groups = block_groups.join(values)
    return block_groups['ACS_13_rent']


@orca.column('block_groups', cache=True, cache_scope='step')
def mean_home_value(blocks, block_groups):
    blocks = blocks.to_frame(['home_value', 'block_group_id'])
    values = pd.DataFrame(blocks.groupby('block_group_id').home_value.mean())
    block_groups = block_groups.local.join(values)
    return block_groups.home_value


@orca.column('block_groups', cache=True, cache_scope='step')
def mean_home_rent(blocks, block_groups):
    blocks = blocks.to_frame(['home_rent', 'block_group_id'])
    rents = pd.DataFrame(blocks.groupby('block_group_id').home_rent.mean())
    block_groups = block_groups.local.join(rents)
    return block_groups.home_rent


@orca.column('block_groups', cache=True, cache_scope='step')
def mean_year_built(block_groups, blocks, residential_units):
    blocks = blocks.to_frame(['block_group_id']).reset_index()
    units = residential_units.to_frame(['year_built', 'block_id'])
    units = units.merge(blocks, on='block_id', how='left')
    mean_year = units.groupby('block_group_id').mean()
    block_groups = block_groups.local.copy()
    block_groups['mean_year'] = mean_year
    return block_groups['mean_year'].fillna(block_groups['mean_year'].median())


@orca.column('block_groups', cache=True, cache_scope='step')
def mean_workers(block_groups, blocks, households):
    blocks = blocks.to_frame(['block_group_id']).reset_index()
    hh = households.to_frame(['workers', 'block_id'])
    hh = hh.merge(blocks, on='block_id', how='left')
    mean_workers = hh.groupby('block_group_id').mean()
    block_groups = block_groups.local.copy()
    block_groups['mean_workers'] = mean_workers
    return block_groups['mean_workers'].fillna(block_groups['mean_workers'].median())


@orca.column('block_groups', cache=True, cache_scope='step')
def mean_children(block_groups, blocks, households):
    blocks = blocks.to_frame(['block_group_id']).reset_index()
    hh = households.to_frame(['children', 'block_id'])
    hh = hh.merge(blocks, on='block_id', how='left')
    mean_children = hh.groupby('block_group_id').mean()
    block_groups = block_groups.local.copy()
    block_groups['mean_children'] = mean_children
    return block_groups['mean_children'].fillna(block_groups['mean_children'].median())


@orca.column('block_groups', cache=True, cache_scope='step')
def mean_age_of_head(block_groups, blocks, households):
    blocks = blocks.to_frame(['block_group_id']).reset_index()
    hh = households.to_frame(['age_of_head', 'block_id'])
    hh = hh.merge(blocks, on='block_id', how='left')
    mean_age_of_head = hh.groupby('block_group_id').mean()
    block_groups = block_groups.local.copy()
    block_groups['mean_age_of_head'] = mean_age_of_head
    return block_groups['mean_age_of_head'].fillna(block_groups['mean_age_of_head'].median())



# -----------------------------------------------------------------------------------------
# TRACT VARIABLES
# -----------------------------------------------------------------------------------------


@orca.column('tracts')
def x(tracts, blocks):
    coords = blocks.local.groupby('tract_id').mean().reset_index()
    coords = coords.set_index('tract_id')[['x']]
    tracts = tracts.local.join(coords)
    return tracts['x']


@orca.column('tracts')
def y(tracts, blocks):
    coords = blocks.local.groupby('tract_id').mean().reset_index()
    coords = coords.set_index('tract_id')[['y']]
    tracts = tracts.local.join(coords)
    return tracts['y']


@orca.column('tracts', cache=True, cache_scope='step')
def county_id(tracts):
    return tracts.local.index.str.slice(0,5)


@orca.column('tracts', cache=True, cache_scope='step')
def mean_income(tracts, blocks, households):
    blocks = blocks.to_frame(['tract_id']).reset_index()
    hh = households.to_frame(['income', 'block_id'])
    hh = hh.merge(blocks, on='block_id', how='left')
    mean_income = hh.groupby('tract_id').mean()
    tracts = tracts.local.copy()
    tracts['mean_income'] = mean_income
    return tracts['mean_income'].fillna(tracts['mean_income'].median())


@orca.column('tracts', cache=True, cache_scope='step')
def mean_hh_size(tracts, blocks, households):
    blocks = blocks.to_frame(['tract_id']).reset_index()
    hh = households.to_frame(['persons', 'block_id'])
    hh = hh.merge(blocks, on='block_id', how='left')
    mean_size = hh.groupby('tract_id').mean()
    tracts = tracts.local.copy()
    tracts['mean_size'] = mean_size
    return tracts['mean_size'].fillna(tracts['mean_size'].median())


@orca.column('tracts', cache=True, cache_scope='step')
def mean_age_of_head(tracts, blocks, households):
    blocks = blocks.to_frame(['tract_id']).reset_index()
    hh = households.to_frame(['age_of_head', 'block_id'])
    hh = hh.merge(blocks, on='block_id', how='left')
    mean_age_of_head = hh.groupby('tract_id').mean()
    tracts = tracts.local.copy()
    tracts['mean_age_of_head'] = mean_age_of_head
    return tracts['mean_age_of_head'].fillna(tracts['mean_age_of_head'].median())


@orca.column('tracts', cache=True, cache_scope='step')
def std_income(tracts, blocks, households):
    blocks = blocks.to_frame(['tract_id']).reset_index()
    hh = households.to_frame(['income', 'block_id'])
    hh = hh.merge(blocks, on='block_id', how='left')
    std_income = hh.groupby('tract_id').std()
    tracts = tracts.local.copy()
    tracts['std_income'] = std_income
    return tracts['std_income']#.fillna(tracts['std_income'].median())


@orca.column('tracts', cache=True, cache_scope='step')
def density_hh(tracts, blocks):
    tract_data = blocks.to_frame(['total_hh', 'sum_acres', 'tract_id']).groupby('tract_id').sum()
    tracts = tracts.local.join(tract_data)
    return (tracts['total_hh']/tracts['sum_acres']).fillna(0)


@orca.column('tracts', cache=True, cache_scope='step')
def density_units(tracts, blocks):
    tract_data = blocks.to_frame(['total_units', 'sum_acres', 'tract_id']).groupby('tract_id').sum()
    tracts = tracts.local.join(tract_data)
    return (tracts['total_units']/tracts['sum_acres']).fillna(0)


@orca.column('tracts', cache=True, cache_scope='step')
def density_jobs(tracts, blocks):
    tract_data = blocks.to_frame(['total_jobs', 'sum_acres', 'tract_id']).groupby('tract_id').sum()
    tracts = tracts.local.join(tract_data)
    return (tracts['total_jobs'] / tracts['sum_acres']).fillna(0)


@orca.column('tracts', cache=True, cache_scope='step')
def mean_year_built(tracts, blocks, residential_units):
    blocks = blocks.to_frame(['tract_id']).reset_index()
    units = residential_units.to_frame(['year_built', 'block_id'])
    units = units.merge(blocks, on='block_id', how='left')
    mean_year = units.groupby('tract_id').mean()
    tracts = tracts.local.copy()
    tracts['mean_year'] = mean_year
    return tracts['mean_year'].fillna(tracts['mean_year'].median())


@orca.column('tracts', cache=True, cache_scope='step')
def prop_children(tracts):
    return (tracts.children / tracts.total_persons).fillna(0)


# -----------------------------------------------------------------------------------------
# COUNTY VARIABLES
# -----------------------------------------------------------------------------------------


@orca.column('counties', cache=True, cache_scope='step')
def mean_income(counties, blocks, households):
    blocks = blocks.to_frame(['county_id']).reset_index()
    hh = households.to_frame(['income', 'block_id'])
    hh = hh.merge(blocks, on='block_id', how='left')
    mean_income = hh.groupby('county_id').mean()
    counties = counties.local.copy()
    counties['mean_income'] = mean_income
    return counties['mean_income'].fillna(counties['mean_income'].median())


@orca.column('counties', cache=True, cache_scope='step')
def mean_age_of_head(counties, blocks, households):
    blocks = blocks.to_frame(['county_id']).reset_index()
    hh = households.to_frame(['age_of_head', 'block_id'])
    hh = hh.merge(blocks, on='block_id', how='left')
    mean_age_of_head = hh.groupby('county_id').mean()
    counties = counties.local.copy()
    counties['mean_age_of_head'] = mean_age_of_head
    return counties['mean_age_of_head'].fillna(counties['mean_age_of_head'].median())


@orca.column('counties', cache=True, cache_scope='step')
def mean_home_value(blocks, block_groups, counties):
    if 'pred_home_value' in blocks.columns:
        blocks = blocks.to_frame(['home_value', 'county_id'])
        values = pd.DataFrame(blocks.groupby('county_id').home_value.mean())
        counties = counties.local.join(values)
        return counties.home_value
    else:
        block_groups = block_groups.to_frame(['median_value_13_acs'])
        block_groups['county_id'] = block_groups.index.str.slice(0,5)
        counties = counties.local.join(block_groups.groupby('county_id').median_value_13_acs.mean())
        return counties.median_value_13_acs


@orca.column('counties', cache=True, cache_scope='step')
def mean_home_rent(blocks, block_groups, counties):
    if 'pred_home_rent' in blocks.columns:
        blocks = blocks.to_frame(['home_rent', 'county_id'])
        rents = pd.DataFrame(blocks.groupby('county_id').home_rent.mean())
        counties = counties.local.join(rents)
        return counties.home_rent
    else:
        block_groups = block_groups.to_frame(['median_rent_13_acs'])
        block_groups['county_id'] = block_groups.index.str.slice(0,5)
        counties = counties.local.join(block_groups.groupby('county_id').median_rent_13_acs.mean())
        return counties.median_rent_13_acs

# -----------------------------------------------------------------------------------------
# TRAVEL DATA VARIABLES
# -----------------------------------------------------------------------------------------
####################
### HELPER FUNCS ###
####################
def skims_lookup(matrix, origin, destination):

    assert origin.shape == destination.shape

    # Convert zone to int for skim look up
    o = origin.astype(int) - 1
    d = destination.astype(int) - 1

    s = pd.Series(data = matrix[o, d],
                  index = [o + 1,d + 1])

    # Convert zone to str for data type consistency
    s.index = s.index.set_levels([s.index.levels[0].astype(str),
                                  s.index.levels[1].astype(str)])

    return s

def read_yaml_file(file_path):
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)
    return data


@orca.column('travel_data', cache=True)
def ASC():
    return 1

@orca.column('travel_data', cache=True)
def tour_sov_in_vehicle_time(travel_data, asim_skims):
    ods = travel_data.to_frame(columns = ['']).reset_index()

    ['from_zone_id', 'to_zone_id']

    #Outbound
    outbound = np.array(asim_skims['SOV_TIME__AM'])
    value_outbound = skims_lookup(outbound, ods.from_zone_id, ods.to_zone_id)

    #Inbound
    inbound = np.array(asim_skims['SOV_TIME__PM'])
    value_inbound = skims_lookup(inbound, ods.to_zone_id, ods.from_zone_id)

    return (value_outbound + value_inbound).apply(np.log1p)

@orca.column('travel_data', cache=True)
def tour_dist(travel_data, asim_skims):

    ods = travel_data.to_frame(columns = ['']).reset_index()

    #Outbound
    outbound = np.array(asim_skims['DIST'])
    value_outbound = skims_lookup(outbound, ods.from_zone_id, ods.to_zone_id)

    #Inbound
    inbound = np.array(asim_skims['DIST'])
    value_inbound = skims_lookup(inbound, ods.to_zone_id, ods.from_zone_id)

    return (value_outbound + value_inbound)


@orca.column('travel_data')
def tour_sov_operating_cost(travel_data, cost_per_mile, avg_parking_cost):

    s = travel_data.tour_dist * cost_per_mile + avg_parking_cost * 100 # PK CST dollars to cents

    return (s).apply(np.log1p)


@orca.column('travel_data', cache=True)
def tour_bus_in_vehicle_time(travel_data, asim_skims, transit_change):

    ods = travel_data.to_frame(columns = ['']).reset_index()

    #Outbound
    outbound = (np.array(asim_skims['WLK_LOC_WLK_TOTIVT__AM'])/100) * transit_change
    value_outbound = skims_lookup(outbound, ods.from_zone_id, ods.to_zone_id)

    #Inbound
    inbound = (np.array(asim_skims['WLK_LOC_WLK_TOTIVT__PM'])/100) * transit_change
    value_inbound = skims_lookup(inbound, ods.to_zone_id, ods.from_zone_id)

    return (value_outbound + value_inbound).apply(np.log1p).replace(0, 20.0) #Replace high high time to make transit very unactrative

@orca.column('travel_data', cache=True)
def tour_bus_fare(travel_data, asim_skims):
    ods = travel_data.to_frame(columns = ['']).reset_index()

    #Outbound
    outbound = np.array(asim_skims['WLK_LOC_WLK_FAR__AM'])
    value_outbound = skims_lookup(outbound, ods.from_zone_id, ods.to_zone_id)

    #Inbound
    inbound = np.array(asim_skims['WLK_LOC_WLK_FAR__PM'])
    value_inbound = skims_lookup(inbound, ods.to_zone_id, ods.from_zone_id)

    return (value_outbound + value_inbound).apply(np.log1p).replace(0, 20.0)

@orca.column('travel_data', cache=True)
def tour_train_in_vehicle_time(travel_data, asim_skims, transit_change):

    ods = travel_data.to_frame(columns = ['']).reset_index()

    #Outbound
    outbound = (np.array(asim_skims['WLK_HVY_WLK_TOTIVT__AM'])/100) * transit_change
    value_outbound = skims_lookup(outbound, ods.from_zone_id, ods.to_zone_id)

    #Inbound
    inbound = (np.array(asim_skims['WLK_HVY_WLK_TOTIVT__PM'])/100) * transit_change
    value_inbound = skims_lookup(inbound, ods.to_zone_id, ods.from_zone_id)

    return (value_outbound + value_inbound).apply(np.log1p).replace(0, 20.0) #Replace high high time to make transit very unactrative

@orca.column('travel_data', cache=True)
def tour_train_fare(travel_data, asim_skims):
    ods = travel_data.to_frame(columns = ['']).reset_index()

    #Outbound
    outbound = np.array(asim_skims['WLK_HVY_WLK_FAR__AM'])
    value_outbound = skims_lookup(outbound, ods.from_zone_id, ods.to_zone_id)

    #Inbound
    inbound = np.array(asim_skims['WLK_HVY_WLK_FAR__PM'])
    value_inbound = skims_lookup(inbound, ods.to_zone_id, ods.from_zone_id)

    return (value_outbound + value_inbound).apply(np.log1p).replace(0, 20.0)

@orca.column('travel_data', cache=True)
def walk_time_up_to_2_miles(travel_data, asim_skims, walkThresh, walkSpeed):
    ods = travel_data.to_frame(columns = ['']).reset_index()

    #Outbound
    outbound = np.array(asim_skims['DISTWALK'])
    value_outbound = skims_lookup(outbound, ods.from_zone_id, ods.to_zone_id).clip(upper = walkThresh)

    #Inbound
    inbound = np.array(asim_skims['DISTWALK'])
    value_inbound = skims_lookup(inbound, ods.to_zone_id, ods.from_zone_id).clip(upper = walkThresh)

    return ((value_outbound + value_inbound) * 60/walkSpeed).apply(np.log1p)

@orca.column('travel_data', cache=True)
def walk_time_beyond_2_of_a_miles(travel_data, asim_skims, walkThresh, walkSpeed):

    ods = travel_data.to_frame(columns = ['']).reset_index()

    #Outbound
    outbound = np.array(asim_skims['DISTWALK'])
    value_outbound = skims_lookup(outbound, ods.from_zone_id, ods.to_zone_id)
    value_outbound = (value_outbound -  walkThresh).clip(lower = 0)

    #Inbound
    inbound = np.array(asim_skims['DISTWALK'])
    value_inbound = skims_lookup(inbound, ods.to_zone_id, ods.from_zone_id)
    value_inbound = (value_inbound - walkThresh).clip(lower = 0)

    return ((value_outbound + value_inbound) * 60/walkSpeed).apply(np.log1p)

@orca.column('travel_data', cache=True)
def bike_time_up_to_6_miles(travel_data, asim_skims, bikeThresh, bikeSpeed):
    ods = travel_data.to_frame(columns = ['']).reset_index()

    #Outbound
    outbound = np.array(asim_skims['DISTBIKE'])
    value_outbound = skims_lookup(outbound, ods.from_zone_id, ods.to_zone_id).clip(upper = bikeThresh)

    #Inbound
    inbound = np.array(asim_skims['DISTBIKE'])
    value_inbound = skims_lookup(inbound, ods.to_zone_id, ods.from_zone_id).clip(upper = bikeThresh)

    return ((value_outbound + value_inbound) * 60/bikeSpeed).apply(np.log1p)

@orca.column('travel_data', cache=True)
def bike_time_beyond_6_of_a_miles(travel_data, asim_skims, bikeThresh, bikeSpeed):
    ods = travel_data.to_frame(columns = ['']).reset_index()

    #Outbound
    outbound = np.array(asim_skims['DISTBIKE'])
    value_outbound = skims_lookup(outbound, ods.from_zone_id, ods.to_zone_id)
    value_outbound = (value_outbound -  bikeThresh).clip(lower = 0)

    #Inbound
    inbound = np.array(asim_skims['DISTBIKE'])
    value_inbound = skims_lookup(inbound, ods.to_zone_id, ods.from_zone_id).clip(upper = 2)
    value_inbound = (value_inbound - bikeThresh).clip(lower = 0)

    return ((value_outbound + value_inbound) * 60/bikeSpeed).apply(np.log1p)

@orca.column('travel_data', cache=True)
def outbound_dist(travel_data, asim_skims):

    ods = travel_data.to_frame(columns = ['']).reset_index()

    #Outbound
    outbound = np.array(asim_skims['DIST'])
    value_outbound = skims_lookup(outbound, ods.from_zone_id, ods.to_zone_id)

    return value_outbound

@orca.column('travel_data', cache=True)
def inbound_dist(travel_data, asim_skims):

    ods = travel_data.to_frame(columns = ['']).reset_index()

    #Inbound
    inbound = np.array(asim_skims['DIST'])
    value_inbound = skims_lookup(inbound, ods.to_zone_id, ods.from_zone_id)

    return value_inbound

@orca.column('travel_data', cache=True)
def tour_tnc_cost(travel_data, asim_skims, tnc_baseline, tnc_cost_minute, tnc_cost_mile, tnc_min_fare):
    ods = travel_data.to_frame(columns = ['outbound_dist', 'inbound_dist']).reset_index()

    #Outbound
    outbound = np.array(asim_skims['SOV_TIME__AM'])
    time_outbound = skims_lookup(outbound, ods.from_zone_id, ods.to_zone_id)
    dist_outbound = ods.set_index(['from_zone_id', 'to_zone_id']).outbound_dist
    cost_outbound = tnc_baseline + tnc_cost_minute*time_outbound + tnc_cost_mile*dist_outbound
    cost_outbound = cost_outbound.where(cost_outbound >= tnc_min_fare, tnc_min_fare) * 100 #from dollars to cents

    #Inbound
    inbound = np.array(asim_skims['SOV_TIME__AM'])
    time_inbound = skims_lookup(inbound, ods.from_zone_id, ods.to_zone_id)
    dist_inbound = ods.set_index(['from_zone_id', 'to_zone_id']).inbound_dist
    cost_inbound = tnc_baseline + tnc_cost_minute*time_inbound + tnc_cost_mile*dist_inbound
    cost_inbound = cost_inbound.where(cost_inbound >= tnc_min_fare, tnc_min_fare) * 100 #from dollars to cents

    return (cost_outbound + cost_inbound).apply(np.log1p)

@orca.column('travel_data', cache=True)
def tour_tnc_wait_time(travel_data, asim_skims):
    ods = travel_data.to_frame(columns = ['']).reset_index()

    #Outbound
    outbound = np.array(asim_skims['RH_SOLO_WAIT__AM'])
    value_outbound = skims_lookup(outbound, ods.from_zone_id, ods.to_zone_id)

    #Inbound
    inbound = np.array(asim_skims['RH_SOLO_WAIT__PM'])
    value_inbound = skims_lookup(inbound, ods.to_zone_id, ods.from_zone_id)

    return (value_outbound + value_inbound).apply(np.log1p)

@orca.column('travel_data', cache=True)
def dest_cbd(travel_data, zones):
    ods = travel_data.to_frame(columns = ['']).reset_index()
    df = zones.to_frame(columns = ['cbd'])

    s = ods.merge(df, how = 'left', left_on = 'to_zone_id', right_index = True)
    s = s.set_index(['from_zone_id', 'to_zone_id'])
    return s

@orca.column('travel_data', cache = True)
def dest_employment_density(travel_data, zones):
    ods = travel_data.to_frame(columns = ['']).reset_index()
    df = zones.to_frame(columns = ['employment_density'])

    s = ods.merge(df, how = 'left', left_on = 'to_zone_id', right_index = True)
    s = s.set_index(['from_zone_id', 'to_zone_id'])
    return s.apply(np.log1p)

########################
## AUXILIARY COLUMNS ##
########################

@orca.column('households', cache=True)
def home_taz(households, blocks):
    return misc.reindex(blocks['taz_zone_id'], households['block_id']).astype(str)

@orca.column('jobs', cache=True)
def taz(jobs, blocks):
    return misc.reindex(blocks['taz_zone_id'], jobs['block_id']).astype(str)

@orca.column('zones', cache=True)
def totpop(households):
    s = households.to_frame(columns = ['home_taz', 'persons'])
    return s.groupby('home_taz')['persons'].sum().fillna(0)

@orca.column('zones', cache=True)
def totemp(jobs):
    s = jobs.to_frame(columns = ['taz', 'block_id'])
    return s.groupby('taz').size().fillna(0)

@orca.column('zones', cache=True)
def totacre(blocks):
    s = blocks.to_frame(columns = ['taz_zone_id','sum_acres'])
    return s.groupby('taz_zone_id')['sum_acres'].sum().fillna(0)

@orca.column('zones', cache=True)
def density(zones):
    df = zones.to_frame(columns = ['totpop','totemp','totacre'])
    return (df['totpop'] + 2.5 * df['totemp'])/df['totacre']

@orca.column('zones', cache=True)
def area_type(zones):
    # Integer, 0=regional core, 1=central business district,
    # 2=urban business, 3=urban, 4=suburban, 5=rural
    area_types = pd.cut(
        zones['density'],
        [0, 6, 30, 55, 100, 300, float("inf")],
        labels=['5', '4', '3', '2', '1', '0'],
        include_lowest=True).astype(str)
    return area_types

@orca.column('zones', cache=True)
def cbd(zones):
    return zones.area_type.isin(['0', '1']).astype(int)


@orca.column('zones')
def employment_density(zones):
    return zones.totemp / zones.totacre
# @orca.column('zones', cache=True)
# def county_id(zones):
#
#     #FIX ME: Consider generalizing this for other regions
#
#     county_map = {'Alameda': '06001',
#                   'Contra Costa': '06013',
#                   'Marin': '06041',
#                   'Napa': '06055',
#                   'San Francisco': '06075' ,
#                   'San Mateo':'06081',
#                   'Santa Clara':'06085',
#                   'Solano': '06095',
#                   'Sonoma':'06097'}
#
#     return zones.county.replace(county_map).astype(str)

# @orca.column('travel_data', cache=True)
# def county_id(travel_data, zones):
#     td = travel_data.local.reset_index()
#
#     series = misc.reindex(zones['county_id'], td['from_zone_id'].astype(str))
#     series.name = 'county_id'
#
#     df = pd.DataFrame(data = series)
#     df['from_zone_id'] = td['from_zone_id'].astype(int)
#     df['to_zone_id'] = td['to_zone_id'].astype(int)
#     df = df.set_index(['from_zone_id','to_zone_id'])
#
#     return df.county_id

# @orca.column('travel_data', cache=True)
# def county_cat(travel_data):
#     s = travel_data.county_id
#     return pd.Series(pd.factorize(s)[0], index = s.index)

@orca.column('persons', cache=True)
def home_taz(households, persons):
    return misc.reindex(households.home_taz, persons.household_id)

@orca.column('job_flows', cache=True)
def from_zone_id(job_flows, blocks):
    return misc.reindex( blocks.taz_zone_id, job_flows.home_block_id)

@orca.column('job_flows', cache=True)
def to_zone_id(job_flows, blocks):
    return misc.reindex( blocks.taz_zone_id, job_flows.work_block_id)

# @orca.column('travel_data', cache=True)
# def job_flows(travel_data, job_flows):
#
#     # Job flows by TAZ
#     jf = job_flows.to_frame(columns = ['from_zone_id', 'to_zone_id','job_flows'])
#     jf = jf.groupby(['from_zone_id', 'to_zone_id']).agg({'job_flows':'sum'}).reset_index()
#
#     # Load all possible OD pairs
#     td = travel_data.local.reset_index().astype(str)
#
#     # Merge flows with OD pairs, fill with zero if OD flow is NaN
#     flows = td.merge(jf, how = 'left', on = ['from_zone_id','to_zone_id']).fillna(0)
#     flows[['from_zone_id','to_zone_id']] = flows[['from_zone_id','to_zone_id']].astype(int)
#     return flows.set_index(['from_zone_id','to_zone_id'])['job_flows']

# @orca.injectable()
# def county_cat_map(travel_data, cache=True):
#     s = travel_data.to_frame(columns = ['county_id']).county_id
#     dict_ = {}
#     for i, value in enumerate(pd.factorize(s)[1]):
#         dict_[value] = i
#     return dict_

@orca.injectable()
def mode_choice_template(region_code, calibrated_folder):
    # region_code = orca.get_injectable("region_code")
    # calibrated_folder = orca.get_injectable("calibrated_folder")
    calibrated_path = os.path.join(
        'configs',
        'calibrated_configs/',
        calibrated_folder,
        region_code,
        'mode_choice',
        'mode_choice_logsum.yaml')
    # output_file = f"configs/mode_choice/mode_choice_logsum.yaml"
    return read_yaml_file(calibrated_path)

@orca.column('travel_data', cache = 'iteration')
def logsum(mode_choice_template, travel_data):

    # Read Mode Choice Specs
    coeffs = mode_choice_template['saved_object']['fitted_parameters']
    specs = OrderedDict(mode_choice_template['saved_object']['model_expression'])
    nest = mode_choice_template['saved_object']['nest']

    # Get list of explanatory Variables
    exp_vars = list(set([key for values in specs.values() for key in values.keys()]))

    # Read Data from Orca
    t = travel_data.to_frame(columns = exp_vars)

    # Organize Data in Dictionary - (mode, data)
    x = {mode: np.array([t[key] for key in spec.keys()]) for mode, spec in specs.items()}

    # Iterate over the list of dictionaries and convert list values to NumPy arrays
    # (for alternative indexing)
    for d in nest['alternatives']:
        for key, value in d.items():
            if isinstance(value, list):
                d[key] = np.array(value).astype(int)


    logsums_ = mode_choice.mode_choice_logsums(params = coeffs,
                             specs = specs,
                             exp_vars = x,
                             nest = nest)


    return pd.Series(logsums_, index = t.index)

# -----------------------------------------------------------------------------------------
# DERIVED VARIABLES
# -----------------------------------------------------------------------------------------

# TODO: Rremove this variable from specifications, keep prop_hh_rent
@orca.column('block_groups', cache=False)
def prop_households_rent(block_groups):
    return block_groups.prop_hh_rent


def register_jobs_sector(sector, table):
    @orca.column(table, 'jobs_' + sector, cache=True, cache_scope='iteration')
    def column_func(blocks, jobs):
        blocks = blocks.local.copy()
        jobs = jobs.local.copy()
        jobs = jobs[jobs['agg_sector'] == sector]
        jobs = pd.DataFrame(jobs.groupby('block_id').size())
        jobs.columns = ['agg_jobs']
        df = blocks.join(jobs).fillna(0)
        if table != 'blocks':
            df = df.groupby(table.replace('s', '_id')).sum()
        return df['agg_jobs']
    return column_func


def register_prop_jobs_sector(sector, table):
    @orca.column(table, 'prop_jobs_' + sector, cache=True, cache_scope='iteration')
    def column_func(blocks):
        blocks = blocks.to_frame(list(blocks.local.columns) + ['jobs_' + sector, 'total_jobs'])
        df = blocks.groupby(table.replace('s', '_id')).sum()
        return (df['jobs_' + sector] / df['total_jobs']).replace([np.inf, -np.inf], np.nan).fillna(0)
    return column_func


def register_units_building_type(building_type, table):
    @orca.column(table, 'units_' + building_type, cache=True, cache_scope='iteration')
    def column_func(blocks, residential_units):
        blocks = blocks.local.copy()
        units = residential_units.to_frame(['building_type', 'block_id'])
        units = units[units['building_type'] == building_type]
        units = pd.DataFrame(units.groupby('block_id').size())
        units.columns = ['agg_units']
        df = blocks.join(units).fillna(0)
        if table != 'blocks':
            df = df.groupby(table.replace('s', '_id')).sum()
        return df['agg_units']
    return column_func


def register_predominant_building_type_cat(building_type):
    @orca.column('block_groups', 'predominant_is_' + building_type, cache=True, cache_scope='iteration')
    def column_func(block_groups):
        bg = block_groups.to_frame(['predominant_building_type'])
        bg['is_most_common'] = 0
        bg.loc[bg['predominant_building_type'] == building_type, 'is_most_common'] = 1
        return bg['is_most_common']
    return column_func


def register_sub_skim(threshold, impedance_column, units, sub_skim_name):
    travel_data = orca.get_table('travel_data').to_frame().reset_index(level=1)
    # print(travel_data.columns)
    # print(travel_data.index)
    if units == 'km':
        threshold = threshold * 0.621
    travel_data = travel_data[travel_data[impedance_column] < threshold]
    orca.add_injectable(sub_skim_name, travel_data)


def register_skim_var(table_name, column_name, threshold, var, impedance_column, agg_func, units):
    @orca.column(table_name, column_name, cache=True, cache_scope='iteration')
    def column_func():
        print('Calculating {} of {} within {} {} based on {} from skim'.format(agg_func, var, units, threshold, impedance_column))
        if units == 'km':
            sub_skim_name = 'travel_data_' + str(threshold) + 'km_' + impedance_column
        else:
            sub_skim_name = 'travel_data_' + str(threshold) + 'min_' + impedance_column
        if sub_skim_name not in orca.list_injectables():
                register_sub_skim(threshold, impedance_column, units, sub_skim_name)
        elif 'skims' in orca.list_injectables():
            skims = orca.get_injectable('skims')
            year = int(orca.get_injectable('year'))
            skims['year'] = skims['year'].astype(int)
            if year in skims['year'].unique():
                register_sub_skim(threshold, impedance_column, units, sub_skim_name)
        travel_data = orca.get_injectable(sub_skim_name)
        zones_table = orca.get_table(table_name).to_frame(var)
        zones_table.index.names = ['zone_id']
        zones_table = zones_table.reset_index()
        # # travel_data["to_zone_id"] = travel_data["to_zone_id"].astype(int) ## Austin vs Bay Area specific, data types need fixing
        # print(travel_data.dtypes)
        # print(zones_table.dtypes)
        travel_data = travel_data.reset_index().merge(zones_table, how='left', left_on='to_zone_id', right_on='zone_id')
        travel_data = travel_data.set_index(['from_zone_id'])
        travel_data[var] = travel_data[var].fillna(0)
        return travel_data.groupby(level=0)[var].apply(eval('np.' + agg_func))
    return column_func


def register_pandana_access_variable(column_name, onto_table, variable_to_summarize,
                                     distance, agg_type='sum', decay='linear', log=False):
    @orca.column(onto_table, column_name, cache=True, cache_scope='iteration')
    def column_func():
        print('Calculating {} of {} within {} km based on pandana network'.format(agg_type, variable_to_summarize, distance/1000))
        net = orca.get_injectable('net')
        table = orca.get_table(onto_table).to_frame(['node_id', variable_to_summarize])
        net.set(table.node_id,  variable=table[variable_to_summarize])
        results = net.aggregate(distance, type=agg_type, decay=decay)
        if log:
            results = results.apply(eval('np.log1p'))
        return misc.reindex(results, table.node_id)
    return column_func


def register_disag_var(table_from, table_to, column_name, prefix=True):
    if prefix == True:
        disag_col_name = table_from + '_' + column_name
        if table_from == 'block_groups':
            disag_col_name = 'bg_' + column_name
    else:
        disag_col_name = column_name
    @orca.column(table_to, disag_col_name, cache=True, cache_scope='iteration')
    def column_func():
        # print(column_name)
        # print(table_from)
        from_df = orca.get_table(table_from).to_frame(column_name)
        from_idx = from_df.index.name
        to_df = orca.get_table(table_to).to_frame(from_idx)
        to_idx = to_df.index.name
        to_df = to_df.reset_index().merge(from_df.reset_index(), on=from_idx, how='left')
        to_df = to_df.set_index(to_idx)
        return to_df[column_name].fillna(0)
    return column_func


def register_agg_var(table_from, table_to, column_name, agg_type, prefix=True):
    if prefix == True:
        if table_from == 'block_groups':
            var_name = 'bg_' + '_' + column_name
        else:
            var_name = table_from  + '_' + column_name
    else:
        var_name = column_name
    @orca.column(table_to, var_name, cache=True, cache_scope='iteration')
    def column_func():
        to_df = orca.get_table(table_to).local.copy()
        to_idx = to_df.index.name
        from_df = orca.get_table(table_from).to_frame([column_name, to_idx])
        from_df = from_df.groupby(to_idx).agg(agg_type)
        to_df = to_df.join(from_df[[column_name]])
        return to_df[column_name].fillna(0)
    return column_func


def register_prop_variable(table_name, agents_name, col):
    @orca.column(table_name, 'prop_' + col, cache=True, cache_scope='iteration')
    def column_func():
        totals_col = 'total_' + agents_name
        totals_col = totals_col.replace('households', 'hh').replace('residential_', '')
        df = orca.get_table(table_name).to_frame([col, totals_col])
        return (df[col]/df[totals_col]).fillna(0)
    return column_func


def register_ln_variable(table_name, col):
    @orca.column(table_name, 'ln_' + col, cache=True, cache_scope='iteration')
    def column_func():
        return np.log1p(orca.get_table(table_name)[col])
    return column_func


def register_standardized_variable(table_name, col):
    @orca.column(table_name, 'st_' + col, cache=True, cache_scope='iteration')
    def column_func():
        # print("here", str(table_name))
        # print('st_' + col)
        # print(table_name)
        # print(col)
        df = orca.get_table(table_name).to_frame(col)
        df['st_col'] = (df[col] - df[col].mean())/df[col].std()
        return df['st_col'].fillna(1)
    return column_func


def register_geog_dummy(table_name, geog):
    @orca.column(table_name, 'county_id_is_' + geog, cache=True, cache_scope='iteration')
    def column_func():
        df = orca.get_table(table_name).to_frame('county_id')
        df['county_id_is_geog'] = 0
        df.loc[df['county_id']==geog,'county_id_is_geog'] = 1
        return df['county_id_is_geog']
    return column_func


for sector in orca.get_table('jobs').local.agg_sector.unique():
    for geo in ['blocks', 'block_groups', 'tracts']:
        register_jobs_sector(sector, geo)
        register_prop_jobs_sector(sector, geo)


for building_type in orca.get_table('residential_units').building_type.unique():
    for geo in ['blocks', 'block_groups', 'tracts']:
        register_units_building_type(building_type, geo)


for building_type in orca.get_table('residential_units').building_type.unique():
    register_predominant_building_type_cat(building_type)


for var in ['total_units', 'total_jobs', 'total_hh', 'hh_size_1', 'total_persons', 'children',
            'persons_65plus', 'persons_black', 'persons_hispanic',
            'persons_asian', 'density_hh', 'density_units', 'density_jobs',
            'ratio_households_to_units', 'mean_income',  'prop_income_segment_1_hh', 'prop_income_segment_6_hh',
            'median_value_13_acs', 'median_rent_13_acs',  'mean_year_built', 'mean_workers', 'mean_children', 'mean_age_of_head',
            'prop_hh_rent', 'prop_households_rent', 'prop_units_rent', 'prop_units_sf', 'prop_units_mf']:
    register_disag_var('block_groups', 'blocks', var)


for var in ['density_hh', 'density_units', 'density_jobs', 'income_segment_6_hh', 'income_segment_1_hh']:
    register_disag_var('tracts', 'blocks', var)

agg_vars = ['sum_acres', 'total_jobs', 'vacant_job_spaces', 'total_hh', 'hh_rent', 'hh_own', 'hh_size_1', 'hh_size_5plus',
            'income_segment_1_hh', 'income_segment_6_hh',  'total_persons',  'children', 'persons_65plus',
            'persons_black', 'persons_hispanic', 'persons_asian', 'total_units',  'units_sf', 'units_mf',  'units_own',
            'units_rent', 'units_mf', 'units_sf', 'units_before_1930', 'units_after_2000',
            'vacant_residential_units', 'vacant_du_spaces', 'jobs_0', 'jobs_1', 'jobs_2', 'jobs_3', 'jobs_4', 'jobs_5']

for var in agg_vars:
    register_agg_var('blocks', 'block_groups', var, 'sum', prefix=False)
    register_agg_var('blocks', 'tracts', var, 'sum', prefix=False)
    register_agg_var('blocks', 'counties', var, 'sum', prefix=False)


#for var in ['income', 'year_built']:
#    register_agg_var('blocks', 'tracts', var, 'mean')
#    register_agg_var('blocks', 'tracts', var, 'std')

prop_vars = {'households':['hh_own', 'hh_rent', 'income_segment_1_hh', 'income_segment_6_hh', 'hh_size_1', 'hh_size_5plus'],
             'residential_units': ['units_own', 'units_rent', 'units_sf', 'units_mf', 'units_before_1930', 'units_after_2000']}
for agent in prop_vars.keys():
    for var in prop_vars[agent]:
        for tbl in ['blocks', 'block_groups', 'tracts', 'counties']:
            register_prop_variable(tbl, agent, var)

prop_vars = {'households':['hh_own', 'hh_rent', 'income_segment_1_hh', 'income_segment_6_hh', 'hh_size_1', 'hh_size_5plus'],
             'residential_units': ['units_own', 'units_rent', 'units_sf', 'units_mf', 'units_before_1930', 'units_after_2000']}
for agent in prop_vars.keys():
    for var in prop_vars[agent]:
        for tbl in ['blocks', 'block_groups', 'tracts', 'counties']:
            register_prop_variable(tbl, agent, var)



# Register skim variables for the zone level
sum_variables = ['total_jobs', 'total_units', 'total_hh', 'hh_size_1', 'total_persons', 'children',
                 'persons_65plus', 'persons_black', 'persons_hispanic', 'persons_asian',
                 'income_segment_1_hh', 'income_segment_6_hh']
sum_variables += ['jobs_' + sector for sector in orca.get_table('jobs').local.agg_sector.unique()]
mean_variables = ['density_jobs', 'density_units', 'density_hh', 'mean_home_rent', 'mean_home_value']
names_dict = {'euclidean_distance': 'euclidean', 'pandana_distance': 'pandana'}
impedance_columns = ['euclidean', 'pandana']
units = 'km'
impedance_thresholds = [1, 5, 10, 15, 20, 30]
zones_table = 'block_groups'
region_code = orca.get_injectable('region_code')
if 'custom_settings' in orca.list_injectables():
    custom_settings = orca.get_injectable('custom_settings')
    skim_source = orca.get_injectable('skim_source')
    custom_settings = custom_settings[region_code]
    if 'skims' in custom_settings.keys():
        names_dict = custom_settings['skims'][skim_source]['impedance_names']
        impedance_columns = [names_dict[col] for col in custom_settings['skims'][skim_source]['columns']]
        units = custom_settings['skims'][skim_source]['impedance_units']
        impedance_thresholds = custom_settings['skims'][skim_source]['impedance_thresholds']
        zones_table = custom_settings['skims'][skim_source]['zones_table']
travel_data = orca.get_table('travel_data').local.copy()
travel_data = travel_data.rename(columns=names_dict)
orca.add_table('travel_data', travel_data)
orca.add_injectable('zones_table', zones_table)
orca.add_injectable('skim_input_columns', impedance_columns)
orca.add_injectable('impedance_thresholds', impedance_thresholds)
orca.add_injectable('impedance_units', units)
if zones_table != 'block_groups':
    for var in sum_variables:
        register_agg_var('blocks', 'zones', var, 'sum', prefix=False)
    for var in mean_variables:
        register_agg_var('blocks', 'zones', var, 'mean', prefix=False)


for column in impedance_columns:
    for threshold in impedance_thresholds:
        for sum_var in sum_variables:
            column_name = column
            column_name = '%s_sum_%s_%s_%s' % (sum_var, threshold, units, column_name)
            column_name = column_name.replace('_segment', '')
            register_skim_var(zones_table, column_name, threshold, sum_var, column, 'sum', units)
            register_disag_var(zones_table, 'blocks', column_name)
        for mean_var in mean_variables:
            column_name = column
            column_name = '%s_ave_%s_%s_%s' % (mean_var, threshold, units, column_name)
            column_name = column_name.replace('_segment', '').replace('mean_', '')
            register_skim_var(zones_table, column_name, threshold, mean_var, column, 'mean', units)
            register_disag_var(zones_table, 'blocks', column_name)


# Calculate pandana-based accessibility variable
distances = range(400, 5000, 800)
agg_types = ['ave', 'sum', 'std']
decay_types = ['linear', 'flat']
variables_to_aggregate = ['total_hh', 'total_jobs', 'total_units']
variables_to_aggregate_avg_only = ['density_hh', 'density_jobs', 'density_units',
                                   'mean_income', 'mean_hh_size', 'bg_mean_age_of_head',
                                   'prop_income_segment_1_hh', 'prop_income_segment_6_hh',
                                   'home_rent', 'home_value',
                                   'mean_home_rent', 'mean_home_value',
                                   'prop_units_own', 'prop_units_rent', 'prop_units_sf',
                                   'prop_units_mf', 'prop_units_before_1930',
                                   'prop_units_after_2000']

for distance in distances:
    for decay in decay_types:
        for variable in variables_to_aggregate:
            for agg_type in agg_types:
                var_name = '_'.join([variable, agg_type, str(distance), decay[0]])
                for text in ['bg_', 'total_', 'mean_', '_segment']:
                    var_name = var_name.replace(text, '')
                var_name = var_name.replace('before_1930', 'old')
                var_name = var_name.replace('after_2000', 'new')
                log_var_name = 'ln_' + var_name
                if variable in orca.get_table('blocks').columns:
                    register_pandana_access_variable(var_name, 'blocks', variable, distance, agg_type=agg_type, decay=decay)
                    register_pandana_access_variable(log_var_name, 'blocks', variable, distance, agg_type=agg_type, decay=decay, log=True)
                    register_agg_var('blocks', 'block_groups', var_name, agg_type.replace('ave', 'mean'))
                    register_agg_var('blocks', 'block_groups', log_var_name, agg_type.replace('ave', 'mean'))
                if variable in orca.get_table('block_groups').columns:
                    register_pandana_access_variable(var_name, 'block_groups', variable, distance, agg_type=agg_type, decay=decay)
                    register_pandana_access_variable(log_var_name, 'block_groups', variable, distance, agg_type=agg_type, decay=decay, log=True)
                    register_disag_var('block_groups', 'blocks', var_name)
                    register_disag_var('block_groups', 'blocks', log_var_name)
        for variable in variables_to_aggregate_avg_only:
            var_name = '_'.join([variable, 'ave', str(distance), decay[0]])
            var_name = var_name.replace('before_1930', 'old')
            var_name = var_name.replace('after_2000', 'new')
            if 'income_segment' in var_name:
                var_name = var_name.replace('_hh', '')
            for text in ['bg_', 'mean_', '_segment']:
                var_name = var_name.replace(text, '')
            log_var_name = 'ln_' + var_name
            if variable in orca.get_table('blocks').columns:
                register_pandana_access_variable(var_name, 'blocks', variable, distance, agg_type='ave', decay=decay)
                register_pandana_access_variable(log_var_name, 'blocks', variable, distance, agg_type='ave', decay=decay, log=True)
                register_agg_var('blocks', 'block_groups', var_name, 'mean')
                register_agg_var('blocks', 'block_groups', log_var_name, 'mean')
            if variable in orca.get_table('block_groups').columns:
                register_pandana_access_variable(var_name, 'block_groups', variable, distance, agg_type='ave', decay=decay)
                register_pandana_access_variable(log_var_name, 'block_groups', variable, distance, agg_type='ave', decay=decay, log=True)
                register_disag_var('block_groups', 'blocks', log_var_name)
                register_disag_var('block_groups', 'blocks', var_name)


for table in ['blocks', 'block_groups', 'tracts', 'counties']:
    cols = orca.get_table(table).columns
    non_numeric = ['_id', '_ID', 'state', 'predominant_building_type', 'cousub']
    numeric_vars = [s for s in cols if not any(x in s for x in non_numeric)]
    numeric_vars = [var for var in numeric_vars if (var != 'x') and (var != 'y')]
    for var in numeric_vars:
        register_ln_variable(table, var)
    cols = orca.get_table(table).columns
    numeric_vars = [s for s in cols if not any(x in s for x in non_numeric)]
    for var in numeric_vars:
        register_standardized_variable(table, var)


for county in orca.get_table('blocks').county_id.unique():
    register_geog_dummy('blocks', county)


# -----------------------------------------------------------------------------------------
# CALIBRATION/VALIDATION VARIABLES
# -----------------------------------------------------------------------------------------

@orca.column('tracts', cache=True)
def jobs_obs_growth_10_17(tracts, job_targets):
    obs_growth = job_targets.local.copy()
    obs_growth = obs_growth.clip(0, None)
    obs_growth['jobs_obs_growth_10_17'] = obs_growth.sum(axis=1)
    tracts = tracts.local.join(obs_growth)
    return tracts.jobs_obs_growth_10_17.fillna(0)


@orca.column('tracts', cache=True)
def hh_obs_growth_13_18(tracts, household_targets_acs):
    obs_growth = household_targets_acs.local.copy()
    obs_growth = obs_growth.clip(0, None)
    obs_growth['hh_obs_growth_13_18'] = obs_growth.sum(axis=1)
    tracts = tracts.local.join(obs_growth)
    return tracts.hh_obs_growth_13_18.fillna(0)


@orca.column('tracts', cache=True)
def units_obs_growth_13_18(tracts, unit_targets):
    obs_growth = unit_targets.local.copy()
    obs_growth = obs_growth.drop(columns='total_units')
    obs_growth = obs_growth.clip(0, None)
    obs_growth['units_obs_growth_13_18'] = obs_growth.sum(axis=1)
    tracts = tracts.local.join(obs_growth)
    return tracts.units_obs_growth_13_18.fillna(0)


@orca.column('tracts', cache=True)
def jobs_obs_growth_17_18(tracts, job_validation):
    obs_growth = job_validation.local.copy()
    obs_growth = obs_growth.clip(0, None)
    obs_growth['jobs_obs_growth_17_18'] = obs_growth.sum(axis=1)
    tracts = tracts.local.join(obs_growth)
    return tracts.jobs_obs_growth_17_18.fillna(0)


@orca.column('tracts', cache=True)
def hh_obs_growth_18_19(tracts, household_validation_acs):
    obs_growth = household_validation_acs.local.copy()
    obs_growth = obs_growth.clip(0, None)
    obs_growth['hh_obs_growth_18_19'] = obs_growth.sum(axis=1)
    tracts = tracts.local.join(obs_growth)
    return tracts.hh_obs_growth_18_19.fillna(0)


@orca.column('tracts', cache=True)
def units_obs_growth_18_19(tracts, unit_validation):
    obs_growth = unit_validation.local.copy()
    obs_growth = obs_growth.drop(columns='total_units')
    obs_growth = obs_growth.clip(0, None)
    obs_growth['units_obs_growth_13_18'] = obs_growth.sum(axis=1)
    tracts = tracts.local.join(obs_growth)
    return tracts['units_obs_growth_13_18'].fillna(0)


@orca.column('tracts', cache=True)
def jobs_obs_growth_10_17_noclip(tracts, job_targets):
    obs_growth = job_targets.local.copy()
    obs_growth['jobs_obs_growth_10_17'] = obs_growth.sum(axis=1)
    tracts = tracts.local.join(obs_growth)
    return tracts.jobs_obs_growth_10_17.fillna(0)


@orca.column('tracts', cache=True)
def hh_obs_growth_13_18_noclip(tracts, household_targets_acs):
    obs_growth = household_targets_acs.local.copy()
    obs_growth['hh_obs_growth_13_18'] = obs_growth.sum(axis=1)
    tracts = tracts.local.join(obs_growth)
    return tracts.hh_obs_growth_13_18.fillna(0)


@orca.column('tracts', cache=True)
def units_obs_growth_13_18_noclip(tracts, unit_targets):
    obs_growth = unit_targets.local.drop(columns='total_units')
    obs_growth['units_obs_growth_13_18'] = obs_growth.sum(axis=1)
    tracts = tracts.local.join(obs_growth)
    return tracts.units_obs_growth_13_18.fillna(0)


@orca.column('tracts', cache=True)
def jobs_obs_growth_17_18_noclip(tracts, job_validation):
    obs_growth = job_validation.local.copy()
    obs_growth['jobs_obs_growth_17_18'] = obs_growth.sum(axis=1)
    tracts = tracts.local.join(obs_growth)
    return tracts.jobs_obs_growth_17_18.fillna(0)


@orca.column('tracts', cache=True)
def hh_obs_growth_18_19_noclip(tracts, household_validation_acs):
    obs_growth = household_validation_acs.local.copy()
    obs_growth['hh_obs_growth_18_19'] = obs_growth.sum(axis=1)
    tracts = tracts.local.join(obs_growth)
    return tracts.hh_obs_growth_18_19.fillna(0)


@orca.column('tracts', cache=True)
def units_obs_growth_18_19_noclip(tracts, unit_validation):
    obs_growth = unit_validation.local.copy()
    obs_growth = obs_growth.drop(columns='total_units')
    obs_growth['units_obs_growth_13_18'] = obs_growth.sum(axis=1)
    tracts = tracts.local.join(obs_growth)
    return tracts['units_obs_growth_13_18'].fillna(0)


@orca.column('tracts', cache=True)
def jobs_prop_obs_growth_10_17(tracts):
    tracts = tracts.to_frame('jobs_obs_growth_10_17')
    tracts['prop_growth'] = (tracts['jobs_obs_growth_10_17']/tracts['jobs_obs_growth_10_17'].sum())*100
    return tracts.prop_growth


@orca.column('tracts', cache=True)
def hh_prop_obs_growth_13_18(tracts):
    tracts = tracts.to_frame('hh_obs_growth_13_18')
    tracts['prop_growth'] = (tracts['hh_obs_growth_13_18']/tracts['hh_obs_growth_13_18'].sum())*100
    return tracts.prop_growth


@orca.column('tracts', cache=True)
def units_prop_obs_growth_13_18(tracts):
    tracts = tracts.to_frame('units_obs_growth_13_18')
    tracts['prop_growth'] = (tracts['units_obs_growth_13_18']/tracts['units_obs_growth_13_18'].sum())*100
    return tracts.prop_growth


@orca.column('tracts', cache=True)
def jobs_prop_obs_growth_17_18(tracts):
    tracts = tracts.to_frame('jobs_obs_growth_17_18')
    tracts['prop_growth'] = (tracts['jobs_obs_growth_17_18']/tracts['jobs_obs_growth_17_18'].sum())*100
    return tracts.prop_growth


@orca.column('tracts', cache=True)
def hh_prop_obs_growth_18_19(tracts):
    tracts = tracts.to_frame('hh_obs_growth_18_19')
    tracts['prop_growth'] = (tracts['hh_obs_growth_18_19']/tracts['hh_obs_growth_18_19'].sum())*100
    return tracts.prop_growth


@orca.column('tracts', cache=True)
def units_prop_obs_growth_18_19(tracts):
    tracts = tracts.to_frame('units_obs_growth_18_19')
    tracts['prop_growth'] = (tracts['units_obs_growth_18_19']/tracts['units_obs_growth_18_19'].sum())*100
    return tracts.prop_growth


@orca.column('tracts', cache=True)
def jobs_prop_obs_growth_10_17_noclip(tracts):
    tracts = tracts.to_frame('jobs_obs_growth_10_17_noclip')
    tracts['prop_growth'] = (tracts['jobs_obs_growth_10_17_noclip']/tracts['jobs_obs_growth_10_17_noclip'].sum())*100
    return tracts.prop_growth


@orca.column('tracts', cache=True)
def hh_prop_obs_growth_13_18_noclip(tracts):
    tracts = tracts.to_frame('hh_obs_growth_13_18_noclip')
    tracts['prop_growth'] = (tracts['hh_obs_growth_13_18_noclip']/tracts['hh_obs_growth_13_18_noclip'].sum())*100
    return tracts.prop_growth


@orca.column('tracts', cache=True)
def units_prop_obs_growth_13_18_noclip(tracts):
    tracts = tracts.to_frame('units_obs_growth_13_18_noclip')
    tracts['prop_growth'] = (tracts['units_obs_growth_13_18_noclip']/tracts['units_obs_growth_13_18_noclip'].sum())*100
    return tracts.prop_growth


@orca.column('tracts', cache=True)
def jobs_prop_obs_growth_17_18_noclip(tracts):
    tracts = tracts.to_frame('jobs_obs_growth_17_18_noclip')
    tracts['prop_growth'] = (tracts['jobs_obs_growth_17_18_noclip']/tracts['jobs_obs_growth_17_18_noclip'].sum())*100
    return tracts.prop_growth


@orca.column('tracts', cache=True)
def hh_prop_obs_growth_18_19_noclip(tracts):
    tracts = tracts.to_frame('hh_obs_growth_18_19_noclip')
    tracts['prop_growth'] = (tracts['hh_obs_growth_18_19_noclip']/tracts['hh_obs_growth_18_19_noclip'].sum())*100
    return tracts.prop_growth


@orca.column('tracts', cache=True)
def units_prop_obs_growth_18_19_noclip(tracts):
    tracts = tracts.to_frame('units_obs_growth_18_19_noclip')
    tracts['prop_growth'] = (tracts['units_obs_growth_18_19_noclip']/tracts['units_obs_growth_18_19_noclip'].sum())*100
    return tracts.prop_growth


@orca.column('tracts', cache=False)
def jobs_sim_growth_10_17(year, tracts, region_code):
    if year >= 2017:
        totals_10 = pd.read_csv('runs/%s_tract_download_indicators_2010.csv' % region_code, dtype={'tract_id': object})
        totals_10 = totals_10.set_index('tract_id')[['total_jobs']].rename(columns={'total_jobs': 'total_jobs_10'})
        if year == 2017:
            totals_17 = tracts.to_frame('total_jobs').rename(columns={'total_jobs': 'total_jobs_17'})
        else:
            totals_17 = pd.read_csv('runs/%s_tract_download_indicators_2017.csv' % region_code, dtype={'tract_id': object})
            totals_17 = totals_17.set_index('tract_id')[['total_jobs']].rename(columns={'total_jobs': 'total_jobs_17'})
        diffs = totals_10.join(totals_17)
        diffs['jobs_sim_growth_10_17'] = diffs['total_jobs_17'] - diffs['total_jobs_10']
    else:
        diffs = tracts.local.copy()
        diffs['jobs_sim_growth_10_17'] = 0
    return diffs.jobs_sim_growth_10_17


@orca.column('tracts', cache=False)
def hh_sim_growth_13_18(year, tracts, region_code):
    if year >= 2018:
        totals_13 = pd.read_csv('runs/%s_tract_layer_indicators_2013.csv' % region_code, dtype={'tract_id': object})
        totals_13 = totals_13.set_index('tract_id')[['total_hh']].rename(columns={'total_hh': 'total_hh_13'})
        if year == 2018:
            totals_18 = tracts.to_frame('total_hh').rename(columns={'total_hh': 'total_hh_18'})
        else:
            totals_18 = pd.read_csv('runs/%s_tract_download_indicators_2018.csv' % region_code, dtype={'tract_id': object})
            totals_18 = totals_18.set_index('tract_id')[['total_hh']].rename(columns={'total_hh': 'total_hh_18'})
        diffs = totals_13.join(totals_18)
        diffs['hh_sim_growth_13_18'] = diffs['total_hh_18'] - diffs['total_hh_13']
    else:
        diffs = tracts.local.copy()
        diffs['hh_sim_growth_13_18'] = 0
    return diffs.hh_sim_growth_13_18


@orca.column('tracts', cache=False)
def units_sim_growth_13_18(year, tracts, region_code):
    if year >= 2018:
        totals_13 = pd.read_csv('runs/%s_tract_download_indicators_2013.csv' % region_code, dtype={'tract_id': object})
        totals_13 = totals_13.set_index('tract_id')[['total_units']].rename(columns={'total_units': 'total_units_13'})
        if year == 2018:
            totals_18 = tracts.to_frame('total_units').rename(columns={'total_units': 'total_units_18'})
        else:
            totals_18 = pd.read_csv('runs/%s_tract_download_indicators_2018.csv' % region_code, dtype={'tract_id': object})
            totals_18 = totals_18.set_index('tract_id')[['total_units']].rename(columns={'total_units': 'total_units_18'})
        diffs = totals_13.join(totals_18)
        diffs['units_sim_growth_13_18'] = diffs['total_units_18'] - diffs['total_units_13']
    else:
        diffs = tracts.local.copy()
        diffs['units_sim_growth_13_18'] = 0
    return diffs.units_sim_growth_13_18


@orca.column('tracts', cache=False)
def jobs_sim_growth_17_18(year, tracts, region_code):
    if year >= 2018:
        totals_17 = pd.read_csv('runs/%s_tract_download_indicators_2017.csv' % region_code, dtype={'tract_id': object})
        totals_17 = totals_17.set_index('tract_id')[['total_jobs']].rename(columns={'total_jobs': 'total_jobs_17'})
        if year == 2018:
            totals_18 = tracts.to_frame('total_jobs').rename(columns={'total_jobs': 'total_jobs_18'})
        else:
            totals_18 = pd.read_csv('runs/%s_tract_download_indicators_2018.csv' % region_code, dtype={'tract_id': object})
            totals_18 = totals_18.set_index('tract_id')[['total_jobs']].rename(columns={'total_jobs': 'total_jobs_18'})
        diffs = totals_17.join(totals_18)
        diffs['jobs_sim_growth_17_18'] = diffs['total_jobs_18'] - diffs['total_jobs_17']
    else:
        diffs = tracts.local.copy()
        diffs['jobs_sim_growth_17_18'] = 0
    return diffs.jobs_sim_growth_17_18


@orca.column('tracts', cache=False)
def hh_sim_growth_18_19(year, tracts, region_code):
    if year >= 2019:
        totals_18 = pd.read_csv('runs/%s_tract_download_indicators_2018.csv' % region_code, dtype={'tract_id': object})
        totals_18 = totals_18.set_index('tract_id')[['total_hh']].rename(columns={'total_hh': 'total_hh_18'})
        if year == 2019:
            totals_19 = tracts.to_frame('total_hh').rename(columns={'total_hh': 'total_hh_19'})
        else:
            totals_19 = pd.read_csv('runs/%s_tract_download_indicators_2019.csv' % region_code, dtype={'tract_id': object})
            totals_19 = totals_19.set_index('tract_id')[['total_hh']].rename(columns={'total_hh': 'total_hh_19'})
        diffs = totals_18.join(totals_19)
        diffs['hh_sim_growth_18_19'] = diffs['total_hh_19'] - diffs['total_hh_18']
    else:
        diffs = tracts.local.copy()
        diffs['hh_sim_growth_18_19'] = 0
    return diffs.hh_sim_growth_18_19


@orca.column('tracts', cache=False)
def units_sim_growth_18_19(year, tracts, region_code):
    if year >= 2019:
        totals_18 = pd.read_csv('runs/%s_tract_download_indicators_2018.csv' % region_code, dtype={'tract_id': object})
        totals_18 = totals_18.set_index('tract_id')[['total_units']].rename(columns={'total_units': 'total_units_18'})
        if year == 2019:
            totals_19 = tracts.to_frame('total_units').rename(columns={'total_units': 'total_units_19'})
        else:
            totals_19 = pd.read_csv('runs/%s_tract_download_indicators_2019.csv' % region_code, dtype={'tract_id': object})
            totals_19 = totals_19.set_index('tract_id')[['total_units']].rename(columns={'total_units': 'total_units_19'})
        diffs = totals_18.join(totals_19)
        diffs['units_sim_growth_18_19'] = diffs['total_units_19'] - diffs['total_units_18']
    else:
        diffs = tracts.local.copy()
        diffs['units_sim_growth_18_19'] = 0
    return diffs.units_sim_growth_18_19


@orca.column('tracts', cache=False)
def jobs_prop_sim_growth_10_17(tracts):
    tracts = tracts.to_frame('jobs_sim_growth_10_17')
    tracts['prop_growth'] = (tracts['jobs_sim_growth_10_17']/tracts['jobs_sim_growth_10_17'].sum())*100
    return tracts.prop_growth.fillna(0)


@orca.column('tracts', cache=False)
def hh_prop_sim_growth_13_18(tracts):
    tracts = tracts.to_frame('hh_sim_growth_13_18')
    tracts['prop_growth'] = (tracts['hh_sim_growth_13_18']/tracts['hh_sim_growth_13_18'].sum())*100
    return tracts.prop_growth.fillna(0)


@orca.column('tracts', cache=False)
def units_prop_sim_growth_13_18(tracts):
    tracts = tracts.to_frame('units_sim_growth_13_18')
    tracts['prop_growth'] = (tracts['units_sim_growth_13_18']/tracts['units_sim_growth_13_18'].sum())*100
    return tracts.prop_growth.fillna(0)


@orca.column('tracts', cache=False)
def jobs_prop_sim_growth_17_18(tracts):
    tracts = tracts.to_frame('jobs_sim_growth_17_18')
    tracts['prop_growth'] = (tracts['jobs_sim_growth_17_18']/tracts['jobs_sim_growth_17_18'].sum())*100
    return tracts.prop_growth.fillna(0)


@orca.column('tracts', cache=False)
def hh_prop_sim_growth_18_19(tracts):
    tracts = tracts.to_frame('hh_sim_growth_18_19')
    tracts['prop_growth'] = (tracts['hh_sim_growth_18_19']/tracts['hh_sim_growth_18_19'].sum())*100
    return tracts.prop_growth.fillna(0)


@orca.column('tracts', cache=False)
def units_prop_sim_growth_18_19(tracts):
    tracts = tracts.to_frame('units_sim_growth_18_19')
    tracts['prop_growth'] = (tracts['units_sim_growth_18_19']/tracts['units_sim_growth_18_19'].sum())*100
    return tracts.prop_growth.fillna(0)


@orca.column('tracts', cache=True)
def base_jobs(jobs, tracts):
    jobs = jobs.to_frame('tract_id')
    jobs = jobs.tract_id.value_counts()
    return tracts.local.join(jobs).fillna(0)


@orca.column('tracts', cache=True)
def base_hh(households, tracts):
    hh = households.to_frame('tract_id')
    hh = hh.tract_id.value_counts()
    return tracts.local.join(hh).fillna(0)


@orca.column('tracts', cache=True)
def base_units(residential_units, tracts):
    units = residential_units.to_frame('tract_id')
    units = units.tract_id.value_counts()
    return tracts.local.join(units).fillna(0)


@orca.column('tracts', cache=True, cache_scope='iteration')
def jobs_sim_growth_year(tracts):
    tracts = tracts.to_frame(['base_jobs', 'total_jobs'])
    return tracts.total_jobs - tracts.base_jobs


@orca.column('tracts', cache=True, cache_scope='iteration')
def hh_sim_growth_year(tracts):
    tracts = tracts.to_frame(['base_hh', 'total_hh'])
    return tracts.total_hh - tracts.base_hh


@orca.column('tracts', cache=True, cache_scope='iteration')
def units_sim_growth_year(tracts):
    tracts = tracts.to_frame(['base_units', 'total_units'])
    return tracts.total_units - tracts.base_units


@orca.column('tracts', cache=True, cache_scope='iteration')
def jobs_prop_sim_growth_year(tracts):
    tracts = tracts.to_frame(['jobs_sim_growth_year'])
    return (tracts.jobs_sim_growth_year/tracts.jobs_sim_growth_year.sum())*100


@orca.column('tracts', cache=True, cache_scope='iteration')
def hh_prop_sim_growth_year(tracts):
    tracts = tracts.to_frame(['hh_sim_growth_year'])
    return (tracts.hh_sim_growth_year/tracts.hh_sim_growth_year.sum())*100


@orca.column('tracts', cache=True, cache_scope='iteration')
def units_prop_sim_growth_year(tracts):
    tracts = tracts.to_frame(['units_sim_growth_year'])
    return (tracts.units_sim_growth_year/tracts.units_sim_growth_year.sum())*100


def register_base_hh_type(type):
    @orca.column('tracts', 'base_hh_' + str(type), cache=True)
    def column_func(tracts, households):
        tracts = tracts.local.copy()
        hh = households.to_frame(['hh_type', 'tract_id'])
        hh = hh[hh['hh_type'] == type]
        hh = pd.DataFrame(hh.groupby('tract_id').size())
        hh.columns = ['hh']
        df = tracts.join(hh).fillna(0)
        df = df.groupby('tract_id').sum()
        return df['hh']
    return column_func


def register_current_hh_type(type):
    @orca.column('tracts', 'current_hh_' + str(type), cache=True, cache_scope='iteration')
    def column_func(tracts, households):
        tracts = tracts.local.copy()
        hh = households.to_frame(['hh_type', 'tract_id'])
        hh = hh[hh['hh_type'] == type]
        hh = pd.DataFrame(hh.groupby('tract_id').size())
        hh.columns = ['hh']
        df = tracts.join(hh).fillna(0)
        df = df.groupby('tract_id').sum()
        return df['hh']
    return column_func


for hh_type in orca.get_table('households').hh_type.unique():
    register_base_hh_type(hh_type)
    register_current_hh_type(hh_type)
