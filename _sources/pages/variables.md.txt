# Variables Reference

DEMOS computes lazily evaluated columns (via the [orca](https://udst.github.io/orca/) framework) that
are available as inputs to model expressions and calibration steps. This page describes the naming
conventions used for these columns and provides an inventory of every computed variable by table.

```{contents}
:local:
:depth: 2
:backlinks: none
```

---

## Naming Conventions

### Persons table

| Prefix / Pattern | Meaning | Example |
|---|---|---|
| `age_` | Age bin for persons | `age_23_35`, `age_60plus` |
| `age_emp_` | Age bin used in employment (enter/exit labor-force) models | `age_emp_20_40`, `age_emp_70plus` |
| `age_mort_` | Age bin used in the mortality model | `age_mort_21_40`, `age_mort_90plus` |
| `age_km_` | Age bin used in the kids-move model | `age_km_19_20`, `age_km_30plus` |
| `sex_` | Sex indicator | `sex_female` |
| `emp_idle_` | Idle (non-working) employment status bin | `emp_idle_under60`, `emp_idle_over60` |
| `edu_` | Education attainment bin | `edu_hs_ged`, `edu_college_plus` |
| `race_` | Race/ethnicity indicator | `race_white`, `race_black`, `race_asian_pi`, `race_native_am`, `race_acs_other`, `race_hawaiian`, `race_asian`, `race_other` |
| `mar_` | Marital status indicator | `mar_married`, `mar_widowed`, `mar_div_or_sep`, `mar_widowed_or_never` |
| `emp_idle_*_age_km_*` | Cross-product: employment status × kids-move age bin | `emp_idle_under60_age_km_19_20` |
| `edu_*_age_km_*` | Cross-product: education bin × kids-move age bin | `edu_hs_ged_age_km_21_25` |
| `intercept` | Constant term (ones vector) | `intercept` |

### Households table

| Prefix / Pattern | Meaning | Example |
|---|---|---|
| `hh_` | Computed household-level aggregate or indicator | `hh_n_persons`, `hh_n_children` |
| `hh_income_` | Household income bin | `hh_income_bin1` … `hh_income_bin5` |
| `hh_edu_top_` | Education bin of the most-educated head/spouse | `hh_edu_top`, `hh_edu_top_bin2`, `hh_edu_top_bin3` |
| `hh_fam_work` | Count of working head/spouse members | `hh_fam_work`, `hh_fam_work2` |
| `hh_age_avg_` | Average age of head/spouse, and bins | `hh_age_avg`, `hh_age_avg_bin2` … `hh_age_avg_bin4` |
| `hh_age_min_` | Minimum age of head/spouse, and bins | `hh_age_min`, `hh_age_min_bin2` … `hh_age_min_bin4` |
| `hh_fsize_` | Household-size bin for birth model | `hh_fsize_bin23`, `hh_fsize_bingt3` |
| `hh_birth_age_` | Age of relevant female member, for birth model | `hh_birth_age_lt27`, `hh_birth_age_27_35` |
| `hh_head_` | Attribute of the household head (relate == 0) | `hh_head_age`, `hh_head_race_id`, `hh_head_race_str`, `hh_head_race_white` |
| `hh_head_edu_` | Education bin of head only (used in income model) | `hh_head_edu_bin1` … `hh_head_edu_bin3` |
| `hh_head_race_` | Race indicator derived from the household head | `hh_head_race_black`, `hh_head_race_native_am`, `hh_head_race_asian`, `hh_head_race_hawaiian`, `hh_head_race_acs_other` |
| `hh_size_str` | Categorical household size string ("one", "two", …) | `hh_size_str` |
| `intercept` | Constant term (ones vector) | `intercept` |

> **`hh_edu_top_bin` vs. `hh_head_edu_bin`**  
> These are distinct columns. `hh_edu_top_bin` reflects the maximum education of the head *or* spouse
> (relate < 2) and is used in the divorce and cohabitation models. `hh_head_edu_bin` reflects the
> education of the household head only (relate == 0) and is used in the income model.

---

## Column Inventory

### Persons columns

#### Age bins (general)

| Column | Definition |
|---|---|
| `age_23_35` | Age ∈ [23, 35] |
| `age_36_60` | Age ∈ [36, 60] |
| `age_60plus` | Age > 60 |
| `intercept` | 1 for every person |

#### Age bins — employment models

| Column | Definition |
|---|---|
| `age_emp_20_40` | Age ∈ [20, 40] |
| `age_emp_41_50` | Age ∈ [41, 50] |
| `age_emp_51_70` | Age ∈ [51, 70] |
| `age_emp_70plus` | Age > 70 |

#### Age bins — mortality model

| Column | Definition |
|---|---|
| `age_mort_21_40` | Age ∈ [21, 40] |
| `age_mort_41_50` | Age ∈ [41, 50] |
| `age_mort_51_70` | Age ∈ [51, 70] |
| `age_mort_71_90` | Age ∈ [71, 90] |
| `age_mort_90plus` | Age > 90 |

#### Age bins — kids-move model

| Column | Definition |
|---|---|
| `age_km_16_18` | Age ∈ [16, 18] |
| `age_km_19_20` | Age ∈ [19, 20] |
| `age_km_21_25` | Age ∈ [21, 25] |
| `age_km_26_30` | Age ∈ [26, 30] |
| `age_km_30plus` | Age > 30 |

#### Sex

| Column | Definition |
|---|---|
| `sex_female` | sex == 2 |

#### Employment status

| Column | Definition |
|---|---|
| `emp_idle_under60` | worker == 0 and age < 60 |
| `emp_idle_over60` | worker == 0 and age ≥ 60 |

#### Education

| Column | Definition |
|---|---|
| `edu_hs_ged` | edu ∈ [16, 17] (high school diploma or GED) |
| `edu_college_plus` | edu > 17 (some college or higher) |

#### Race / ethnicity

| Column | Definition | Notes |
|---|---|---|
| `race_white` | race_id == 1 | |
| `race_black` | race_id == 2 | |
| `race_native_am` | race_id ∈ {3, 4, 5} | |
| `race_asian` | race_id == 6 | |
| `race_hawaiian` | race_id == 7 | |
| `race_asian_pi` | race_id ∈ {6, 7} | Asian or Pacific Islander combined |
| `race_acs_other` | race_id == 8 | ACS "some other race" category; building block for head-of-household columns |
| `race_other` | race_id ∈ {3, 4, 5, 8, 9} | Broad "other" category used in employment models |

#### Marital status

| Column | Definition |
|---|---|
| `mar_married` | MAR == 1 |
| `mar_widowed` | MAR == 2 |
| `mar_div_or_sep` | MAR ∈ {3, 4} |
| `mar_widowed_or_never` | MAR ∈ {2, 5} |

#### Cross-products (kids-move model)

Eight `emp_idle_under60 × age_km_*` and eight `emp_idle_over60 × age_km_*` columns, plus eight
`edu_hs_ged × age_km_*` and eight `edu_college_plus × age_km_*` columns. The pattern is
`<base_col>_<age_km_bin>`, e.g. `emp_idle_under60_age_km_19_20`.

---

### Households columns

#### Head-of-household attributes

| Column | Definition |
|---|---|
| `hh_head_age` | Age of person with relate == 0 |
| `hh_head_race_id` | race_id of person with relate == 0 |
| `hh_head_race_str` | String label of head's race: `'white'`, `'black'`, `'asian'`, or `'other'` |
| `hh_head_race_white` | 1 if head's race_id == 1 |
| `hh_head_race_black` | 1 if head's race_id == 2 |
| `hh_head_race_native_am` | 1 if head's race_id ∈ {3, 4, 5} |
| `hh_head_race_asian` | 1 if head's race_id == 6 |
| `hh_head_race_hawaiian` | 1 if head's race_id == 7 |
| `hh_head_race_acs_other` | 1 if head's race_id == 8 |
| `hh_head_edu_bin1` | head's edu < 16 (less than high school) |
| `hh_head_edu_bin2` | head's edu ∈ [16, 17] (HS/GED) |
| `hh_head_edu_bin3` | head's edu > 17 (some college or more) |
| `hh_size_str` | Categorical size: `'one'`, `'two'`, etc. |

#### Household composition

| Column | Definition |
|---|---|
| `hh_n_persons` | Total number of persons in the household |
| `hh_n_children` | Number of persons with age ≤ 17 |

#### Income bins

| Column | Definition |
|---|---|
| `hh_income_bin1` | income < $25,000 |
| `hh_income_bin2` | income ∈ [$25k, $50k) |
| `hh_income_bin3` | income ∈ [$50k, $75k) |
| `hh_income_bin4` | income ∈ [$75k, $150k) |
| `hh_income_bin5` | income ≥ $150,000 |

#### Top education of head/spouse (relate < 2)

| Column | Definition |
|---|---|
| `hh_edu_top` | Maximum edu value among head and spouse |
| `hh_edu_top_bin2` | hh_edu_top ∈ [16, 17] |
| `hh_edu_top_bin3` | hh_edu_top > 17 |

#### Working members (head/spouse)

| Column | Definition |
|---|---|
| `hh_fam_work` | Number of working (worker == 1) head/spouse members |
| `hh_fam_work2` | 1 if hh_fam_work == 2 |

#### Average age of head/spouse

| Column | Definition |
|---|---|
| `hh_age_avg` | Mean age of head and spouse (relate < 2) |
| `hh_age_avg_bin2` | hh_age_avg ∈ (22, 35] |
| `hh_age_avg_bin3` | hh_age_avg ∈ (35, 60] |
| `hh_age_avg_bin4` | hh_age_avg > 60 |

#### Minimum age of head/spouse

| Column | Definition |
|---|---|
| `hh_age_min` | Minimum age of head and spouse (relate < 2) |
| `hh_age_min_bin2` | hh_age_min ∈ (22, 35] |
| `hh_age_min_bin3` | hh_age_min ∈ (35, 60] |
| `hh_age_min_bin4` | hh_age_min > 60 |

#### Birth-model household columns

| Column | Definition |
|---|---|
| `hh_birth_age_lt27` | Age of relevant female member ≤ 27 |
| `hh_birth_age_27_35` | Age of relevant female member ∈ (27, 35] |
| `hh_fsize_bin23` | hh_n_persons ∈ {2, 3} |
| `hh_fsize_bingt3` | hh_n_persons > 3 |

#### Other

| Column | Definition |
|---|---|
| `income_segment` | Quantile-based income segment (1–6) |
| `hh_type` | Categorical type code (1–8) based on tenure, size, and age of head |
| `county_id` | First 5 characters of block_id |
| `tract_id` | First 11 characters of block_id |
| `intercept` | 1 for every household |

---

## Where variables are defined

Each variable is defined in the module that owns it, or in `variables.py` if it is shared across models.

| File | Variables defined |
|---|---|
| `demos/variables.py` | Shared persons variables (age bins, sex, employment, education, race, marital status), shared household variables (intercept, county/tract IDs, income segment, hh_type), and blocks aggregations |
| `demos/models/aging.py` | `hh_head_age` |
| `demos/models/birth.py` | `hh_n_persons`, `hh_fsize_bin23`, `hh_fsize_bingt3`, `hh_birth_age_lt27`, `hh_birth_age_27_35` |
| `demos/models/employment.py` | `age_emp_20_40`, `age_emp_41_50`, `age_emp_51_70`, `age_emp_70plus` |
| `demos/models/fatality.py` | `age_mort_21_40`, `age_mort_41_50`, `age_mort_51_70`, `age_mort_71_90`, `age_mort_90plus` |
| `demos/models/household_reorg.py` | `hh_head_race_id`, `hh_head_race_str`, `hh_size_str`, `hh_n_children`, `hh_income_bin1–5`, `hh_edu_top`, `hh_edu_top_bin2–3`, `hh_fam_work`, `hh_fam_work2`, `hh_age_avg` + bins, `hh_age_min` + bins, `hh_head_race_white` |
| `demos/models/income.py` | `hh_head_edu_bin1–3`, `hh_head_race_black`, `hh_head_race_native_am`, `hh_head_race_asian`, `hh_head_race_hawaiian`, `hh_head_race_acs_other` |
| `demos/models/kids_moving.py` | `age_km_*` bins, all `emp_idle_*_age_km_*` and `edu_*_age_km_*` cross-products |
