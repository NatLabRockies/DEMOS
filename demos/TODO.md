# TODOs
- Why were there so many problems with file ownership?
- What is in the runs folder?
    - Looks like there is some kind of intermediate output.
- What steps from the bottom of `models.py` do we want to keep?
- Refactor the model calibration steps
- Why the laborforce models do not return a value for every person?
- All computed columns should return a series with a valid index for the corresponding table
- I noticed the number of households with multiple partners increases

- `models.py`
    - `work_location` seems obsolete

- `laborforce.py`
    - Fix the filter in estimated model for out workforce (worker==0 should be 1)
    - students are currently selected for work

- `education.py`
    - Is there no way for students to enter school?
    - School progression is faulty because of the order of the calculation
    - Double check the logic for transitioning from 14,15,16

- `household_reorg.py`
    - <del>Check if the three models interfere with themselves (they are applied according to filters to the same dataset)</del>
        - Models do not seem to intersect
    - Household 382474 has 4 people labeled as relate == 1 and some of them have MAR == 5
        - This means the outputs will not be exactly the same after the refactoring
    - `hh_income` is incorrectly computed: There is a hard-coded 30_000 and the rest are 60, 100, etc (not thousands)
    - I believe the only reason `fix_erroneous_households` exists is in case people are flagged by two models at once

- `marriage.py`
    - There is a filter that when the number of people getting married is too low, the module does nothing
    - Filter for <= 10 weddings
    - There was also this code `if (min_mar == 0) or (min_mar == 0):`
    - <del>Discuss `CONDITIONS` part of the code</del>
    - `MAR` is not correctly being updated because final is filtered to those that move
    - If both new partners are head of household, one could potentially leave dependents behind.
        - I think the current code is just making the person that earns the most head of household
        - In fact at the moment there are children labeled as head of household (9 year olds earning 0 for instance)
    - I ignored the "marriage_table" and "divorce_table", consider re-implementing it after the refactor
    - What is `member_id` and why is set 1 for leaving person in a divorce but "relate" for those staying?

- `kids_moving`
    - The filters in the estimated model consider relate values of 7 and 9 children as well
    - I ignored the "kids_move_table", consider re-implementing it after the refactor

- `mortaility.py`
    - I ignored the "mortalities" table, consider re-implementing it after the refactor
    - In `rel_map` table, `6,6 = 1`, which assumes marriage?
    - In `rez` function, if spouse or partner becomes head, `relate` is not updated.
    - If `relate==13` dies, the head is also labeled as `MAR=3` (I thought that was necessarily marriage widow).

- `birth.py`
    - I ignored the "btable" table, consider re-implementing it after the refactor
    - Review the values of `education_group`, `age_group`, etc.
    - Default `MAR == 5`?
    - There is duplication of information between `race_id` and `race`
    - `race` ignores `asian` values (it only maps `white` and `black`)
    - Check for the need to add the logic of `hispanic`, `hispanic.1`, ... 

- `transition`
    - If we need to increase the number of household and have none, skip

## Ideas for cheking sanity of input data
- Check there is only one head of household
    - We should fix the households that don't have a head at the start

# Commit history review
August 8th, 2024
- ⬜️ `cc558e4`: Edits `README.md`
- 🟥 `03efa62`: Changes to cohabitation and marriage YAML files as well as changes to `models.py` and `datasources.py`
- ⬜️ `f382ca0`: Edits to `README.md`
- ⬜️ `67d3969`: Edits to `README.md`
- ⬜️ `5e3d53e`: Edits to `README.md`

----
August 7th, 2024
- 🟥 `d5aae57`: Changes to marriage YAML (changed a table name) files as well as changes to `models.py`
- 🟥 `2f29b7a`: Changes to cohabitation and marriage YAML files as well as changes to `models.py`, `datasources.py` and `multinomial_logit.py`

----
August 5th, 2024
- 🟥 `f3f3c8d`: Too much to describe. Commit message says "detach all urbansim packages from demos"
----
July 31st, 2024
- 🟥 `747a8aa`: Too much to describe. Seems that most of it is moving files around
- 🟥 `9474012`: Too much to describe. Commit message says "detach all urbansim packages from demos" (again)
- 🟥 `5d4f8a5`: "Import urbansim template as templates"
----
July 30th, 2024
- ⬜️ `2b73d21`: Small change to function that creates a directory
- ⬜️ `dbf8dbc`: Small change to function that creates a directory
----
July 29th, 2024
- 🟧 `394d459`: Changes to `plotting.py` and a bunch of csv files
----
July 26th, 2024
- 🟧 `bac973e`: Changes to `plotting.py`
- ⬜️ `082ba53`: Changes a bunch of csv files
- ⬜️ `180ab57`: Remove `process_skims.py` and `settings.yaml` from `demos_urbansim`
- ⬜️ `11bea38`: Same as `180ab57`
- 🟥 `bda827d`: "Removing unused models in configs"
----
July 25th, 2024
- 🟥 `f35650d`: Same as `bda827d`
----
July 23rd, 2024
- ⬜️ `994852f`: Changes `.gitignore` and a hardcoded string to and `.h5` file
- ⬜️ `9960309`: Same as `994852f`
----
July 22nd, 2024
- ⬜️ `a3e8656`: Removing `utils.py`
----
July 20th, 2024
- ⬜️ `9c2dd6e`: Same as `a3e8656`
- 🟧 `3b9781c`: Changes to `plotting.py`
----
July 19th, 2024
- 🟧 `fb78d01`: Changes to `plotting.py`
- 🟦 `2de236f`: Changes to how `simulate.py` loads some parameters
- 🟦 `985dff3`: Same as `2de236f`
- 🟥 `85a61de`: A lot happening. Seems more like a refactoring
----
July 18th, 2024
- 🟥 `0af091e`: Same as `85a61de`
----
July 1st, 2024
- ⬜️ `261fdeb`: Removing files from `demos_urbansim`
- 🟧 `231bfc0`: Changes to printing statements in `models.py`
- ⬜️ `9820535`: Removing files from `demos_urbansim`
- 🟧 `1c9b154`: Changes to printing statements in `models.py`
----
June 12th, 2024
- ⬜️ `1bf7356`: Added `cmp_hdf5_files.py`
----
June 10th, 2024
- ⬜️ `bcb6a7d`: Same as `1bf7356`. Apparently this functions compare outputs
----
June 6th, 2024
- ⬜️ `fb2e4eb`: Removing unused imports
- ⬜️ `006b48e`: Removing a call to `os.chown`
----
June 5th, 2024
- ⬜️ `2758a74`: Removing unused imports
- 🟥 `e1849a4`: Commenting out a bunch of steps from the model
----
May 28th, 2024
- 🟦 `974daee`: Very similar to `e1849a4`. Commit message says "Fix code to run"
- ⬜️ `49935ef`: pycache handling
----
May 27th, 2024
- 🟩 `58c5fe2`: First commit
