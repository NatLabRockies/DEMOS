import orca
from config import DEMOSConfig, get_config
from loguru import logger


@orca.step()
def validate_persons_table(persons, households):
    config: DEMOSConfig = get_config()
    raise_errors: bool = config.inconsistent_persons_table_behavior == "error"
    fix_inconsistent: bool = config.inconsistent_persons_table_behavior == "fix"

    # Check for households with no head
    n_heads_idx = (persons.relate == 0).groupby(persons.household_id).sum()
    assert (n_heads_idx > 1).sum() == 0, "Some households have more than one head"

    no_head_hhs = n_heads_idx[n_heads_idx < 1].index
    no_head_hh_sizes = (
        persons.local[persons.household_id.isin(no_head_hhs)]
        .groupby("household_id")
        .size()
    )

    if len(no_head_hhs) > 0:
        if raise_errors:
            raise ValueError(
                f"{len(no_head_hhs)} households with no head in input data"
            )
        if fix_inconsistent:
            logger.warning(
                f"{len(no_head_hhs)} households with no head. {(no_head_hh_sizes == 1).sum()} were single-person households, the person was made head. The other {(no_head_hh_sizes != 1).sum()} were dropped."
            )
            persons.local.loc[
                persons.household_id.isin(
                    no_head_hh_sizes[no_head_hh_sizes == 1].index
                ),
                "relate",
            ] = 0
            persons.local = persons.local[
                ~persons.household_id.isin(
                    no_head_hh_sizes[no_head_hh_sizes != 1].index
                )
            ]

    # Check for consistency in `relate` and `MAR`
    ## First, fix households with multiple partners
    n_partners = (
        (
            (persons["relate"] == 0)
            | (persons["relate"] == 1)
            | (persons["relate"] == 13)
        )
        .groupby(persons.household_id)
        .sum()
    )
    multi_partner_hh = n_partners[n_partners > 2].index

    if len(multi_partner_hh) > 0:
        if raise_errors:
            raise ValueError(
                f"{len(multi_partner_hh)} households with multiple partners"
            )
        if fix_inconsistent:
            logger.warning(
                f"{len(multi_partner_hh)} households with multiple partners ({len(persons.local.loc[persons.household_id.isin(multi_partner_hh)])} total people) were dropped"
            )
            persons.local = persons.local[~persons.household_id.isin(multi_partner_hh)]

    ## Then, sanitize `MAR` column
    head_or_spouse_idx = (persons["relate"] == 0) | (persons["relate"] == 1)
    married_hhs = persons.local.loc[persons["relate"] == 1]["household_id"].values
    cohabitating_hhs = persons.local.loc[persons["relate"] == 13]["household_id"].values

    married_hh_idx = persons.household_id.isin(married_hhs)
    cohabitating_hh_idx = persons.household_id.isin(cohabitating_hhs)

    ### Handling of married households
    hhs_with_wrong_MAR_idx = (
        persons.local.loc[married_hh_idx & head_or_spouse_idx]
        .groupby(["household_id", "MAR"])
        .size()
        .loc[:, 1]
        != 2
    )
    hhs_with_wrong_MAR = hhs_with_wrong_MAR_idx[hhs_with_wrong_MAR_idx].index
    married_head_no_spouse = ~married_hh_idx & head_or_spouse_idx & (persons.MAR == 1)

    if married_head_no_spouse.sum() > 0:
        if raise_errors:
            raise ValueError(
                f"{married_head_no_spouse.sum()} heads have MAR == 1 but no spouse in house."
            )
        if fix_inconsistent:
            logger.warning(
                f"{married_head_no_spouse.sum()} heads have MAR == 1 but no spouse in house. Changing to MAR = 0"
            )
            persons.local.loc[
                ~married_hh_idx & head_or_spouse_idx & (persons.MAR == 1), "MAR"
            ] = 0

    if len(hhs_with_wrong_MAR) > 0:
        if raise_errors:
            raise ValueError(
                f"{len(hhs_with_wrong_MAR)} households have married people but the number of MAR == 1 is different than 2."
            )
        if fix_inconsistent:
            logger.warning(
                f"{len(hhs_with_wrong_MAR)} households have married people but the number of MAR == 1 is different than 2. Spouses were flagged with MAR = 1"
            )

            problematic_hh_idx = persons["household_id"].isin(hhs_with_wrong_MAR)
            persons.local.loc[problematic_hh_idx & head_or_spouse_idx, "MAR"] = 1
            # persons.local.loc[problematic_hh_idx & ~head_or_spouse_idx, "MAR"] = 0

    ### Handling of cohabitating households
    cohabitating_people_married_idx = (
        cohabitating_hh_idx & (persons.relate.isin([0, 13])) & (persons.MAR == 1)
    )
    if cohabitating_people_married_idx.sum():
        if raise_errors:
            raise ValueError(
                f"{cohabitating_people_married_idx.sum()} people cohabitating (relate == 0 or 13), are flagged as married (MAR == 1)."
            )
        if fix_inconsistent:
            logger.warning(
                f"{cohabitating_people_married_idx.sum()} people cohabitating (relate == 0 or 13), are flagged as married (MAR == 1). Changing to 0"
            )
            persons.local.loc[cohabitating_people_married_idx, "MAR"] = 0

    households.local = households.local.reindex(sorted(persons.household_id.unique()))
