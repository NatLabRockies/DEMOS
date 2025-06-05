import orca

@orca.step()
def fix_persons_table(persons, households):
    # TODO: Add identifiers to which hh are problematic

    # Check for households with no head
    n_heads_idx = (persons.relate == 0).groupby(persons.household_id).sum()
    assert (n_heads_idx > 1).sum() == 0, "Some households have more than one head"

    no_head_hhs = n_heads_idx[n_heads_idx < 1].index
    no_head_hh_sizes = persons.local[persons.household_id.isin(no_head_hhs)].groupby("household_id").size()
    
    if len(no_head_hhs) > 0:
        print(f"{len(no_head_hhs)} households with no head. {(no_head_hh_sizes == 1).sum()} were single-person households, the person was made head. The other {(no_head_hh_sizes != 1).sum()} were dropped.")
        persons.local.loc[persons.household_id.isin(no_head_hh_sizes[no_head_hh_sizes == 1].index),
                        "relate"] = 0
        persons.local = persons.local[~persons.household_id.isin(no_head_hh_sizes[no_head_hh_sizes != 1].index)]

    # Check for consistency in `relate` and `MAR`
    ## First, fix households with multiple partners
    n_partners = ((persons["relate"] == 0) | (persons["relate"] == 1) | (persons["relate"] == 13)).groupby(persons.household_id).sum()
    multi_partner_hh = n_partners[n_partners > 2].index
    
    if len(multi_partner_hh) > 0:
        print(f"{len(multi_partner_hh)} households with multiple partners ({len(persons.local.loc[persons.household_id.isin(multi_partner_hh)])} total people) were dropped")
        persons.local = persons.local[~persons.household_id.isin(multi_partner_hh)]
    
    ## Then, sanitize `MAR` column
    head_or_spouse_idx = (persons["relate"] == 0) | (persons["relate"] == 1)
    married_hhs = persons.local.loc[persons["relate"] == 1]["household_id"].values
    cohabitating_hhs = persons.local.loc[persons["relate"] == 13]["household_id"].values

    married_hh_idx = persons.household_id.isin(married_hhs)
    cohabitating_hh_idx = persons.household_id.isin(cohabitating_hhs)

    ### Handling of married households
    hhs_with_wrong_MAR_idx = persons.local.loc[married_hh_idx & head_or_spouse_idx].groupby(["household_id", "MAR"]).size().loc[:, 1] != 2
    hhs_with_wrong_MAR = hhs_with_wrong_MAR_idx[hhs_with_wrong_MAR_idx].index
    married_head_no_spouse = ~married_hh_idx & head_or_spouse_idx & (persons.MAR == 1)

    if married_head_no_spouse.sum() > 0:
        print(f"{married_head_no_spouse.sum()} heads have MAR == 1 but no spouse in house. Changing to MAR = 0")
        persons.local.loc[~married_hh_idx & head_or_spouse_idx & (persons.MAR == 1), "MAR"] = 0

    if len(hhs_with_wrong_MAR) > 0:
        print(f"{len(hhs_with_wrong_MAR)} households have married people but the number of MAR == 1 is different than 2. Spouses were flagged with MAR = 1")
        
        problematic_hh_idx = persons["household_id"].isin(hhs_with_wrong_MAR)
        persons.local.loc[problematic_hh_idx & head_or_spouse_idx, "MAR"] = 1
        # persons.local.loc[problematic_hh_idx & ~head_or_spouse_idx, "MAR"] = 0
    
    ### Handling of cohabitating households
    cohabitating_people_married_idx = cohabitating_hh_idx & (persons.relate.isin([0, 13])) & (persons.MAR == 1)
    if cohabitating_people_married_idx.sum():
        print(f"{cohabitating_people_married_idx.sum()} people cohabitating (relate == 0 or 13), are flagged as married (MAR == 1). Changing to 0")
        persons.local.loc[cohabitating_people_married_idx, "MAR"] = 0

    ### Handling of people with MAR == 1 but not spouse or head
    # incorrect_MAR_label = ~persons.relate.isin([0, 1]) & (persons.MAR == 1)
    # if incorrect_MAR_label.sum() > 0:
    #     print(f"{incorrect_MAR_label.sum()} people are flagged as married (MAR == 1) but are neither head not spouse in relate column. Changing MAR to 0")
    #     persons.local.loc[incorrect_MAR_label, "MAR"] = 0

    # TODO: This needs to be reevaluated after the refactoring
    households.local = households.local.reindex(sorted(persons.household_id.unique()))
    ...