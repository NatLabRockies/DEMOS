"""
Created on Thu Apr  3 14:28:19 2025
@author: gzhao
"""

import argparse
import numpy as np
import pandas as pd


if __name__ == "__main__":
    # CLI arguments
    parser = argparse.ArgumentParser(description="SCAG Application data preprocessing script")
    parser.add_argument("dir", type=str, help="root directory of the synthetic population input")
    parser.add_argument("-o", "--out", type=str, help="output directory")
    args = parser.parse_args()

    ROOT_DIR = args.dir
    OUT_DIR = args.out if args.out else ROOT_DIR

    synpop_hh = pd.read_csv(f"{ROOT_DIR}/synpop_2019/expand_hh_2019.csv")
    synpop_pp = pd.read_csv(f"{ROOT_DIR}/synpop_2019/expand_pp_2019.csv")
    addnm_pinc = pd.read_csv(f"{ROOT_DIR}/synpop_2019/addnm_pinc_2019.csv")
    synpop_hh = synpop_hh.rename(columns={"hhid": "household_id", 
                                      "hhsize": "persons",
                                      "hhinc": "income",
                                      "hcounty": "lcm_county_id",
                                      })
    synpop_pp = synpop_pp.rename(columns={"hhid": "household_id", 
                                        "pid": "person_id", 
                                        "gender": "sex",
                                        "rac1p": "race_id",
                                        "mar": "MAR",
                                        "pnum": "member_id",
                                        })
    addnm_pinc = addnm_pinc.rename(columns={"pid": "person_id",
                                            "pinc": "earning", # personal "earning" in DEMOS acutally means personal "income"
                                            })
    
    age_of_head = synpop_pp[synpop_pp['relshipp']==20][['household_id', 'age']].rename(columns={"age": "age_of_head"})
    synpop_hh = synpop_hh.merge(age_of_head, on=["household_id"])

    race_of_head = synpop_pp[synpop_pp['relshipp']==20][['household_id', 'race_id']].rename(columns={"race_id": "race_of_head"})
    synpop_hh = synpop_hh.merge(race_of_head, on=["household_id"])

    synpop_hh['tenure'] = np.where(synpop_hh['ten'].isin([1,2]), 1, 2)

    # relation definition https://cloud.urbansim.com/docs/general/documentation/technical.html#pums-relp-variable-table
    relate = pd.DataFrame({'relshipp': [20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38], 
                        'relate':   [ 0, 1,13, 1,13, 2, 3, 4, 5, 6, 7, 8, 9,10,12,14,15,16,17]})
    synpop_pp = synpop_pp.merge(relate, on=["relshipp"])

    # education status definition https://cloud.urbansim.com/docs/general/documentation/technical.html#pums-schl-variable-table
    edu = pd.DataFrame({'schg': [0,1,2,3,4,5,6,7,8, 9,10,11,12,13,14,15,16], 
                        'edu':  [0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,19,21]})
    synpop_pp = synpop_pp.merge(edu, on=["schg"])
    synpop_pp['edu'] = np.where((synpop_pp['schg'].isin([14,15]) & (synpop_pp['eduatt']==2)), 16, synpop_pp['edu'])
    synpop_pp['edu'] = np.where((synpop_pp['schg'].isin([14,15,16]) & (synpop_pp['eduatt']==3)), 20, synpop_pp['edu'])
    synpop_pp['edu'] = np.where(((synpop_pp['schg']==15) & (synpop_pp['eduatt']==1)), 17, synpop_pp['edu']) # IS IT TRUE?
    synpop_pp['edu'] = np.where(((synpop_pp['schg']==15) & (synpop_pp['eduatt']==4)), 21, synpop_pp['edu'])
    synpop_pp['edu'] = np.where((synpop_pp['schg'].isin([15,16]) & synpop_pp['eduatt']==5), 22, synpop_pp['edu'])

    synpop_pp['worker'] = np.where(synpop_pp['worker']==1, 1, 0)
    synpop_pp['student'] = np.where(synpop_pp['schg']==0, 0, 1)
    synpop_pp = synpop_pp.drop(columns=['serialno']) 
    synpop_pp['hispanic'] = np.where(synpop_pp['race']==1, 1, 0)
    hh_worker = synpop_pp.groupby('household_id').agg({'worker': 'sum'}).rename(columns={"worker": "workers"})
    synpop_hh = synpop_hh.merge(hh_worker, how="left", on=["household_id"]).fillna(0)
    synpop_pp = synpop_pp.merge(addnm_pinc[["person_id", "earning"]], how="left", on=["person_id"]).fillna(0)
    
    synpop_hh['lcm_county_id'] = synpop_hh['lcm_county_id'].astype(str).str.zfill(3)
    synpop_hh['lcm_county_id'] = '06' + synpop_hh['lcm_county_id']
    synpop_hh['block_id'] = synpop_hh['htier2tazid'].astype(str)
    synpop_hh['TAZ'] = synpop_hh['block_id']

    synpop_pp["person_sex"] = synpop_pp["sex"].map({1: "male", 2: "female"})

    synpop_hh["hh_size"] = np.where( 
        synpop_hh["persons"] == 1,"one", 
        np.where(synpop_hh["persons"] == 2,"two", 
                np.where(synpop_hh["persons"] == 3, "three", 
                        "four or more"),
                ),
        )
    
    synpop_hh = synpop_hh.drop(columns=['puma10', 'htier2tazid', 'htier2tazseq', 'rt', 'htype', 'ten', 'hht', 'hht2']) 
    synpop_hh.to_csv(f"{OUT_DIR}/households.csv", index=False)
    synpop_pp.to_csv(f"{OUT_DIR}/persons.csv", index=False)