import orca
import pandas as pd

# Relational adjustment mapping (equivalent to relmap_06197001.csv)
# Rows = old relate of person changing household head; Columns = old relate of new head
# Values = new relate code to assign
RELATIONAL_ADJUSTMENT_MAPPING = pd.DataFrame(
    data=[
        [ 5,  5,  5, 10,  7,  6,  7,  1, 10, 11, 12,  5, 15, 16, 17],  # index=2
        [ 5,  5,  5, 10,  7,  6,  7,  1, 10, 11, 12,  5, 15, 16, 17],  # index=3
        [ 5,  5,  5, 10,  7,  6,  7,  1, 10, 11, 12,  5, 15, 16, 17],  # index=4
        [10, 10, 10,  5,  2, 10,  9, 10, 10, 11, 12, 10, 15, 16, 17],  # index=5
        [10, 10, 10,  6,  1, 10, 10, 10, 10, 11, 12, 10, 15, 16, 17],  # index=6
        [ 2,  2,  2, 10, 10,  5, 10, 10, 10, 11, 12,  2, 15, 16, 17],  # index=7
        [10, 10, 10,  8, 10, 10,  1,  6, 10, 11, 12, 10, 15, 16, 17],  # index=8
        [10, 10, 10,  9, 10, 10,  2,  5, 10, 11, 12, 10, 15, 16, 17],  # index=9
        [10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 12, 10, 15, 16, 17],  # index=10
        [11, 11, 11, 11, 11, 11, 11, 11, 10, 11, 12, 11, 15, 16, 17],  # index=11
        [12, 12, 12, 12, 12, 12, 12, 12, 10, 11, 12, 12, 15, 16, 17],  # index=12
        [ 5,  5,  5, 10,  7, 10, 10, 10, 10, 11, 12,  5, 15, 16, 17],  # index=14
        [15, 15, 15, 15, 15, 15, 15, 15, 10, 11, 12, 15, 15, 16, 17],  # index=15
        [16, 16, 16, 16, 16, 16, 16, 16, 10, 11, 12, 16, 15, 16, 17],  # index=16
        [17, 17, 17, 17, 17, 17, 17, 17, 10, 11, 12, 17, 15, 16, 17],  # index=17
    ],
    index=pd.Index([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17], name="index"),
    columns=["2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "14", "15", "16", "17"],
)
orca.add_table("relational_adjustment_mapping", RELATIONAL_ADJUSTMENT_MAPPING)

# FIPS Code to Income Quartile
STATE_QUARTILE_LABELS = {
    "01": 1,  # Alabama
    "02": 4,  # Alaska
    "04": 3,  # Arizona
    "05": 1,  # Arkansas
    "06": 4,  # California
    "08": 3,  # Colorado
    "09": 4,  # Connecticut
    "10": 3,  # Delaware
    "11": 4,  # District of Columbia
    "12": 3,  # Florida
    "13": 1,  # Georgia
    "15": 4,  # Hawaii
    "16": 3,  # Idaho
    "17": 2,  # Illinois
    "18": 1,  # Indiana
    "19": 1,  # Iowa
    "20": 1,  # Kansas
    "21": 2,  # Kentucky
    "22": 2,  # Louisiana
    "23": 3,  # Maine
    "24": 4,  # Maryland
    "25": 4,  # Massachusetts
    "26": 1,  # Michigan
    "27": 2,  # Minnesota
    "28": 1,  # Mississippi
    "29": 1,  # Missouri
    "30": 2,  # Montana
    "31": 2,  # Nebraska
    "32": 3,  # Nevada
    "33": 4,  # New Hampshire
    "34": 4,  # New Jersey
    "35": 2,  # New Mexico
    "36": 4,  # New York
    "37": 3,  # North Carolina
    "38": 1,  # North Dakota
    "39": 2,  # Ohio
    "40": 1,  # Oklahoma
    "41": 3,  # Oregon
    "42": 2,  # Pennsylvania
    "44": 4,  # Rhode Island
    "45": 2,  # South Carolina
    "46": 2,  # South Dakota
    "47": 1,  # Tennessee 
    "48": 2,  # Texas 
    "49": 3,  # Utah 
    "50": 4,  # Vermont 
    "51": 3,  # Virginia
    "53": 4,  # Washington
    "54": 1,  # West Virginia
    "55": 3,  # Wisconsin
    "56": 2,  # Wyoming
    "72": 3,  # Puerto Rico
}

