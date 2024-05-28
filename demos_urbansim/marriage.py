import orca
import pandas as pd


loc = 'https://s3-us-west-1.amazonaws.com/synthpop-data2/puma_p_06.csv'
types = {"PUMA10": "object",
                "PUMA00": "object",
                "ST": "object",
                "SERIALNO": 'str',
                "serialno": 'str'}
col_rename = {'PUMA10': 'puma10',
                'PUMA00': 'puma00',
                'SERIALNO': 'serialno',
              'SPORDER':'member_id'}
relevant_vars = ['serialno', 'member_id','AGEP','MAR']
# Age is just to make sure member_id is the correct matching
# MAR is the marital status as coded in https://data.census.gov/mdat/#/search?ds=ACSPUMS1Y2019
census_data = pd.read_csv(loc, dtype=types, low_memory=False).rename(columns = col_rename)[relevant_vars]

store = pd.HDFStore('data/custom_mpo_06197001_model_data.h5')
persons_df = store['persons'].copy()
households_df = store['households'].copy()
persons_df = pd.merge(persons_df, households_df[['serialno']], on='household_id', how='left')
persons_df.index.name = "person_id"
census_data['serialno'] = census_data['serialno'].astype(int)
persons_df = pd.merge(persons_df, census_data[['serialno', 'MAR']], on='serialno', how='left')
persons_df.index.name = "person_id"
persons_df.drop(columns=['serialno'], inplace=True)

pd.set_option("display.max_columns", 100)
print(persons_df.head(10))
store['persons'] = persons_df
store.close()

