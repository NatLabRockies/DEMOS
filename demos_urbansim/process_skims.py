import pandas as pd
import geopandas

beam_skims_types = {'timePeriod': str,'pathType': str,'origin': int,'destination': int,'TIME_minutes': float, 'TOTIVT_IVT_minutes': float, 'VTOLL_FAR': float, 'DIST_meters': float, 'WACC_minutes': float, 'WAUX_minutes': float, 'WEGR_minutes': float, 'DTIM_minutes': float, 'DDIST_meters': float, 'KEYIVT_minutes': float, 'FERRYIVT_minutes': float, 'BOARDS': float, 'DEBUG_TEXT': str}


print('READING')
chunk = pd.read_csv('data/bay_area_skims.csv.gz', compression="gzip", dtype=beam_skims_types, chunksize=1000000)
print('CONCATENATING')
df = pd.concat(chunk)
print('FILTERING')
sub_df = df.loc[(df['pathType'] == 'SOV') & (df['timePeriod'] == 'AM')]
print('EXPORTING')
sub_df.to_csv('data/auto_skims.csv')
print('DONE')
chunk  = pd.read_csv('data/auto_skims.csv', dtype=beam_skims_types, chunksize=1000000)
df = pd.concat(chunk)
df = df[['origin', 'destination', 'TOTIVT_IVT_minutes', 'DIST_meters']].copy()
df = df.rename(columns={'origin': 'from_zone_id', 'destination': 'to_zone_id', 'TOTIVT_IVT_minutes': 'SOV_AM_IVT_mins'})
df['from_zone_id'] = df['from_zone_id'].astype('str')
df['to_zone_id'] = df['to_zone_id'].astype('str')
df = df.set_index(['from_zone_id', 'to_zone_id'])
store = pd.HDFStore('data/custom_mpo_06197001_model_data.h5')
store['travel_data'] = df




def correct_index(index_int):
    if type(index_int)==type('string'):
        return index_int
    return '0'+str(index_int)


blocks = store['blocks'].copy()
block_taz = pd.read_csv('data/block_w_taz.csv', dtype={'GEOID10': object, 'taz1454': object})
block_taz = block_taz.rename(columns={'GEOID10': 'block_id', 'taz1454':'TAZ'})
block_taz = block_taz.set_index('block_id')[['TAZ']]
block_taz.index = block_taz.index.map(correct_index)

blocks = blocks.join(block_taz)
blocks['TAZ'] = blocks['TAZ'].fillna(0)

blocks = blocks[blocks['TAZ']!=0].copy()

blocks = blocks.rename(columns = {'TAZ': 'zone_id'})

households = store['households'].copy()
persons = store['persons'].copy()
jobs = store['jobs'].copy()
units = store['residential_units'].copy()
households = households[households['block_id'].isin(blocks.index)].copy()
persons = persons[persons['household_id'].isin(households.index)].copy()
jobs = jobs[jobs['block_id'].isin(blocks.index)].copy()
units = units[units['block_id'].isin(blocks.index)].copy()


store['blocks'] = blocks
store['households'] = households
store['persons'] = persons
store['jobs'] = jobs
store['residential_units'] = units
store.close()
print("Done!")






