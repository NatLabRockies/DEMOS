import orca
orca.add_injectable('all_local', True)
orca.add_injectable('region_code', '06197001')
orca.add_injectable('calibrated_folder', 'custom')
orca.add_injectable('running_calibration_routine', False)
import datasources
import variables
import update_demos
import sys
import pandas as pd



p_df = orca.get_table('persons').to_frame(columns=['age', 'person_age', 'edu', 'MAR', 'race_id', 'race'])
print(p_df[p_df['age']==0].to_string())
h_df = orca.get_table('households').to_frame()

print(p_df)
print(h_df['hh_size'].unique())
print(h_df[['persons', 'children', 'hh_size', 'hh_children']])
print()

orca.eval_step('birth')
p_df = orca.get_table('persons').to_frame(columns=['age', 'person_age', 'edu'])
h_df = orca.get_table('households').to_frame()
print("after evaluation")
p_df=p_df[p_df['age']<15]
print(p_df)
print()
print(h_df[['persons', 'children', 'hh_size', 'hh_children']])
print()

