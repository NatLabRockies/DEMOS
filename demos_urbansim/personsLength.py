import orca
import pandas as pd
orca.add_injectable('all_local', True)
orca.add_injectable('region_code', '06197001')
orca.add_injectable('calibrated_folder', 'custom')
orca.add_injectable('running_calibration_routine', False)
import datasources
import variables
import update_demos

#Length of persons: 6802465
print(orca.get_table('persons').to_frame(columns='(Intercept)'))

households_df = orca.get_table('households')
cols = pd.Series(households_df.columns)

print(cols)