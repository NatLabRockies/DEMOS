import orca

orca.add_injectable('all_local', True)
orca.add_injectable('region_code', '06197001')
orca.add_injectable('calibrated_folder', 'custom')
orca.add_injectable('running_calibration_routine', False)
import datasources
import variables

import update_demos

person_df = orca.get_table('persons').to_frame()
person_df.sort_values('household_id', inplace=True)
person_df.head(100).to_csv('persons_sample.csv')

pop_size = len(person_df.index)

orca.add_injectable('fatality_passed', [0] * pop_size)

orca.eval_step('fatality_test')
