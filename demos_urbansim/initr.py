import orca
orca.add_injectable('all_local', True)
orca.add_injectable('region_code', '06197001')
orca.add_injectable('calibrated_folder', 'custom')
orca.add_injectable('running_calibration_routine', False)
import datasources
import variables
import update_demos

