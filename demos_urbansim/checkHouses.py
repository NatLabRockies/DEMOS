import orca
orca.add_injectable('all_local', True)
orca.add_injectable('region_code', '06197001')
orca.add_injectable('calibrated_folder', 'custom')
orca.add_injectable('running_calibration_routine', False)
import datasources


persons = orca.get_table('persons').to_frame()
houses = orca.get_table('households').to_frame()
aggregates = persons.groupby('household_id').agg({'sex': 'count',
                                                    'earning': 'sum',
                                                    'worker': 'sum',
                                                    'race_id': 'first',
                                                    'age': 'first'})
print(aggregates['earning'].compare(houses['income']))
