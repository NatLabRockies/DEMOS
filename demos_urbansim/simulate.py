import argparse
import grp
import os
import pwd

import numpy as np
import orca
import pandas as pd
from urbansim_templates import modelmanager as mm
from urbansim_templates.models import LargeMultinomialLogitStep, OLSRegressionStep


def run(
        region_code, initial_run, base_year, forecast_year, random_seed,
        calibrated, calibrated_folder, multi_level, segmented, capacity_boost,
        all_local, freq_interval, output_fname, skim_source, random_match, table_save, scenario_name):
    orca.add_injectable('running_calibration_routine', False)
    orca.add_injectable('local_simulation', True)
    orca.add_injectable('initial_run', initial_run)
    orca.add_injectable('region_code', region_code)
    orca.add_injectable('base_year', base_year)
    orca.add_injectable('forecast_year', forecast_year)
    orca.add_injectable('calibrated', calibrated)
    orca.add_injectable('calibrated_folder', calibrated_folder)
    orca.add_injectable('multi_level_lcms', multi_level)
    orca.add_injectable('segmented_lcms', segmented)
    orca.add_injectable('capacity_boost', capacity_boost)
    orca.add_injectable('all_local', all_local)
    orca.add_injectable('table_save', table_save)
    orca.add_injectable('skim_source', skim_source)
    orca.add_injectable('random_match', random_match)
    orca.add_injectable('scenario_name', scenario_name)

    import datasources
    import models
    import variables

    if random_seed:
        np.random.seed(random_seed)

    calibrated_path = os.path.join(
        'calibrated_configs/', calibrated_folder, region_code)
    if os.path.exists(os.path.join('configs', calibrated_path, skim_source)):
        calibrated_path = os.path.join(calibrated_path, skim_source)
    configs_folder = calibrated_path if orca.get_injectable('calibrated') else 'estimated_configs'
    mm.initialize('configs/' + configs_folder)
    orca.run(orca.get_injectable('pre_processing_steps'))

    # for table in models.demos_tables:
    #     orca.add_table(table, pd.DataFrame())
    # print(datasources.hdf_tables)
    if table_save:
        out_tables = datasources.hdf_tables + ["graveyard"]
    else:
        out_tables = datasources.hdf_tables + ["graveyard"] #TODO: FIX THIS
    iter_vars = list(range(
        base_year + freq_interval, forecast_year + freq_interval, freq_interval))
    orca.run(
        orca.get_injectable('sim_steps'),
        data_out=output_fname,
        iter_vars=iter_vars,
        out_base_tables=[],
        out_run_tables=out_tables,
        out_run_local=True,
        out_interval= 1
    )
    # update_demos.dead_facts()
    # update_demos.calculate(True)
    # update_demos.export('pop_over_time')
    # update_demos.export('age_over_time')
    # update_demos.export('edu_over_time')
    # update_demos.export('households')
    # update_demos.export('graveyard')
    # orca.run(orca.get_injectable('export_demo_stats'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--region_code", type=str, help="region fips code")
    parser.add_argument("--initial_run", action="store_true", help="generate calibration/validation charts")
    parser.add_argument("-y", "--year", type=int, help="forecast year to simulate to")
    parser.add_argument("-s", "--random_seed", type=int, help="value to set as random seed")
    parser.add_argument("-c", "--calibrated", action="store_true", help="whether to run with calibrated coefficients")
    parser.add_argument("-cf", "--calibrated_folder", type=str, help="name of the calibration folder to read configs from")
    parser.add_argument("-sl", "--single_level_lcms", action="store_true", help="run with single_level LCMs")
    parser.add_argument("-sg", "--segmented", action="store_true", help="run with segmented LCMs")
    parser.add_argument("-b", "--capacity_boost", type=int, help="value to multiply capacities during simulation")
    parser.add_argument("-l", "--all_local", action="store_true", help="no cloud access whatsoever")
    parser.add_argument("-i", "--input_year", type=int, help="input data (base) year")
    parser.add_argument("-f", "--freq_interval", type=int, help="intra-simulation frequency interval")
    parser.add_argument("-o", "--output_fname", type=str, help="output file name")
    parser.add_argument("-t", "--travel_model", type=str, help="source of skims data. e.g. beam, polaris")
    parser.add_argument("-ts", "--table_save", action="store_true", help="store all other generated tables")
    parser.add_argument("-rm", "--random_matching", action="store_true", help="random matching in marriage")
    parser.add_argument("-sn", "--scenario_name", type=str, help="name of scenario of simulation")

    args = parser.parse_args()
    region_code = args.region_code
    initial_run = args.initial_run if args.initial_run else False
    base_year = args.input_year if args.input_year else 2010
    forecast_year = args.year if args.year else 2020
    freq_interval = args.freq_interval if args.freq_interval else 1
    random_seed = args.random_seed if args.random_seed else False
    calibrated = args.calibrated if args.calibrated else False
    calibrated_folder = args.calibrated_folder if args.calibrated_folder \
        else 'multilevel_segmented_grouped_controls_200iters_0.05step'
    multi_level = False if args.single_level_lcms else True
    segmented = True if args.segmented else False
    capacity_boost = args.capacity_boost if args.capacity_boost else 1
    all_local = args.all_local if args.all_local else False
    table_save = args.table_save if args.table_save else False
    random_match = args.random_matching if args.random_matching else False
    skim_source = args.travel_model if args.travel_model else 'beam'
    scenario_name = args.scenario_name if args.scenario_name else False
    output_fname = args.output_fname if args.output_fname \
        else "data/model_data_{0}.h5".format(forecast_year)

    run(
        region_code, initial_run, base_year, forecast_year, random_seed,
        calibrated, calibrated_folder, multi_level, segmented, capacity_boost,
        all_local, freq_interval, output_fname, skim_source, random_match, table_save, scenario_name)

    # make sure output data has same permissions as input (only an
    # issue when running from inside docker which will execute this
    # script as root)
    input_data_name = orca.get_injectable('data_name')
    data_stats = os.stat('data/{0}'.format(input_data_name))
    uid = data_stats.st_uid
    gid = data_stats.st_gid
    # breakpoint()
    # uid = pwd.getpwnam(usernmae).pw_uid
    # gid = grp.getgrnam(groupname).gr_gid
    os.chown(output_fname, uid, gid)