import argparse
# import os

import numpy as np
import orca
import pandas as pd
from templates import modelmanager as mm
from config import load_config_file, set_config


def run(region_code, base_year, forecast_year, random_seed,
        output_fname):
    orca.add_injectable('region_code', region_code)
    orca.add_injectable('base_year', base_year)
    orca.add_injectable('forecast_year', forecast_year)

    orca.add_table('run_times', pd.DataFrame())
    orca.add_table('marital_rebalanced', pd.DataFrame())

    import datasources
    import models
    import variables

    if random_seed:
        np.random.seed(random_seed)

    mm.initialize(datasources.configs_folder)

    out_tables = datasources.hdf_tables + ["graveyard", "run_times", "marital_rebalanced"]
    iter_vars = list(range(base_year + 1, forecast_year + 1, 1))
    orca.run(["fix_persons_table"])
    orca.run(
        orca.get_injectable('sim_steps'),
        data_out=output_fname,
        iter_vars=iter_vars,
        out_base_tables=[],
        out_run_tables=out_tables,
        out_run_local=True,
        out_interval= 1
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-cfg", "--config_file", type=str, help="TOML config file")

    # CLI arguments will override config file. TODO: Should we just delete them?
    parser.add_argument("-r", "--region_code", type=str, help="region fips code", required=False)
    parser.add_argument("-y", "--forecast_year", type=int, help="forecast year to simulate to", required=False)
    parser.add_argument("-s", "--random_seed", type=int, help="value to set as random seed", required=False)
    parser.add_argument("-i", "--base_year", type=int, help="input data (base) year", required=False)
    parser.add_argument("-o", "--output_fname", type=str, help="output file name", required=False)
    args = parser.parse_args()

    # Load config file
    load_config_file(args.config_file)
    set_config(args)

    region_code = args.region_code
    base_year = args.input_year
    forecast_year = args.year
    random_seed = args.random_seed
    output_fname = args.output_fname if args.output_fname else "data/model_data_{0}.h5".format(forecast_year)
    
    run(region_code, base_year, forecast_year, random_seed, output_fname)
