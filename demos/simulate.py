import argparse
# import os

import numpy as np
import orca
import pandas as pd
from templates import modelmanager as mm
from config import load_config_file, get_config


def run():
    CONFIG = get_config()
    
    orca.add_table('run_times', pd.DataFrame())
    orca.add_table('marital_rebalanced', pd.DataFrame())
    orca.add_table('marital_status_output', pd.DataFrame())

    import datasources
    import models
    import variables

    if CONFIG.random_seed is not None:
        np.random.seed(CONFIG.random_seed)

    mm.initialize(datasources.configs_folder)

    out_tables = datasources.hdf_tables + ["graveyard", "run_times", "marital_rebalanced", "marital_status_output"]
    iter_vars = list(range(CONFIG.base_year + 1, CONFIG.forecast_year + 1, 1))
    orca.run(["fix_persons_table"])
    orca.run(
        orca.get_injectable('sim_steps'),
        data_out=CONFIG.output_fname,
        iter_vars=iter_vars,
        out_base_tables=[],
        out_run_tables=out_tables,
        out_run_local=True,
        out_interval= 1
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-cfg", "--config_file", type=str, help="TOML config file")
    args = parser.parse_args()

    # Load config file
    load_config_file(args.config_file)
    run()
