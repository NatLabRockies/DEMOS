import orca
import argparse
import numpy as np
import pandas as pd
from templates import modelmanager as mm
from logging_logic import capture_orca_logs
from config import load_config_file, get_config


def run():
    CONFIG = get_config()
    capture_orca_logs()

    orca.add_table('run_times', pd.DataFrame())
    orca.add_table('marital_rebalanced', pd.DataFrame())
    orca.add_table('marital_status_output', pd.DataFrame())

    import models
    import variables

    if CONFIG.random_seed is not None:
        np.random.seed(CONFIG.random_seed)

    mm.initialize(CONFIG.calibrated_models_dir)
    iter_vars = list(range(CONFIG.base_year + 1, CONFIG.forecast_year + 1, 1))
    orca.run(["validate_persons_table"])
    orca.run(
        orca.get_injectable('sim_steps'),
        data_out=CONFIG.output_fname,
        iter_vars=iter_vars,
        out_base_tables=[],
        out_run_tables=CONFIG.output_tables,
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
