import orca
import argparse
import numpy as np
import pandas as pd
from tables import HDF5ExtError
from templates import modelmanager as mm
from logging_logic import _StdoutToLoguru
from config import load_config_file, get_config
from loguru import logger
import contextlib

def run():
    # Load references and config
    import models
    import variables
    CONFIG = get_config()

    # Initialization
    mm.initialize(CONFIG.calibrated_models_dir)
    orca.add_table('run_times', pd.DataFrame())
    orca.add_table('marital_rebalanced', pd.DataFrame())
    orca.add_table('marital_status_output', pd.DataFrame())

    if CONFIG.random_seed is not None:
        np.random.seed(CONFIG.random_seed)

    # orca logs management, which by default are print statements.
    # This class allows us to capture all stdout and redirect it through
    # loguru with a custom and consistent format
    stdout_to_log = _StdoutToLoguru(level="INFO", prefix="[external/orca] ")
    with contextlib.redirect_stdout(stdout_to_log):
        orca.run(["validate_persons_table"])

        # Execute DEMOS, add error handling for common IO error
        iter_vars = list(range(CONFIG.base_year + 1, CONFIG.forecast_year + 1, 1))    
        try:
            orca.run(
                orca.get_injectable('sim_steps'),
                data_out=CONFIG.output_fname,
                iter_vars=iter_vars,
                out_base_tables=[],
                out_run_tables=CONFIG.output_tables,
                out_run_local=True,
                out_interval= 1
            )
        except HDF5ExtError as e:
            logger.error(f"Error using the HDF5 interface. This typically occurs when the output file already exists and is corrupt. Try moving/renaming/deleting {CONFIG.output_fname}")
            logger.error(e)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-cfg", "--config_file", type=str, help="TOML config file")
    args = parser.parse_args()

    # Load config file
    load_config_file(args.config_file)
    run()
