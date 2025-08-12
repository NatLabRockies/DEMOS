import orca
import time
import pandas as pd
import logging
import inspect
from loguru import logger

class InterceptHandler(logging.Handler):
    def __init__(self, target_name: str, prefix: str):
        super().__init__()
        self.target_name = target_name
        self.prefix = prefix

    def emit(self, record: logging.LogRecord) -> None:
        # 1. map stdlib level to Loguru level
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # 2. compute correct stack depth so Loguru shows your caller, not the Handler
        frame, depth = inspect.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        # 3. optionally tag or prefix messages from your library
        msg = record.getMessage()
        if record.name == self.target_name:
            msg = f"{self.prefix} {msg}"

        # 4. re-emit through Loguru
        logger.opt(depth=depth, exception=record.exc_info).log(level, msg)

def capture_orca_logs():
    LIB_NAME = "orca.orca"
    PREFIX   = "[ORCA]"
    handler  = InterceptHandler(target_name=LIB_NAME, prefix=PREFIX)

    lib_logger = logging.getLogger(LIB_NAME)
    lib_logger.handlers = [handler]
    lib_logger.propagate = False


def log_execution_time(start_time, year, module_name):
    now = time.time()
    run_table = orca.get_table('run_times')
    run_table.local = pd.concat([run_table.local,
                                pd.DataFrame([[year, module_name, now - start_time]],
                                            columns=["year", "module", "walltime"])
                                ])