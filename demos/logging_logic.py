import orca
import time
import pandas as pd
from loguru import logger
import sys, contextlib


class _StdoutToLoguru:
    def __init__(self, level="INFO", prefix="[external] "):
        self.level = level
        self.prefix = prefix
        self._buf = ""

    def write(self, msg):
        # buffer to handle partial writes / no trailing newline
        self._buf += msg
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line = line.rstrip()
            if line:
                logger.opt(depth=1).log(self.level, f"{self.prefix}{line}")

    def flush(self):
        if self._buf.strip():
            logger.opt(depth=1).log(self.level, f"{self.prefix}{self._buf.rstrip()}")
        self._buf = ""


def log_execution_time(start_time, year, module_name):
    now = time.time()
    run_table = orca.get_table("run_times")
    run_table.local = pd.concat(
        [
            run_table.local,
            pd.DataFrame(
                [[year, module_name, now - start_time]],
                columns=["year", "module", "walltime"],
            ),
        ]
    )
