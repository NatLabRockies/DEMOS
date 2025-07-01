from ..estimated_models.binary_logit import BinaryLogitStep
from pydantic import BaseModel
from typing import Literal
import pandas as pd
import numpy as np

import orca

# TODO: Logging

class AbsoluteErrorCalibration(BaseModel):
    procedure_type: Literal["absolute_error"]
    observed_values_table: str
    tolerance: float
    max_iter: int = 20
    
    def calibrate_model(self, model: BinaryLogitStep, data: pd.DataFrame):
        # Retrive orca values
        year = orca.get_injectable("year")
        target_table = orca.get_table(self.observed_values_table)
        target_value = target_table[target_table["year"] == year]

        prediction = model.predict(data)
        ...


class RMSECalibration(BaseModel):
    procedure_type: Literal["rmse_error"]
    observed_values_table: str
    tolerance: float
    max_iter: int = 20
    logging_level: int = 20 # INFO

    def calibration_step(self, model: BinaryLogitStep, update_delta: float):
        model.fitted_parameters[0] += update_delta

    def calibrate_model(self, model: BinaryLogitStep, data: pd.DataFrame):
        # Retrive orca values
        year = orca.get_injectable("year")
        target_table = orca.get_table(self.observed_values_table).to_frame()
        target_value = target_table[target_table["year"] == year]["count"].sum() # TODO: Review

        prediction = model.predict(data)
        error = np.sqrt(np.mean((prediction.sum() - target_value)**2))

        total_iterations = 0
        while error > self.tolerance and total_iterations < self.max_iter:
            print(f"{total_iterations} iter: {error}")
            self.calibration_step(
                model,
                np.log(target_value / prediction.sum())
            )
            total_iterations += 1

            prediction = model.predict(data)
            error = np.sqrt(np.mean((prediction.sum() - target_value)**2))
        print(f"{total_iterations} iter: {error}")
        return prediction
