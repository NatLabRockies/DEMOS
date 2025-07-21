from ..estimated_models.binary_logit import BinaryLogitStep
from pydantic import BaseModel
from typing import Literal
import pandas as pd
import numpy as np

import orca

# TODO: Logging
class RMSECalibration(BaseModel):
    procedure_type: Literal["rmse_error"]
    tolerance_type: Literal["relative", "absolute"] = "absolute"
    observed_values_table: str
    tolerance: float
    max_iter: int = 20
    logging_level: int = 20 # INFO

    def calibration_step(self, model: BinaryLogitStep, update_delta: float):
        model.fitted_parameters[0] += update_delta

    def compute_error(self, prediction: pd.Series, target: float):
        if self.tolerance_type == "relative":
            return np.sqrt(np.mean((prediction.sum() / len(prediction) - target)**2))
        
        elif self.tolerance_type == "absolute":
            return np.sqrt(np.mean((prediction.sum() - target)**2))
        
        raise NotImplementedError(f"Tolerance type {self.tolerance_type} not implemented")

    def calibrate_and_run_model(self, model: BinaryLogitStep, data: pd.DataFrame):
        table_column = "count" if self.tolerance_type == "absolute" else "share"

        # Sort for reproducibility
        data = data.sort_index(axis=0)

        # Retrive orca values
        year = orca.get_injectable("year")
        target_table = orca.get_table(self.observed_values_table).to_frame()
        target_value = target_table[target_table["year"] == year][table_column].iloc[0]

        prediction = model.predict(data)
        error = self.compute_error(prediction, target_value)

        total_iterations = 0
        while error > self.tolerance and total_iterations < self.max_iter:
            print(f"{total_iterations} iter: {error}")
            target_for_update = target_value if self.tolerance_type == "absolute" else target_value * len(data)
            self.calibration_step(
                model,
                np.log(target_for_update / prediction.sum())
            )
            total_iterations += 1

            prediction = model.predict(data)
            error = self.compute_error(prediction, target_value)
        print(f"{total_iterations} iter: {error}")
        return prediction


class SimultaneousCalibrationConfig(BaseModel):
    tolerance: float
    max_iter: int = 20
    logging_level: int = 20 # INFO

    # scaling_factor: float = 1.5
    learning_rate: float = 2.5
    momentum_weight: float = 0.3
