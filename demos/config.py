import toml
import orca
from pydantic import BaseModel
from typing import Optional
from templates.calibration import CalibrationConfig

CONFIG = None

class MortalityModuleConfig(BaseModel):
    calibration_procedure: Optional[CalibrationConfig] = None

class EmploymentModuleConfig(BaseModel):
    enter_model_calibration_procedure: Optional[CalibrationConfig] = None
    exit_model_calibration_procedure: Optional[CalibrationConfig] = None

class BirthModuleConfig(BaseModel):
    calibration_procedure: Optional[CalibrationConfig] = None

class DEMOSConfig(BaseModel):
    # Global config
    region_code: str
    forecast_year: int = 2020
    random_seed: int
    base_year: int
    output_fname: str = None
    calibrated_folder: str = "custom"

    # Module-specific config
    employment_module_config: EmploymentModuleConfig
    mortality_module_config: MortalityModuleConfig
    birth_module_config: BirthModuleConfig
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.output_fname is None:
            self.output_fname = "data/model_data_{0}.h5".format(self.forecast_year)


    def update(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

def load_config_file(dir: str) -> DEMOSConfig:
    global CONFIG
    CONFIG = DEMOSConfig(**toml.load(dir))

def set_config(params_dict: dict):
    global CONFIG
    CONFIG.update(**vars(params_dict))

def get_config():
    global CONFIG
    return CONFIG