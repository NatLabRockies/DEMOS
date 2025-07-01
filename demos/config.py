import toml
import orca
from pydantic import BaseModel
from typing import Optional
from templates.calibration import CalibrationConfig

class BirthModuleConfig(BaseModel):
    calibration_procedure: Optional[CalibrationConfig] = None

class DEMOSConfig(BaseModel):
    # Global config
    region_code: str
    forecast_year: int = 2020
    random_seed: int
    base_year: int
    output_fname: str = None

    # Module-specific config
    birth_module_config: BirthModuleConfig
    
    def __init__(self):
        if self.output_fname is None:
            self.output_fname = "data/model_data_{0}.h5".format(self.forecast_year)

def load_config_file(dir: str) -> DEMOSConfig:
    orca.add_injectable("demos_config", DEMOSConfig(**toml.load(dir)))


def get_config():
    return orca.get_injectable("demos_config")

def set_config(params_dict: dict):
    get_config().local.update(params_dict)