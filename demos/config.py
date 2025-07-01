import toml
import orca
from pydantic import BaseModel
from typing import Optional
from templates.calibration import CalibrationConfig

class BirthModuleConfig(BaseModel):
    calibration_procedure: Optional[CalibrationConfig] = None

class DEMOSConfig(BaseModel):
    birth_module_config: BirthModuleConfig
    ...

def load_config_file(dir: str) -> DEMOSConfig:
    orca.add_injectable("demos_config", DEMOSConfig(**toml.load(dir)))