import os
import toml
import orca
import pandas as pd
from pydantic import BaseModel, model_validator, Field
from typing import Literal, Optional
from loguru import logger
from templates.calibration import CalibrationConfig, SimultaneousCalibrationConfig
from datasources import DataSourceModel

CONFIG = None

class HHRebalancingModuleConfig(BaseModel):
    """
    Configuration for Household Rebalancing module
    """
    control_table: str
    control_col: str
    geoid_col: str

class EmploymentModuleConfig(BaseModel):
    simultaneous_calibration_config: Optional[SimultaneousCalibrationConfig] = None
    enter_model_calibration_procedure: Optional[CalibrationConfig] = None
    exit_model_calibration_procedure: Optional[CalibrationConfig] = None

    @model_validator(mode="after")
    def check_calibration_config_exclusivity(self):
        sim_cal = self.simultaneous_calibration_config is not None
        enter_cal = self.enter_model_calibration_procedure is not None
        exit_cal = self.exit_model_calibration_procedure is not None
        if sim_cal and (enter_cal or exit_cal):
            raise ValueError(f"Simultaneous calibration cannot be used at the same time as " + \
                             f"individual model calibration. Simultaneous selected: {sim_cal}, " + \
                             f"EnterModel selected: {enter_cal}, ExitModel selected: {exit_cal}")
        return self

class HHReorgModuleConfig(BaseModel):
    simultaneous_calibration_config: Optional[SimultaneousCalibrationConfig] = None
    geoid_col: Optional[str] = None

class MortalityModuleConfig(BaseModel):
    calibration_procedure: Optional[CalibrationConfig] = None

class BirthModuleConfig(BaseModel):
    calibration_procedure: Optional[CalibrationConfig] = None

class AgingModuleConfig(BaseModel):
    """
    """
    #: Age at which a person qualifies as senior
    senior_age: int = 65


class DEMOSConfig(BaseModel):
    """
    Global configuration for DEMOS. Individual fields in this class control the configuration of each module.
    """
    random_seed: int

    #: Year represented in synthetic population input
    base_year: int
    #: Last year of simulation
    forecast_year: int = 2020
    #: Path to DEMOS outputs
    output_dir: str = "../data/output"
    #: Name of output HDF5 file. Defaults to `demos_output_{forecast_year}.h5`.
    output_fname: str = None
    #: List of orca tables to include in output
    output_tables: list[str] = None
    #: Path to directory with calibration models
    calibrated_models_dir: str = None
    #: Behavior of inconsistent `persons` input table
    inconsistent_persons_table_behavior: Literal["error", "fix", "ignore"] = "error"
    #: Name of tables to be initialized as empty
    initialize_empty_tables: list[str] = None

    #: List of tables to be loaded into orca
    tables: Optional[list[DataSourceModel]] = None
    #: List of modules to be run
    modules: Optional[list[str]] = None

    # Module-specific config
    aging_module_config: AgingModuleConfig = Field(default_factory=AgingModuleConfig)
    employment_module_config: EmploymentModuleConfig = Field(default_factory=EmploymentModuleConfig)
    hh_reorg_module_config: HHReorgModuleConfig = Field(default_factory=HHReorgModuleConfig)
    mortality_module_config: MortalityModuleConfig = Field(default_factory=MortalityModuleConfig)
    birth_module_config: BirthModuleConfig = Field(default_factory=BirthModuleConfig)
    hh_rebalancing_module_config: HHRebalancingModuleConfig = Field(default_factory=HHRebalancingModuleConfig)
    
    def model_post_init(self, __context) -> None:
        if self.output_fname is None:
            self.output_fname = f"{self.output_dir}/demos_output_{self.forecast_year}.h5"
            os.makedirs(self.output_dir, exist_ok=True)
            logger.info(f"Output file set to default: {self.output_fname}")
        
        if self.output_tables is None:
            self.output_tables = []
            
        if self.initialize_empty_tables is None:
            self.initialize_empty_tables = []
        
        # Load all table datasources
        for t in self.tables:
            t.load_into_orca()

        for n in self.initialize_empty_tables:
            orca.add_table(n, pd.DataFrame())
        
        if self.modules is None:
            self.modules = [
                "aging",
                "laborforce_model",
                "households_reorg",
                "kids_moving_model",
                "fatality_model",
                "birth_model",
                "education_model",
                "household_rebalancing",
                "update_income"
            ]
        

    @model_validator(mode='after')
    def require_persons_and_households(self):
        loaded_table_names = [t.table_name for t in self.tables]
        if "persons" not in loaded_table_names or "households" not in loaded_table_names:
            raise ValueError(f"Both 'persons' and 'households' tables are required. Tables defined: {loaded_table_names}")
        return self

def load_config_file(dir: str) -> DEMOSConfig:
    global CONFIG
    CONFIG = DEMOSConfig(**toml.load(dir))

def get_config():
    global CONFIG
    if CONFIG is None:
        CONFIG = DEMOSConfig()
    return CONFIG
