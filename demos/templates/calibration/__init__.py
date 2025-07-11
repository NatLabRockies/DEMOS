from .procedures import *

from pydantic import Field
from typing import Union
from typing_extensions import TypeAlias, Annotated

# This defines the type `CalibrationConfig` in way that can be automatically parsed from a config file
CalibrationConfig: TypeAlias = Annotated[Union[RMSECalibration], Field(discriminator="procedure_type")]