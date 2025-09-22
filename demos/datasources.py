import orca
import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field
from typing import Annotated, Literal, Optional, Dict


class CSVTableSource(BaseModel):
    """"""

    file_type: Literal["csv"]
    #: Path to source file
    filepath: str
    #: Column in the file to be used as index (e.g. `person_id`)
    index_col: str
    #: Identifier of the table in orca
    table_name: str
    #: Delimiter character (pass down to pandas)
    delimiter: Optional[str] = None
    #: Apply custom casting to incoming values when reading CSV
    custom_dtype_casting: Optional[Dict[str, str]] = None

    def load_into_orca(self):
        logger.info(f"Loading CSV '{self.table_name}' table from {self.filepath}")
        df = pd.read_csv(
            self.filepath, delimiter=self.delimiter, dtype=self.custom_dtype_casting
        ).set_index(self.index_col)
        orca.add_table(self.table_name, df)


class H5TableSource(BaseModel):
    """"""

    file_type: Literal["h5"]
    #: Path to source file
    filepath: str
    #: key in the source HDF5 to be loaded
    h5_key: str
    #: Identifier of the table in orca
    table_name: str

    def load_into_orca(self):
        logger.info(
            f"Loading HDF5 '{self.table_name}' table from {self.filepath}/{self.h5_key}"
        )
        df = pd.read_hdf(self.filepath, key=self.h5_key)
        orca.add_table(self.table_name, df)


DataSourceModel = Annotated[
    H5TableSource | CSVTableSource, Field(discriminator="file_type")
]
