"""Datasources to fetch thresholds."""

from pathlib import Path
from typing import ClassVar, Self

import pandas as pd
import xarray as xr

from veriflow.configuration.default.datasources import CsvConfig
from veriflow.constants import DataSourceKind, DataType, StandardDim
from veriflow.datasources.base import BaseDatasource

__all__ = [
    "Csv",
    "CsvConfig",
]


class Csv(BaseDatasource):
    """Datasource for reading CSV files."""

    kind: str = DataSourceKind.CSV
    config_class = CsvConfig
    supported_data_types: ClassVar[set[DataType]] = {
        DataType.threshold,
    }

    def __init__(self, config: CsvConfig) -> None:
        self.config: CsvConfig = config
        self.dataset: xr.Dataset = xr.Dataset()

    def fetch_data(self) -> Self:
        """Parse thresholds from csv file."""
        file_path = Path(self.config.directory) / self.config.filename
        threshold_df = pd.read_csv(file_path)

        # Check that the df has the correct structure
        expected_columns = [
            StandardDim.station,
            StandardDim.variable,
            StandardDim.threshold,
            "value",
        ]
        if not all(k in expected_columns for k in threshold_df.columns):
            msg = f"Expected columns: {expected_columns}. Got: {threshold_df.columns}"
            raise ValueError(msg)

        # Pivot the long-form table into a Dataset where each unique variable becomes a
        # data variable with dims (station, threshold).
        pivoted = threshold_df.set_index(
            [StandardDim.station, StandardDim.variable, StandardDim.threshold],
        ).to_xarray()["value"]

        # Filter the array based on the configured station, variable and threshold ids
        try:
            pivoted = pivoted.sel(
                station=self.config.stations,
                variable=self.config.variables,
                threshold=self.config.thresholds,
            )
        except KeyError as e:
            msg = "One of the configured station, variable or threshold ids was not found in the . "
            f"data. Details: {e}"
            raise ValueError(msg) from e

        # Convert the variable dim into separate data variables, one per variable.
        dataset = pivoted.to_dataset(dim=StandardDim.variable)

        # Set the data type and source as attributes for later use in the verification process
        dataset.attrs["data_type"] = "threshold"  # type:ignore[misc]
        dataset.attrs["source"] = self.config.source  # type:ignore[misc]
        self.dataset = dataset
        return self
