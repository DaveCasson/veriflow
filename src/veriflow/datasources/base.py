"""Module with the base class that all datasources should inherit from."""

import hashlib
from abc import abstractmethod
from os import R_OK, access
from pathlib import Path
from typing import ClassVar, Self

import xarray as xr

from veriflow.base import Base
from veriflow.configuration.base import (
    BaseDatasourceConfig,
)
from veriflow.configuration.utils import ForecastPeriods, TimePeriod
from veriflow.constants import FORECAST_DATA_TYPES, HISTORICAL_DATA_TYPES, DataType, StandardDim

__all__ = [
    "BaseDatasource",
    "BaseDatasourceConfig",
]


class BaseDatasource(Base):
    """Class to inherit from, defines the required methods and attributes."""

    kind: str = ""
    config_class: type[BaseDatasourceConfig] = BaseDatasourceConfig
    supported_data_types: ClassVar[set[DataType]] = set()

    def __init__(self, config: BaseDatasourceConfig) -> None:
        self.config: BaseDatasourceConfig = config
        self.data_type = config.data_type
        self.dataset: xr.Dataset = xr.Dataset()

    @property
    def data_type(self) -> DataType:
        """Whether the instance represents sim or obs data."""
        return self.config.data_type

    @data_type.setter
    def data_type(self, new_data_type: DataType) -> None:
        if new_data_type not in self.supported_data_types:
            msg = (
                f"Data type '{new_data_type}' is not supported ",
                f"by {self.__class__.__name__}",
            )
            raise NotImplementedError(msg)

        self._data_type = new_data_type

    @abstractmethod
    def fetch_data(self) -> Self:
        """Fetch data from datasource."""

    @staticmethod
    def _validate_cache_dir_exists_and_accessible(cache_dir: Path) -> None:
        """Check that cache dir exists and is a directory."""
        if not cache_dir.exists():
            cache_dir.mkdir(parents=True)
        elif not cache_dir.is_dir() and access(cache_dir, R_OK):
            msg = "Cache directory is not an accessible directory."
            raise NotADirectoryError(msg)

    @staticmethod
    def _validate_data_type(dataset: xr.Dataset, expected_data_type: DataType) -> None:
        # Check that the datatype is defined, and consistent with the config
        if "data_type" not in dataset.attrs:  # type:ignore[misc]
            msg = "The fetched dataset does not have a 'data_type' attribute."
            raise ValueError(msg)
        if dataset.attrs["data_type"] != expected_data_type:  # type:ignore[misc]
            msg = (
                f"The data type of the fetched dataset "
                f"({dataset.attrs['data_type']}) does not match the expected data "  # type:ignore[misc]
                f"type ({expected_data_type})."
            )
            raise ValueError(msg)

    @staticmethod
    def _validate_source(dataset: xr.Dataset, expected_source: str) -> None:
        # Make sure the source attribute is set to the expected source
        dataset.attrs["source"] = expected_source  # type:ignore[misc]

    @staticmethod
    def _validate_forecast_periods(
        dataset: xr.Dataset,
        forecast_periods: ForecastPeriods | None,
    ) -> None:
        """Check that forecast periods are provided for forecast data types."""
        if not forecast_periods and dataset.attrs["data_type"] in FORECAST_DATA_TYPES:  # type:ignore[misc]
            msg = (
                "Forecast periods must be provided in the config for forecast data types, "
                "but got None."
            )
            raise ValueError(msg)

    @staticmethod
    def _filter_forecast_periods(
        dataset: xr.Dataset,
        forecast_periods: ForecastPeriods | None,
    ) -> xr.Dataset:
        """Filter forecast dataset on forecast periods."""
        if dataset.attrs["data_type"] in FORECAST_DATA_TYPES and forecast_periods is not None:  # type:ignore[misc]
            # Select only relevant forecast periods for simulations
            dataset = dataset.sel(
                forecast_period=forecast_periods.timedelta64,
            )
        return dataset

    @staticmethod
    def _filter_times(dataset: xr.Dataset, verification_period_on_time: TimePeriod) -> xr.Dataset:
        """Filter the times outside the verification period and forecast periods."""
        data_type = dataset.attrs["data_type"]  # type:ignore[misc]

        if data_type in FORECAST_DATA_TYPES:  # type:ignore[misc]
            # Mask and drop time values outside of the configured vp
            filtered = dataset.where(
                (dataset[StandardDim.time] >= verification_period_on_time.start_datetime64)
                & (dataset[StandardDim.time] <= verification_period_on_time.end_datetime64),
            )
            # Drop NaN values along frt and fp dims, if all values are NaN
            return filtered.dropna(dim=StandardDim.forecast_reference_time, how="all").dropna(
                dim=StandardDim.forecast_period,
                how="all",
            )
        if data_type in HISTORICAL_DATA_TYPES:  # type:ignore[misc]
            # Mask and drop time values outside of the configured vp
            # Historical data type
            dataset = dataset.sel(
                {
                    StandardDim.time: slice(  # type:ignore[misc]
                        verification_period_on_time.start,
                        verification_period_on_time.end,
                    ),
                },
            )
        return dataset

    def get_data(self) -> Self:
        """Get cached data, or fetch and cache."""
        config_json = self.config.model_dump_json().encode("utf-8")
        config_hash = hashlib.sha256(config_json).hexdigest()

        cache_dir = Path(self.config.general.cache_dir)

        # Validate cache directory
        self._validate_cache_dir_exists_and_accessible(cache_dir)

        # Define file path for caching
        cached_dataset_path = cache_dir / f"{self.__class__.__name__}_{config_hash}.nc"

        if cached_dataset_path.exists():
            self.dataset = xr.open_dataset(cached_dataset_path)
            return self

        # Go fetch and cache
        self.fetch_data()

        dataset = self.dataset

        self._validate_data_type(dataset, self.data_type)
        self._validate_source(dataset, self.config.source)

        # Apply re-naming based on configured id mapping, if not None
        if self.config.id_mapping is not None:
            dataset = self.config.id_mapping.rename_dataset(dataset)

        # Check that forecast periods are provided for forecast data types, if not already checked
        #   in validation of config
        self._validate_forecast_periods(dataset, self.config.forecast_periods)

        # Filter forecast_periods
        dataset = self._filter_forecast_periods(
            dataset,
            self.config.forecast_periods,
        )

        # Filter times outside verification period
        dataset = self._filter_times(
            dataset,
            self.config.verification_period_on_time,
        )

        # Cache
        dataset.to_netcdf(cached_dataset_path)

        # Re-open to read from cache and prevent links to original files from which the dataset
        #   was loaded
        dataset_reloaded = xr.open_dataset(cached_dataset_path)

        # Explicitly close original backing files
        if hasattr(dataset, "close"):
            dataset.close()

        # Re-assign from cache
        self.dataset = dataset_reloaded
        return self
