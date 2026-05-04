"""Module with the base class that all datasources should inherit from."""

import hashlib
from abc import abstractmethod
from os import R_OK, access
from typing import ClassVar, Self

import xarray as xr

from veriflow.base import Base
from veriflow.configuration.base import (
    BaseDatasourceConfig,
)
from veriflow.configuration.utils import TimePeriod
from veriflow.constants import FORECAST_DATA_TYPES, DataType, StandardDim

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
    def data_type(self) -> str:
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
    def _drop_times_outside_vp(
        ds: xr.Dataset,
        verification_period_on_time: TimePeriod,
    ) -> xr.Dataset:
        """Mask times outside of verification period with inclusive endpoints."""
        # Mask values outside of verification period
        filtered = ds.where(
            (ds[StandardDim.time] >= verification_period_on_time.start_datetime64)
            & (ds[StandardDim.time] <= verification_period_on_time.end_datetime64),
        )
        # Drop NaN values along frt and fp dims, if all values are NaN
        return filtered.dropna(dim=StandardDim.forecast_reference_time, how="all").dropna(
            dim=StandardDim.forecast_period,
            how="all",
        )

    def get_data(self) -> Self:
        """Get cached data, or fetch and cache."""
        config_json = self.config.model_dump_json().encode("utf-8")
        config_hash = hashlib.sha256(config_json).hexdigest()

        cache_dir = self.config.general.cache_dir

        # Create cache if not exists
        if not cache_dir.exists():
            cache_dir.mkdir(parents=True)

        # If it exists, check it's an accessible dir
        elif not cache_dir.is_dir() and access(cache_dir, R_OK):
            msg = "Cache directory is not an accessible directory."
            raise NotADirectoryError(msg)

        # Define file path for caching
        cached_dataset_path = cache_dir / f"{self.__class__.__name__}_{config_hash}.nc"

        if cached_dataset_path.exists():
            self.dataset = xr.open_dataset(cached_dataset_path)
            return self

        # Go fetch and cache
        self.fetch_data()
        dataset_original = self.dataset

        # Check that the datatype is defined, and consistent with the config
        if "data_type" not in dataset_original.attrs:  # type:ignore[misc]
            msg = "The fetched dataset does not have a 'data_type' attribute."
            raise ValueError(msg)
        if dataset_original.attrs["data_type"] != self.config.data_type:  # type:ignore[misc]
            msg = (
                f"The data type of the fetched dataset "
                f"({dataset_original.attrs['data_type']}) does not match the configured data "  # type:ignore[misc]
                f"type ({self.config.data_type})."
            )
            raise ValueError(msg)

        # Make sure the source attribute is set to the configured source
        dataset_original.attrs["source"] = self.config.source  # type:ignore[misc]

        # Apply re-naming based on configured id mapping, if not None
        if self.config.id_mapping is not None:
            dataset_original = self.config.id_mapping.rename_dataset(dataset_original)

        # Additional layer to filter time, frt and fp properly according to config.
        if dataset_original.attrs["data_type"] in FORECAST_DATA_TYPES:  # type:ignore[misc]
            # Select only relevant forecast periods for simulations
            dataset_original = dataset_original.sel(
                forecast_period=self.config.forecast_periods.timedelta64,
            )
            # Mask and drop time values outside of the configured vp
            dataset_original = self._drop_times_outside_vp(
                ds=dataset_original,
                verification_period_on_time=self.config.verification_period_on_time,
            )
        if dataset_original.attrs["data_type"] == DataType.observed_historical:  # type:ignore[misc]
            # Mask and drop time values outside of the configured vp
            # Historical data type
            dataset_original = dataset_original.sel(
                {
                    StandardDim.time: slice(  # type:ignore[misc]
                        self.config.verification_period_on_time.start,
                        self.config.verification_period_on_time.end,
                    ),
                },
            )

        # Cache
        dataset_original.to_netcdf(cached_dataset_path)

        # Re-open to read from cache and prevent links to original files from which the dataset
        #   was loaded
        dataset_reloaded = xr.open_dataset(cached_dataset_path)

        # Explicitly close original backing files
        if hasattr(dataset_original, "close"):
            dataset_original.close()

        # Re-assign from cache
        self.dataset = dataset_reloaded
        return self
