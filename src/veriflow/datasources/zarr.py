"""Read Zarr stores from local disk or S3."""

from typing import ClassVar, Self

import xarray as xr

from veriflow.configuration.default.datasources import ZarrConfig
from veriflow.constants import (
    DataType,
)
from veriflow.datasources.base import BaseDatasource

__all__ = [
    "Zarr",
    "ZarrConfig",
]


class Zarr(BaseDatasource):
    """A datasource for reading Zarr stores compatible with the internal datamodel.

    Wraps :func:`xarray.open_zarr` and supports both local filesystem paths and remote
    URLs (currently ``s3://`` is exercised). For S3 stores, credentials are taken from a
    :class:`~veriflow.configuration.utils.S3AuthConfig` instance loaded from environment
    variables prefixed with ``S3_``. Additional ``storage_options`` configured on the
    :class:`ZarrConfig` are merged on top and forwarded to ``xr.open_zarr``.

    .. note::
        The dataset must carry a ``data_type`` attribute that matches one of the supported
        data types; if the attribute is missing it will be set from the configuration.
    """

    kind = "zarr"
    config_class = ZarrConfig
    supported_data_types: ClassVar[set[DataType]] = {
        DataType.observed_historical,
        DataType.simulated_forecast_ensemble,
        DataType.simulated_forecast_single,
        DataType.simulated_forecast_probabilistic,
        DataType.threshold,
    }

    def __init__(self, config: ZarrConfig) -> None:
        self.config: ZarrConfig = config

    def _build_storage_options(self) -> dict[str, object] | None:
        """Build storage_options for xr.open_zarr based on path and config.

        Returns ``None`` for non-remote (local) paths so that xarray opens the store
        directly from the local filesystem.
        """
        if not self._is_remote_path(self.config.path):
            return None

        options: dict[str, object] = {}
        if self.config.auth_config is not None:
            options.update(self.config.auth_config.to_storage_options())
        if self.config.storage_options is not None:
            options.update(self.config.storage_options)
        return options

    @staticmethod
    def _is_remote_path(path: str) -> bool:
        """Return True if ``path`` looks like a remote/fsspec URL (e.g. ``s3://``)."""
        return "://" in path

    def fetch_data(self) -> Self:
        """Retrieve the configured Zarr store as an xarray Dataset."""
        storage_options = self._build_storage_options()
        dataset = xr.open_zarr(  # type:ignore[misc] # xarray's stubs are loose here
            self.config.path,
            storage_options=storage_options,
            consolidated=self.config.consolidated,
        )
        dataset.attrs["data_type"] = self.config.data_type  # type: ignore[misc]
        self.dataset = dataset
        return self
