"""A collection of schemas for input data.

The pipeline ingests a number of predefined input data types as ``xr.Dataset`` instances. To
validate that the input data has the correct structure, we use Pydantic models in this module.

Using ``xr.Dataset.to_dict(data=False)`` returns a dictionary that can be used as input to a
Pydantic model. Each accepted data type has its own schema, built up of smaller sub-models.

For now, we validate
- Coordinates: name, dtype, dimensions
- Data variables: required dims, required attributes (notably ``units``), CF-compliant naming
- Dataset attributes (notably ``data_type``)
"""


# mypy: ignore-errors
# ruff: noqa: D101

from typing import Annotated, Literal

import xarray as xr
from pydantic import AfterValidator, BaseModel, Field, RootModel

from veriflow.constants import DataType, StandardDim

AllowedDTypeInt = Literal["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"]
AllowedDTypeFloat = Literal["float16", "float32", "float64"]
AllowedDTypeDateTime = Literal["datetime64[ns]"]
AllowedDTypeTimeDelta = Literal["timedelta64[ns]"]


def check_dims(
    required: set[str],
    optional: set[str] | None = None,
) -> callable:
    """Check a if a tuple contains the expected dimensions. Used in a Pydantic AfterValidator."""
    allowed = required | optional if optional else required

    def validator(value: tuple[str, ...]) -> tuple[str, ...]:
        value_set = set(value)

        # Check for missing required
        missing = required - value_set
        if len(missing) > 0:
            msg = f"Missing required dims: {', '.join(missing)}"
            raise ValueError(msg)

        # Check for disallowed dims
        disallowed = value_set - allowed
        if len(disallowed) > 0:
            msg = f"Invalid dims: {disallowed}. Allowed: {', '.join(allowed)}"
            raise ValueError(msg)

        return value

    return validator


# ---------------------------------------------------------------------------
# Coordinate schemas
# ---------------------------------------------------------------------------


class HistoricalTimeCoord(BaseModel):
    dims: Annotated[tuple[str, ...], AfterValidator(check_dims({StandardDim.time}))]
    dtype: AllowedDTypeDateTime


class ForecastTimeCoord(BaseModel):
    dims: Annotated[
        tuple[str, ...],
        AfterValidator(
            check_dims({StandardDim.forecast_reference_time, StandardDim.forecast_period}),
        ),
    ]
    dtype: AllowedDTypeDateTime


class ForecastReferenceTimeCoord(BaseModel):
    dims: Annotated[
        tuple[str, ...],
        AfterValidator(check_dims({StandardDim.forecast_reference_time})),
    ]
    dtype: AllowedDTypeDateTime


class StationCoord(BaseModel):
    dims: Annotated[tuple[str, ...], AfterValidator(check_dims({StandardDim.station}))]


class XYZCoord(BaseModel):
    dims: Annotated[tuple[str, ...], AfterValidator(check_dims({StandardDim.station}))]
    dtype: AllowedDTypeFloat


class ForecastPeriodCoord(BaseModel):
    dims: Annotated[tuple[str, ...], AfterValidator(check_dims({StandardDim.forecast_period}))]
    dtype: AllowedDTypeTimeDelta


class RealizationCoord(BaseModel):
    dims: Annotated[tuple[str, ...], AfterValidator(check_dims({StandardDim.realization}))]
    dtype: AllowedDTypeInt


class ThresholdCoord(BaseModel):
    dims: Annotated[tuple[str, ...], AfterValidator(check_dims({StandardDim.threshold}))]


class BaseCoords(BaseModel):
    station: StationCoord
    station_name: StationCoord | None = None  # Optional station name coordinate
    lat: XYZCoord  # Always required lat, lon
    lon: XYZCoord
    x: XYZCoord | None = None  # Optional x, y, z
    y: XYZCoord | None = None
    z: XYZCoord | None = None


class BaseHistoricalCoords(BaseCoords):
    time: HistoricalTimeCoord


class SimulatedForecastSingleCoords(BaseCoords):
    forecast_reference_time: ForecastReferenceTimeCoord
    forecast_period: ForecastPeriodCoord
    time: ForecastTimeCoord


class SimulatedForecastEnsembleCoords(BaseCoords):
    forecast_reference_time: ForecastReferenceTimeCoord
    forecast_period: ForecastPeriodCoord
    realization: RealizationCoord
    time: ForecastTimeCoord


class SimulatedForecastProbabilisticCoords(BaseCoords):
    forecast_reference_time: ForecastReferenceTimeCoord
    forecast_period: ForecastPeriodCoord
    threshold: ThresholdCoord
    time: ForecastTimeCoord


class ThresholdCoords(BaseModel):
    """The structure of a threshold dataset's coords."""

    station: StationCoord
    station_name: StationCoord | None = None  # Optional station name coordinate
    threshold: ThresholdCoord


# ---------------------------------------------------------------------------
# Data variable schemas
# ---------------------------------------------------------------------------


CFCompliantName = Annotated[
    str,
    Field(
        pattern=r"^[A-Za-z][A-Za-z0-9_]*$",
        description="It is required that variable, dimension, attribute and group names "
        "begin with a letter and be composed of letters, digits, and underscores. "
        "(https://cfconventions.org/Data/cf-conventions/cf-conventions-1.12/cf-conventions.html#_naming_conventions)",
    ),
]


class DataVarAttrs(BaseModel):
    """Required attributes on a data variable."""

    units: Annotated[
        str,
        Field(
            min_length=1,
            description="Units of the data variable. Required for downstream interpretation and "
            "CF-compliant output.",
        ),
    ]

    model_config = {"extra": "allow"}


class HistoricalDataVar(BaseModel):
    dims: Annotated[
        tuple[str, ...],
        AfterValidator(
            check_dims({StandardDim.station, StandardDim.time}),
        ),
    ]
    attrs: DataVarAttrs


class SimulatedForecastSingleDataVar(BaseModel):
    dims: Annotated[
        tuple[str, ...],
        AfterValidator(
            check_dims(
                {
                    StandardDim.station,
                    StandardDim.forecast_reference_time,
                    StandardDim.forecast_period,
                },
            ),
        ),
    ]
    attrs: DataVarAttrs


class SimulatedForecastEnsembleDataVar(BaseModel):
    dims: Annotated[
        tuple[str, ...],
        AfterValidator(
            check_dims(
                {
                    StandardDim.station,
                    StandardDim.forecast_reference_time,
                    StandardDim.forecast_period,
                    StandardDim.realization,
                },
            ),
        ),
    ]
    attrs: DataVarAttrs


class SimulatedForecastProbabilisticDataVar(BaseModel):
    dims: Annotated[
        tuple[str, ...],
        AfterValidator(
            check_dims(
                {
                    StandardDim.station,
                    StandardDim.forecast_reference_time,
                    StandardDim.forecast_period,
                    StandardDim.threshold,
                },
            ),
        ),
    ]
    attrs: DataVarAttrs


class ThresholdDataVar(BaseModel):
    dims: Annotated[
        tuple[str, ...],
        AfterValidator(
            check_dims({StandardDim.station, StandardDim.threshold}),
        ),
    ]
    # Threshold variables don't strictly need units; allow any attrs
    attrs: dict | None = None


# ---------------------------------------------------------------------------
# Data variable collections (dict of CF-compliant name -> DataVar schema)
#
# Each ``RootModel`` validates that data variable names (the dict keys) are
# CF-compliant via the ``CFCompliantName`` constraint, and that each value
# matches the corresponding per-data-type ``*DataVar`` schema.
# ---------------------------------------------------------------------------


HistoricalDataVars = RootModel[dict[CFCompliantName, HistoricalDataVar]]
SimulatedForecastSingleDataVars = RootModel[dict[CFCompliantName, SimulatedForecastSingleDataVar]]
SimulatedForecastEnsembleDataVars = RootModel[
    dict[CFCompliantName, SimulatedForecastEnsembleDataVar]
]
SimulatedForecastProbabilisticDataVars = RootModel[
    dict[CFCompliantName, SimulatedForecastProbabilisticDataVar]
]
ThresholdDataVars = RootModel[dict[CFCompliantName, ThresholdDataVar]]


# ---------------------------------------------------------------------------
# Dataset-level attribute schemas
# ---------------------------------------------------------------------------


class BaseAttrs(BaseModel):
    data_type: str

    model_config = {"extra": "allow"}


# ---------------------------------------------------------------------------
# Top-level dataset schemas
# ---------------------------------------------------------------------------


class ObservedHistorical(BaseModel):
    coords: BaseHistoricalCoords
    data_vars: HistoricalDataVars
    attrs: BaseAttrs


class SimulatedHistorical(BaseModel):
    coords: BaseHistoricalCoords
    data_vars: HistoricalDataVars
    attrs: BaseAttrs


class SimulatedForecastSingle(BaseModel):
    coords: SimulatedForecastSingleCoords
    data_vars: SimulatedForecastSingleDataVars
    attrs: BaseAttrs


class SimulatedForecastEnsemble(BaseModel):
    coords: SimulatedForecastEnsembleCoords
    data_vars: SimulatedForecastEnsembleDataVars
    attrs: BaseAttrs


class SimulatedForecastProbabilistic(BaseModel):
    coords: SimulatedForecastProbabilisticCoords
    data_vars: SimulatedForecastProbabilisticDataVars
    attrs: BaseAttrs


class Thresholds(BaseModel):
    coords: ThresholdCoords
    data_vars: ThresholdDataVars
    attrs: BaseAttrs


# All input schemas, keyed by the corresponding data type
INPUT_SCHEMAS: dict[DataType, BaseModel] = {
    DataType.observed_historical: ObservedHistorical,
    DataType.simulated_historical: SimulatedHistorical,
    DataType.simulated_forecast_single: SimulatedForecastSingle,
    DataType.simulated_forecast_ensemble: SimulatedForecastEnsemble,
    DataType.simulated_forecast_probabilistic: SimulatedForecastProbabilistic,
    DataType.threshold: Thresholds,
}


def validate_input_data(dataset: xr.Dataset) -> None:
    """Validate an input ``xr.Dataset`` against its schema.

    The data type is determined from the ``data_type`` attribute on the dataset.
    """
    if not isinstance(dataset, xr.Dataset):
        msg = f"Expected an xarray Dataset. Got: {type(dataset)}"
        raise TypeError(msg)

    if "data_type" not in dataset.attrs:
        msg = "Input dataset is missing required 'data_type' attribute."
        raise ValueError(msg)

    data_type = dataset.attrs["data_type"]
    schema_class = INPUT_SCHEMAS.get(data_type)
    if not schema_class:
        msg = f"No input schema defined for data type: {data_type}"
        raise ValueError(msg)

    data_dict = dataset.to_dict(data=False)
    schema_class.model_validate(data_dict)
