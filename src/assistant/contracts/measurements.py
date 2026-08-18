"""Numeric and normalized value contracts."""

from __future__ import annotations

from decimal import Decimal
from typing import Annotated, Any, Literal, Union

from pydantic import Field, model_validator

from .base import ContractModel, DecimalString, NonEmptyString
from .enums import Comparator, NormalizationStatus, ValueState


class Measurement(ContractModel):
    raw_text: NonEmptyString
    comparator: Comparator
    value_decimal: DecimalString | None = None
    min_decimal: DecimalString | None = None
    max_decimal: DecimalString | None = None
    unit_raw: NonEmptyString
    unit_code: NonEmptyString | None = None
    dimension: NonEmptyString | None = None
    normalized_value: DecimalString | None = None
    normalized_min_decimal: DecimalString | None = None
    normalized_max_decimal: DecimalString | None = None
    normalized_unit: NonEmptyString | None = None
    normalization_status: NormalizationStatus
    conditions: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_shape(self) -> "Measurement":
        is_range = self.comparator == Comparator.BETWEEN
        if is_range and (self.min_decimal is None or self.max_decimal is None):
            raise ValueError("BETWEEN requires min_decimal and max_decimal")
        if is_range and self.value_decimal is not None:
            raise ValueError("BETWEEN cannot carry value_decimal")
        if is_range and Decimal(self.min_decimal) > Decimal(self.max_decimal):
            raise ValueError("BETWEEN requires min_decimal <= max_decimal")
        if not is_range and self.value_decimal is None:
            raise ValueError("non-range measurement requires value_decimal")
        if not is_range and (self.min_decimal is not None or self.max_decimal is not None):
            raise ValueError("non-range measurement cannot carry range bounds")

        normalized_values = (
            self.unit_code,
            self.dimension,
            self.normalized_value,
            self.normalized_min_decimal,
            self.normalized_max_decimal,
            self.normalized_unit,
        )
        if self.normalization_status is NormalizationStatus.NORMALIZED:
            if self.normalized_unit is None or self.unit_code is None or self.dimension is None:
                raise ValueError("NORMALIZED requires unit_code, dimension, and normalized_unit")
            if is_range:
                if (
                    self.normalized_value is not None
                    or self.normalized_min_decimal is None
                    or self.normalized_max_decimal is None
                ):
                    raise ValueError("normalized BETWEEN requires normalized range bounds")
                if Decimal(self.normalized_min_decimal) > Decimal(self.normalized_max_decimal):
                    raise ValueError("normalized range requires minimum <= maximum")
            elif (
                self.normalized_value is None
                or self.normalized_min_decimal is not None
                or self.normalized_max_decimal is not None
            ):
                raise ValueError("normalized scalar requires only normalized_value")
        elif any(value is not None for value in normalized_values):
            raise ValueError(
                "only NORMALIZED measurements may carry normalized value or unit fields"
            )
        return self


class KnownValue(ContractModel):
    value_state: Literal[ValueState.KNOWN]
    known_value: Any

    @model_validator(mode="after")
    def require_known_value(self) -> "KnownValue":
        if self.known_value is None:
            raise ValueError("KNOWN requires a non-null known_value")
        return self


class UnknownValue(ContractModel):
    value_state: Literal[ValueState.UNKNOWN]
    unknown_reason: NonEmptyString


class NotApplicableValue(ContractModel):
    value_state: Literal[ValueState.NOT_APPLICABLE]
    not_applicable_reason: NonEmptyString


class UnsetValue(ContractModel):
    value_state: Literal[ValueState.UNSET]


class UnresolvedValue(ContractModel):
    value_state: Literal[ValueState.UNRESOLVED]
    raw_value: Any | None = None
    candidate_values: tuple[Any, ...] = ()
    unresolved_reason: NonEmptyString

    @model_validator(mode="after")
    def require_candidate(self) -> "UnresolvedValue":
        if self.raw_value is None and not self.candidate_values:
            raise ValueError("UNRESOLVED requires raw_value or candidate_values")
        return self


ValuePayload = Annotated[
    Union[KnownValue, UnknownValue, NotApplicableValue, UnsetValue, UnresolvedValue],
    Field(discriminator="value_state"),
]
