"""Customer-text locators with strict gold/runtime role separation."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from .base import (
    LOCATOR_PROFILE_VERSION,
    ContractModel,
    NonEmptyString,
    NonNegativeInt,
    SemanticVersion,
    Sha256Digest,
    UUIDString,
    half_open_range_is_valid,
    sha256_text,
)
from .enums import LocatorRole, LocatorStatus


class CustomerTextLocator(ContractModel):
    """Common original-message coordinates; never used directly as a role."""

    locator_profile_version: SemanticVersion = LOCATOR_PROFILE_VERSION
    message_id: UUIDString
    message_sha256: Sha256Digest
    original_unicode_code_point_length: NonNegativeInt
    char_start: NonNegativeInt
    char_end: NonNegativeInt
    quote: str
    quote_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_original_coordinates(self) -> "CustomerTextLocator":
        if not half_open_range_is_valid(
            self.char_start,
            self.char_end,
            self.original_unicode_code_point_length,
        ):
            raise ValueError(
                "customer locator must satisfy 0 <= start <= end <= original length"
            )
        if self.char_end - self.char_start != len(self.quote):
            raise ValueError("customer locator range must equal quote code-point length")
        if sha256_text(self.quote) != self.quote_sha256:
            raise ValueError("quote_sha256 does not match exact quote UTF-8 bytes")
        return self


class GoldCustomerLocator(CustomerTextLocator):
    gold_customer_locator_id: UUIDString
    locator_role: Literal[LocatorRole.GOLD] = LocatorRole.GOLD
    annotation_version: NonEmptyString
    locator_status: LocatorStatus


class RuntimeCustomerLocator(CustomerTextLocator):
    runtime_customer_locator_id: UUIDString
    locator_role: Literal[LocatorRole.RUNTIME] = LocatorRole.RUNTIME
    extraction_run_id: UUIDString
    normalizer_version: NonEmptyString

    @model_validator(mode="after")
    def require_non_empty_attribution(self) -> "RuntimeCustomerLocator":
        if self.char_start >= self.char_end or not self.quote:
            raise ValueError("runtime customer locator must identify non-empty source text")
        return self


AnyCustomerTextLocator = Annotated[
    GoldCustomerLocator | RuntimeCustomerLocator,
    Field(discriminator="locator_role"),
]

