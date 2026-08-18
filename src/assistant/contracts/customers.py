"""Immutable customer messages and verified normalization offset maps."""

from __future__ import annotations

from pydantic import Field, model_validator

from .base import (
    ContractModel,
    LanguageTag,
    NonEmptyString,
    NonNegativeInt,
    Sha256Digest,
    UTCDateTime,
    UUIDString,
    sha256_text,
)
from .enums import (
    CustomerMappingKind,
    CustomerSourceChannel,
    DataClassification,
    NormalizationMapStatus,
    RedactionStatus,
    SenderRole,
)


class CustomerMessage(ContractModel):
    message_id: UUIDString
    case_id: UUIDString
    sequence_no: NonNegativeInt
    source_channel: CustomerSourceChannel
    sender_role: SenderRole
    language: LanguageTag
    text: NonEmptyString
    text_sha256: Sha256Digest
    received_at: UTCDateTime
    data_classification: DataClassification
    redaction_status: RedactionStatus
    supersedes_message_id: UUIDString | None = None
    created_at: UTCDateTime

    @model_validator(mode="after")
    def validate_message_hash(self) -> "CustomerMessage":
        if sha256_text(self.text) != self.text_sha256:
            raise ValueError("text_sha256 does not match exact customer message text")
        if self.supersedes_message_id == self.message_id:
            raise ValueError("a message cannot supersede itself")
        return self


class CustomerNormalizationMappingSegment(ContractModel):
    normalized_start: NonNegativeInt
    normalized_end: NonNegativeInt
    original_start: NonNegativeInt
    original_end: NonNegativeInt
    mapping_kind: CustomerMappingKind

    @model_validator(mode="after")
    def validate_segment(self) -> "CustomerNormalizationMappingSegment":
        if self.normalized_start > self.normalized_end:
            raise ValueError("normalized range must be half-open")
        if self.original_start > self.original_end:
            raise ValueError("original range must be half-open")
        normalized_length = self.normalized_end - self.normalized_start
        original_length = self.original_end - self.original_start
        if self.mapping_kind is CustomerMappingKind.ONE_TO_ONE:
            if not normalized_length or normalized_length != original_length:
                raise ValueError("ONE_TO_ONE requires equal non-empty ranges")
        elif self.mapping_kind is CustomerMappingKind.EXPANSION:
            if not original_length or normalized_length <= original_length:
                raise ValueError("EXPANSION requires a longer normalized range")
        elif self.mapping_kind is CustomerMappingKind.CONTRACTION:
            if not normalized_length or original_length <= normalized_length:
                raise ValueError("CONTRACTION requires a shorter normalized range")
        elif normalized_length or not original_length:
            raise ValueError("DELETION requires an empty normalized and non-empty original range")
        return self


class CustomerNormalizationMap(ContractModel):
    message_id: UUIDString
    original_message_sha256: Sha256Digest
    normalized_text_sha256: Sha256Digest
    normalizer_version: NonEmptyString
    original_code_point_length: NonNegativeInt
    normalized_code_point_length: NonNegativeInt
    mapping_segments: tuple[CustomerNormalizationMappingSegment, ...]
    verification_hash: Sha256Digest
    status: NormalizationMapStatus
    failure_reason_code: NonEmptyString | None = None

    @model_validator(mode="after")
    def validate_map(self) -> "CustomerNormalizationMap":
        if self.status is NormalizationMapStatus.FAILED:
            if self.failure_reason_code is None:
                raise ValueError("FAILED normalization map requires failure_reason_code")
            return self
        if self.failure_reason_code is not None:
            raise ValueError("VERIFIED normalization map must not carry a failure reason")
        if not self.mapping_segments and (
            self.original_code_point_length or self.normalized_code_point_length
        ):
            raise ValueError("non-empty text requires mapping segments")

        normalized_cursor = 0
        original_cursor = 0
        for segment in self.mapping_segments:
            if segment.normalized_start != normalized_cursor:
                raise ValueError("normalized mapping segments must be ordered and gap-free")
            if segment.original_start != original_cursor:
                raise ValueError("original mapping segments must be ordered and gap-free")
            normalized_cursor = segment.normalized_end
            original_cursor = segment.original_end
        if normalized_cursor != self.normalized_code_point_length:
            raise ValueError("mapping segments do not cover normalized text")
        if original_cursor != self.original_code_point_length:
            raise ValueError("mapping segments do not cover original text")
        return self

    def original_span_for_normalized(self, start: int, end: int) -> tuple[int, int]:
        """Map a normalized span only when its endpoints are explicitly verifiable."""

        if self.status is not NormalizationMapStatus.VERIFIED:
            raise ValueError("CUSTOMER_OFFSET_MAPPING_FAILED")
        if not 0 <= start < end <= self.normalized_code_point_length:
            raise ValueError("CUSTOMER_OFFSET_MAPPING_FAILED")
        selected = [
            segment
            for segment in self.mapping_segments
            if segment.normalized_start < end and segment.normalized_end > start
        ]
        if not selected:
            raise ValueError("CUSTOMER_OFFSET_MAPPING_FAILED")
        if selected[0].normalized_start != start or selected[-1].normalized_end != end:
            raise ValueError("CUSTOMER_OFFSET_MAPPING_FAILED")
        if any(segment.mapping_kind is CustomerMappingKind.DELETION for segment in selected):
            raise ValueError("CUSTOMER_OFFSET_MAPPING_FAILED")
        return selected[0].original_start, selected[-1].original_end
