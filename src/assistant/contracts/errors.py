"""Refusal and transport-error contracts."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping
from uuid import UUID

from .base import ContractModel, NonEmptyString
from .enums import ErrorCode, RefusalCode


class FieldError(ContractModel):
    location: NonEmptyString
    message: NonEmptyString
    error_type: NonEmptyString


class APIError(ContractModel):
    code: ErrorCode
    message: NonEmptyString
    retryable: bool = False
    field_errors: tuple[FieldError, ...] = ()


class ErrorEnvelope(ContractModel):
    error: APIError
    trace_id: UUID


class RefusalReason(ContractModel):
    code: RefusalCode
    user_message: NonEmptyString
    missing_information: tuple[NonEmptyString, ...] = ()
    conflicting_technical_citation_ids: tuple[UUID, ...] = ()
    suggested_next_action: NonEmptyString


ERROR_STATUS_MAP: Mapping[ErrorCode, int] = MappingProxyType({
    ErrorCode.INVALID_SCHEMA: 400,
    ErrorCode.NOT_FOUND: 404,
    ErrorCode.REVISION_CONFLICT: 409,
    ErrorCode.IDEMPOTENCY_CONFLICT: 409,
    ErrorCode.PAYLOAD_TOO_LARGE: 413,
    ErrorCode.UNSUPPORTED_MEDIA_TYPE: 415,
    ErrorCode.SEMANTIC_VALIDATION_FAILED: 422,
    ErrorCode.DEPENDENCY_UNAVAILABLE: 503,
    ErrorCode.LIFECYCLE_NOT_YET_CREATED: 409,
    ErrorCode.LIFECYCLE_VERSION_GAP: 409,
    ErrorCode.LIFECYCLE_EVENT_MISSING: 422,
    ErrorCode.LIFECYCLE_EVENT_CONFLICT: 409,
    ErrorCode.LIFECYCLE_INTERVAL_INVALID: 422,
    ErrorCode.LIFECYCLE_CORRECTION_INVALID: 422,
    ErrorCode.RENDER_MAPPING_PARTIAL: 422,
    ErrorCode.RENDER_MAPPING_FAILED: 422,
    ErrorCode.RENDER_RECEIPT_MISMATCH: 409,
    ErrorCode.UNSUPPORTED_TABLE_LOCATION: 422,
    ErrorCode.UNSUPPORTED_COMPLEX_TABLE: 422,
    ErrorCode.UNSUPPORTED_CROSS_PAGE_TABLE: 422,
    ErrorCode.CUSTOMER_OFFSET_MAPPING_FAILED: 422,
    ErrorCode.SIDECAR_IDEMPOTENCY_CONFLICT: 409,
    ErrorCode.RESUME_CONTEXT_MISMATCH: 409,
})

# Backward-compatible name; both aliases point to the same immutable total map.
HTTP_STATUS_BY_ERROR_CODE = ERROR_STATUS_MAP
