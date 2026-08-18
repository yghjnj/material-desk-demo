"""Case aggregate and append-only audit references."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import Field, model_validator

from .base import NonEmptyString, Sha256, UtcDatetime, VersionedContract
from .enums import CaseStatus


class AuditEvent(VersionedContract):
    audit_event_id: UUID
    case_id: UUID
    event_type: NonEmptyString
    actor_type: Literal["HUMAN", "SYSTEM"]
    actor_id: UUID | None = None
    entity_id: UUID
    entity_revision: int = Field(ge=1)
    occurred_at: UtcDatetime
    event_payload_hash: Sha256


class CaseRecord(VersionedContract):
    case_id: UUID
    case_revision: int = Field(ge=1)
    status: CaseStatus
    title: NonEmptyString
    source_channel: Literal["MANUAL", "API", "IMPORT"]
    customer_ref: NonEmptyString
    message_ids: tuple[UUID, ...] = Field(min_length=1)
    requirement_ids: tuple[UUID, ...] = ()
    retrieval_run_ids: tuple[UUID, ...] = ()
    answer_ids: tuple[UUID, ...] = ()
    draft_ids: tuple[UUID, ...] = ()
    review_decision_ids: tuple[UUID, ...] = ()
    audit_events: tuple[AuditEvent, ...] = ()
    retention_policy_id: NonEmptyString
    created_at: UtcDatetime
    updated_at: UtcDatetime

    @model_validator(mode="after")
    def audit_scope(self) -> "CaseRecord":
        if self.updated_at < self.created_at:
            raise ValueError("updated_at cannot precede created_at")
        if any(item.case_id != self.case_id for item in self.audit_events):
            raise ValueError("audit event belongs to another case")
        return self
