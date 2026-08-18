"""Source/runtime metadata and deterministic document lifecycle reconstruction."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Iterable

from pydantic import Field, model_validator

from .base import (
    ContractModel,
    LanguageTag,
    NonEmptyString,
    NonNegativeInt,
    PositiveInt,
    Sha256Digest,
    UTCDateTime,
    UUIDString,
)
from .enums import (
    ApprovalStatus,
    AuthorityClass,
    DocumentFormat,
    HistoricalUsePolicy,
    LifecycleEventType,
    LifecycleResultStatus,
    LifecycleStatus,
    ProcessingStatus,
    RevisionStatus,
)


class ApplicabilityScope(ContractModel):
    languages: tuple[LanguageTag, ...]
    regions: tuple[NonEmptyString, ...]
    product_refs: tuple[NonEmptyString, ...]
    conditions: tuple[NonEmptyString, ...] = ()


class Authority(ContractModel):
    authority_class: AuthorityClass
    issuer: NonEmptyString
    approval_status: ApprovalStatus
    scope: NonEmptyString


class SourceDocumentMetadata(ContractModel):
    """B-owned author/source assertions; runtime ingestion never overwrites them."""

    source_metadata_revision_id: UUIDString
    source_metadata_revision: PositiveInt
    revision_status: RevisionStatus
    document_id: UUIDString
    document_version_id: UUIDString
    title: NonEmptyString
    document_format: DocumentFormat
    language: LanguageTag
    version_label: NonEmptyString
    source_ref: NonEmptyString
    source_sha256: Sha256Digest
    issuer: NonEmptyString
    authors: tuple[NonEmptyString, ...] = ()
    license_id: NonEmptyString
    provenance: tuple[NonEmptyString, ...] = Field(min_length=1)
    data_classification: NonEmptyString
    supersedes_document_version_id: UUIDString | None = None
    effective_from: UTCDateTime | None = None
    effective_to: UTCDateTime | None = None
    withdrawn_at: UTCDateTime | None = None
    withdrawal_reason: NonEmptyString | None = None
    historical_use_policy: HistoricalUsePolicy = HistoricalUsePolicy.PROHIBITED
    authority: Authority
    precedence: int
    precedence_policy_version: NonEmptyString
    applicability_scope: ApplicabilityScope
    created_at: UTCDateTime

    @model_validator(mode="after")
    def validate_source_version(self) -> "SourceDocumentMetadata":
        if self.supersedes_document_version_id == self.document_version_id:
            raise ValueError("a document version cannot supersede itself")
        if (
            self.effective_from is not None
            and self.effective_to is not None
            and self.effective_from >= self.effective_to
        ):
            raise ValueError("effective interval must be [from, to)")
        if (self.withdrawn_at is None) != (self.withdrawal_reason is None):
            raise ValueError("withdrawn_at and withdrawal_reason must be supplied together")
        return self


class RuntimeIngestionMetadata(ContractModel):
    """C-owned parser/canonicalizer output for one exact source revision."""

    runtime_ingestion_revision_id: UUIDString
    runtime_ingestion_revision: PositiveInt
    revision_status: RevisionStatus
    source_metadata_revision_id: UUIDString
    source_sha256: Sha256Digest
    document_id: UUIDString
    document_version_id: UUIDString
    processing_status: ProcessingStatus
    canonical_text_sha256: Sha256Digest | None = None
    parser_id: NonEmptyString
    parser_version: NonEmptyString
    canonicalizer_version: NonEmptyString
    render_receipt_id: UUIDString | None = None
    processing_error_code: NonEmptyString | None = None
    created_at: UTCDateTime
    invalidated_at: UTCDateTime | None = None
    invalidation_reason: NonEmptyString | None = None

    @model_validator(mode="after")
    def validate_processing_result(self) -> "RuntimeIngestionMetadata":
        canonical_required = {
            ProcessingStatus.PARSED,
            ProcessingStatus.INDEXING,
            ProcessingStatus.INDEXED,
        }
        if self.processing_status in canonical_required and self.canonical_text_sha256 is None:
            raise ValueError("parsed/indexed metadata requires canonical_text_sha256")
        if self.processing_status is ProcessingStatus.FAILED:
            if self.processing_error_code is None:
                raise ValueError("FAILED ingestion requires processing_error_code")
        elif self.processing_error_code is not None:
            raise ValueError("processing_error_code is only valid for FAILED ingestion")
        if (self.invalidated_at is None) != (self.invalidation_reason is None):
            raise ValueError("invalidation timestamp and reason must be supplied together")
        return self


class IndexSnapshotMetadata(ContractModel):
    """C-owned immutable index snapshot, never source-author metadata."""

    index_snapshot_id: UUIDString
    index_snapshot_revision: PositiveInt
    revision_status: RevisionStatus
    runtime_ingestion_revision_id: UUIDString
    source_sha256: Sha256Digest
    canonical_text_sha256: Sha256Digest
    parser_version: NonEmptyString
    canonicalizer_version: NonEmptyString
    chunking_version: NonEmptyString
    embedding_id: NonEmptyString
    embedding_version: NonEmptyString
    index_id: NonEmptyString
    index_version: NonEmptyString
    index_configuration_hash: Sha256Digest
    created_at: UTCDateTime
    invalidated_at: UTCDateTime | None = None
    invalidation_reason: NonEmptyString | None = None

    @model_validator(mode="after")
    def validate_invalidation(self) -> "IndexSnapshotMetadata":
        if (self.invalidated_at is None) != (self.invalidation_reason is None):
            raise ValueError("invalidation timestamp and reason must be supplied together")
        return self


class DocumentLifecycleEvent(ContractModel):
    lifecycle_event_id: UUIDString
    document_version_id: UUIDString
    event_type: LifecycleEventType
    from_status: LifecycleStatus | None = None
    to_status: LifecycleStatus
    effective_at: UTCDateTime
    effective_to: UTCDateTime | None = None
    recorded_at: UTCDateTime
    event_sequence: PositiveInt
    reason: NonEmptyString
    corrects_event_id: UUIDString | None = None
    correction_sequence: PositiveInt | None = None
    recorded_by: UUIDString
    correction_reason: NonEmptyString | None = None
    metadata_revision_id: UUIDString

    @model_validator(mode="after")
    def validate_event(self) -> "DocumentLifecycleEvent":
        if self.effective_to is not None and self.effective_at >= self.effective_to:
            raise ValueError("lifecycle interval must be [effective_at, effective_to)")
        if self.event_type is LifecycleEventType.CORRECTED:
            if (
                self.corrects_event_id is None
                or self.correction_sequence is None
                or self.correction_reason is None
            ):
                raise ValueError("CORRECTED event requires target, sequence, and reason")
            if self.corrects_event_id == self.lifecycle_event_id:
                raise ValueError("a lifecycle event cannot correct itself")
        elif any(
            value is not None
            for value in (
                self.corrects_event_id,
                self.correction_sequence,
                self.correction_reason,
            )
        ):
            raise ValueError("correction fields are only valid for CORRECTED events")

        if self.event_type is LifecycleEventType.CREATED and self.from_status is not None:
            raise ValueError("CREATED must not have from_status")
        if self.event_type not in (LifecycleEventType.CREATED, LifecycleEventType.CORRECTED):
            if self.from_status is None:
                raise ValueError("state transition event requires from_status")
        expected_targets = {
            LifecycleEventType.ACTIVATED: LifecycleStatus.ACTIVE,
            LifecycleEventType.REACTIVATED: LifecycleStatus.ACTIVE,
            LifecycleEventType.SUPERSEDED: LifecycleStatus.SUPERSEDED,
            LifecycleEventType.WITHDRAWN: LifecycleStatus.WITHDRAWN,
        }
        expected = expected_targets.get(self.event_type)
        if expected is not None and self.to_status is not expected:
            raise ValueError(f"{self.event_type.value} must transition to {expected.value}")
        valid_from_statuses = {
            LifecycleEventType.ACTIVATED: {LifecycleStatus.DRAFT},
            LifecycleEventType.REACTIVATED: {
                LifecycleStatus.SUPERSEDED,
                LifecycleStatus.WITHDRAWN,
            },
            LifecycleEventType.SUPERSEDED: {LifecycleStatus.ACTIVE},
            LifecycleEventType.WITHDRAWN: {
                LifecycleStatus.DRAFT,
                LifecycleStatus.ACTIVE,
                LifecycleStatus.SUPERSEDED,
            },
        }
        if self.event_type is LifecycleEventType.CREATED and self.to_status is not LifecycleStatus.DRAFT:
            raise ValueError("CREATED must establish DRAFT status")
        allowed_from = valid_from_statuses.get(self.event_type)
        if allowed_from is not None and self.from_status not in allowed_from:
            raise ValueError(f"{self.event_type.value} has an invalid from_status")
        return self


class LifecycleEffectiveInterval(ContractModel):
    effective_from: UTCDateTime
    effective_to: UTCDateTime | None = None

    @model_validator(mode="after")
    def validate_interval(self) -> "LifecycleEffectiveInterval":
        if self.effective_to is not None and self.effective_from >= self.effective_to:
            raise ValueError("effective interval must be [from, to)")
        return self


class LifecycleAtResult(ContractModel):
    document_version_id: UUIDString
    as_of: UTCDateTime
    metadata_snapshot_hash: Sha256Digest
    result_status: LifecycleResultStatus
    lifecycle_status: LifecycleStatus | None = None
    effective_interval: LifecycleEffectiveInterval | None = None
    applied_event_ids: tuple[UUIDString, ...] = ()
    error_code: NonEmptyString | None = None

    @model_validator(mode="after")
    def validate_result(self) -> "LifecycleAtResult":
        if self.result_status is LifecycleResultStatus.RESOLVED:
            if self.lifecycle_status is None or self.effective_interval is None:
                raise ValueError("RESOLVED lifecycle result requires status and interval")
            if self.error_code is not None:
                raise ValueError("RESOLVED lifecycle result must not contain an error")
        elif self.lifecycle_status is not None or self.effective_interval is not None:
            raise ValueError("unresolved lifecycle result must not guess lifecycle state")
        elif self.error_code is None:
            raise ValueError("non-RESOLVED lifecycle result requires error_code")
        return self


LIFECYCLE_ERROR_CODES = frozenset(
    {
        "LIFECYCLE_NOT_YET_CREATED",
        "LIFECYCLE_VERSION_GAP",
        "LIFECYCLE_EVENT_MISSING",
        "LIFECYCLE_EVENT_CONFLICT",
        "LIFECYCLE_INTERVAL_INVALID",
        "LIFECYCLE_CORRECTION_INVALID",
    }
)


def _lifecycle_error(
    *,
    document_version_id: str,
    as_of: datetime,
    metadata_snapshot_hash: str,
    result_status: LifecycleResultStatus,
    error_code: str,
    applied_event_ids: tuple[str, ...] = (),
) -> LifecycleAtResult:
    return LifecycleAtResult(
        document_version_id=document_version_id,
        as_of=as_of,
        metadata_snapshot_hash=metadata_snapshot_hash,
        result_status=result_status,
        applied_event_ids=applied_event_ids,
        error_code=error_code,
    )


def reconstruct_lifecycle_at(
    events: Iterable[DocumentLifecycleEvent],
    *,
    document_version_id: UUIDString,
    as_of: UTCDateTime,
    metadata_snapshot_recorded_at: UTCDateTime,
    metadata_snapshot_hash: Sha256Digest,
) -> LifecycleAtResult:
    """Rebuild lifecycle state from the locked, auditable event snapshot.

    No branch falls back to a current-state cache.  Backdated corrections are
    ordered by business effective time after their correction chain is checked
    using registration time and explicit correction sequence.
    """

    visible = [
        event
        for event in events
        if event.document_version_id == document_version_id
        and event.recorded_at <= metadata_snapshot_recorded_at
    ]
    if not visible:
        return _lifecycle_error(
            document_version_id=document_version_id,
            as_of=as_of,
            metadata_snapshot_hash=metadata_snapshot_hash,
            result_status=LifecycleResultStatus.INVALID,
            error_code="LIFECYCLE_EVENT_MISSING",
        )

    by_id: dict[str, DocumentLifecycleEvent] = {}
    event_sequences: set[int] = set()
    for event in visible:
        if event.lifecycle_event_id in by_id or event.event_sequence in event_sequences:
            return _lifecycle_error(
                document_version_id=document_version_id,
                as_of=as_of,
                metadata_snapshot_hash=metadata_snapshot_hash,
                result_status=LifecycleResultStatus.CONFLICT,
                error_code="LIFECYCLE_EVENT_CONFLICT",
            )
        by_id[event.lifecycle_event_id] = event
        event_sequences.add(event.event_sequence)

    successor: dict[str, DocumentLifecycleEvent] = {}
    correction_ids: set[str] = set()
    for event in visible:
        if event.event_type is not LifecycleEventType.CORRECTED:
            continue
        target = by_id.get(event.corrects_event_id or "")
        if target is None or target.recorded_at >= event.recorded_at:
            return _lifecycle_error(
                document_version_id=document_version_id,
                as_of=as_of,
                metadata_snapshot_hash=metadata_snapshot_hash,
                result_status=LifecycleResultStatus.INVALID,
                error_code="LIFECYCLE_CORRECTION_INVALID",
            )
        if event.corrects_event_id in successor:
            return _lifecycle_error(
                document_version_id=document_version_id,
                as_of=as_of,
                metadata_snapshot_hash=metadata_snapshot_hash,
                result_status=LifecycleResultStatus.CONFLICT,
                error_code="LIFECYCLE_EVENT_CONFLICT",
            )
        expected_sequence = (target.correction_sequence or 0) + 1
        if (
            event.correction_sequence != expected_sequence
            or event.event_sequence <= target.event_sequence
        ):
            return _lifecycle_error(
                document_version_id=document_version_id,
                as_of=as_of,
                metadata_snapshot_hash=metadata_snapshot_hash,
                result_status=LifecycleResultStatus.INVALID,
                error_code="LIFECYCLE_CORRECTION_INVALID",
            )
        successor[event.corrects_event_id] = event
        correction_ids.add(event.lifecycle_event_id)

    effective_events: list[tuple[LifecycleEventType, DocumentLifecycleEvent, tuple[str, ...]]] = []
    roots = [event for event in visible if event.event_type is not LifecycleEventType.CORRECTED]
    visited_corrections: set[str] = set()
    for root in roots:
        chain = [root.lifecycle_event_id]
        terminal = root
        seen = {root.lifecycle_event_id}
        while terminal.lifecycle_event_id in successor:
            previous = terminal
            terminal = successor[terminal.lifecycle_event_id]
            if terminal.lifecycle_event_id in seen:
                return _lifecycle_error(
                    document_version_id=document_version_id,
                    as_of=as_of,
                    metadata_snapshot_hash=metadata_snapshot_hash,
                    result_status=LifecycleResultStatus.INVALID,
                    error_code="LIFECYCLE_CORRECTION_INVALID",
                )
            seen.add(terminal.lifecycle_event_id)
            visited_corrections.add(terminal.lifecycle_event_id)
            chain.append(terminal.lifecycle_event_id)
            if (
                terminal.from_status is not root.from_status
                or terminal.to_status is not root.to_status
                or terminal.event_sequence <= previous.event_sequence
            ):
                return _lifecycle_error(
                    document_version_id=document_version_id,
                    as_of=as_of,
                    metadata_snapshot_hash=metadata_snapshot_hash,
                    result_status=LifecycleResultStatus.INVALID,
                    error_code="LIFECYCLE_CORRECTION_INVALID",
                )
        effective_events.append((root.event_type, terminal, tuple(chain)))
    if visited_corrections != correction_ids:
        return _lifecycle_error(
            document_version_id=document_version_id,
            as_of=as_of,
            metadata_snapshot_hash=metadata_snapshot_hash,
            result_status=LifecycleResultStatus.INVALID,
            error_code="LIFECYCLE_CORRECTION_INVALID",
        )

    effective_events.sort(key=lambda item: (item[1].effective_at, item[1].event_sequence))
    if not effective_events or effective_events[0][0] is not LifecycleEventType.CREATED:
        return _lifecycle_error(
            document_version_id=document_version_id,
            as_of=as_of,
            metadata_snapshot_hash=metadata_snapshot_hash,
            result_status=LifecycleResultStatus.INVALID,
            error_code="LIFECYCLE_EVENT_MISSING",
        )
    if sum(kind is LifecycleEventType.CREATED for kind, _, _ in effective_events) != 1:
        return _lifecycle_error(
            document_version_id=document_version_id,
            as_of=as_of,
            metadata_snapshot_hash=metadata_snapshot_hash,
            result_status=LifecycleResultStatus.CONFLICT,
            error_code="LIFECYCLE_EVENT_CONFLICT",
        )

    grouped: dict[datetime, list[DocumentLifecycleEvent]] = defaultdict(list)
    for _, event, _ in effective_events:
        grouped[event.effective_at].append(event)
    if any(len(group) > 1 for group in grouped.values()):
        return _lifecycle_error(
            document_version_id=document_version_id,
            as_of=as_of,
            metadata_snapshot_hash=metadata_snapshot_hash,
            result_status=LifecycleResultStatus.CONFLICT,
            error_code="LIFECYCLE_EVENT_CONFLICT",
        )

    created_at = effective_events[0][1].effective_at
    if as_of < created_at:
        return _lifecycle_error(
            document_version_id=document_version_id,
            as_of=as_of,
            metadata_snapshot_hash=metadata_snapshot_hash,
            result_status=LifecycleResultStatus.NOT_CREATED,
            error_code="LIFECYCLE_NOT_YET_CREATED",
        )

    applied: list[str] = []
    current_status: LifecycleStatus | None = None
    for index, (semantic_kind, event, chain) in enumerate(effective_events):
        if semantic_kind is LifecycleEventType.CREATED:
            if current_status is not None or event.from_status is not None:
                return _lifecycle_error(
                    document_version_id=document_version_id,
                    as_of=as_of,
                    metadata_snapshot_hash=metadata_snapshot_hash,
                    result_status=LifecycleResultStatus.CONFLICT,
                    error_code="LIFECYCLE_EVENT_CONFLICT",
                )
        elif current_status is None or event.from_status is not current_status:
            return _lifecycle_error(
                document_version_id=document_version_id,
                as_of=as_of,
                metadata_snapshot_hash=metadata_snapshot_hash,
                result_status=LifecycleResultStatus.INVALID,
                error_code="LIFECYCLE_EVENT_MISSING",
                applied_event_ids=tuple(applied),
            )

        next_start = (
            effective_events[index + 1][1].effective_at
            if index + 1 < len(effective_events)
            else None
        )
        interval_end = event.effective_to if event.effective_to is not None else next_start
        if event.effective_to is not None and next_start is not None:
            if event.effective_to > next_start:
                return _lifecycle_error(
                    document_version_id=document_version_id,
                    as_of=as_of,
                    metadata_snapshot_hash=metadata_snapshot_hash,
                    result_status=LifecycleResultStatus.CONFLICT,
                    error_code="LIFECYCLE_EVENT_CONFLICT",
                    applied_event_ids=tuple(applied),
                )

        current_status = event.to_status
        applied.extend(chain)
        if as_of >= event.effective_at and (interval_end is None or as_of < interval_end):
            return LifecycleAtResult(
                document_version_id=document_version_id,
                as_of=as_of,
                metadata_snapshot_hash=metadata_snapshot_hash,
                result_status=LifecycleResultStatus.RESOLVED,
                lifecycle_status=current_status,
                effective_interval=LifecycleEffectiveInterval(
                    effective_from=event.effective_at,
                    effective_to=interval_end,
                ),
                applied_event_ids=tuple(applied),
            )
        if interval_end is not None and as_of >= interval_end:
            if next_start is None or as_of < next_start:
                return _lifecycle_error(
                    document_version_id=document_version_id,
                    as_of=as_of,
                    metadata_snapshot_hash=metadata_snapshot_hash,
                    result_status=LifecycleResultStatus.UNRESOLVED,
                    error_code="LIFECYCLE_VERSION_GAP",
                    applied_event_ids=tuple(applied),
                )

    return _lifecycle_error(
        document_version_id=document_version_id,
        as_of=as_of,
        metadata_snapshot_hash=metadata_snapshot_hash,
        result_status=LifecycleResultStatus.UNRESOLVED,
        error_code="LIFECYCLE_VERSION_GAP",
        applied_event_ids=tuple(applied),
    )


lifecycle_at = reconstruct_lifecycle_at


class DocumentVersionEligibility(ContractModel):
    document_version_id: UUIDString
    as_of: UTCDateTime
    lifecycle_result: LifecycleAtResult
    processing_status: ProcessingStatus
    within_declared_effective_interval: bool
    active_version_conflict: bool = False
    evidence_eligible: bool
    exclusion_reason_codes: tuple[NonEmptyString, ...] = ()

    @model_validator(mode="after")
    def enforce_active_indexed_only(self) -> "DocumentVersionEligibility":
        if self.lifecycle_result.document_version_id != self.document_version_id:
            raise ValueError("eligibility lifecycle result belongs to another document version")
        if self.lifecycle_result.as_of != self.as_of:
            raise ValueError("eligibility as_of must exactly match lifecycle reconstruction")
        expected = (
            self.lifecycle_result.result_status is LifecycleResultStatus.RESOLVED
            and self.lifecycle_result.lifecycle_status is LifecycleStatus.ACTIVE
            and self.processing_status is ProcessingStatus.INDEXED
            and self.within_declared_effective_interval
            and not self.active_version_conflict
        )
        if self.evidence_eligible != expected:
            raise ValueError("evidence_eligible must exactly implement ACTIVE+INDEXED policy")
        if not expected and not self.exclusion_reason_codes:
            raise ValueError("ineligible version requires an exclusion reason")
        return self


class DocumentMetadata(ContractModel):
    """Read aggregate; preserves B source and C runtime records as distinct objects."""

    source_metadata: SourceDocumentMetadata
    runtime_ingestion_metadata: RuntimeIngestionMetadata | None = None
    index_snapshot_metadata: IndexSnapshotMetadata | None = None
    lifecycle_at_result: LifecycleAtResult | None = None

    @model_validator(mode="after")
    def validate_aggregate_identity(self) -> "DocumentMetadata":
        source = self.source_metadata
        runtime = self.runtime_ingestion_metadata
        index = self.index_snapshot_metadata
        lifecycle = self.lifecycle_at_result
        if runtime is not None:
            if (
                runtime.document_id != source.document_id
                or runtime.document_version_id != source.document_version_id
                or runtime.source_metadata_revision_id != source.source_metadata_revision_id
                or runtime.source_sha256 != source.source_sha256
            ):
                raise ValueError("runtime ingestion metadata does not match source revision")
        if index is not None:
            if runtime is None:
                raise ValueError("index snapshot requires runtime ingestion metadata")
            if (
                index.runtime_ingestion_revision_id != runtime.runtime_ingestion_revision_id
                or index.source_sha256 != runtime.source_sha256
                or index.canonical_text_sha256 != runtime.canonical_text_sha256
            ):
                raise ValueError("index snapshot does not match runtime ingestion revision")
        if lifecycle is not None and lifecycle.document_version_id != source.document_version_id:
            raise ValueError("lifecycle result belongs to another document version")
        return self

    @property
    def evidence_eligible(self) -> bool:
        """Only a complete, frozen, ACTIVE+INDEXED aggregate is eligible."""

        source = self.source_metadata
        runtime = self.runtime_ingestion_metadata
        index = self.index_snapshot_metadata
        lifecycle = self.lifecycle_at_result
        return bool(
            runtime is not None
            and index is not None
            and lifecycle is not None
            and source.revision_status is RevisionStatus.FROZEN
            and runtime.revision_status is RevisionStatus.FROZEN
            and index.revision_status is RevisionStatus.FROZEN
            and runtime.processing_status is ProcessingStatus.INDEXED
            and runtime.invalidated_at is None
            and index.invalidated_at is None
            and lifecycle.result_status is LifecycleResultStatus.RESOLVED
            and lifecycle.lifecycle_status is LifecycleStatus.ACTIVE
        )
