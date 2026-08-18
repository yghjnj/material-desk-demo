"""Immutable retrieval-run and result contracts."""

from __future__ import annotations

from uuid import UUID

from pydantic import Field, model_validator

from .base import DecimalString, NonEmptyString, Sha256, UtcDatetime, VersionedContract
from .document_locators import RuntimeDocumentLocator
from .enums import (
    EnvironmentNamespace,
    LifecycleStatus,
    ProcessingStatus,
)


class RetrievalFilters(VersionedContract):
    lifecycle_statuses: tuple[LifecycleStatus, ...] = (LifecycleStatus.ACTIVE,)
    processing_statuses: tuple[ProcessingStatus, ...] = (ProcessingStatus.INDEXED,)
    languages: tuple[NonEmptyString, ...] = ()
    regions: tuple[NonEmptyString, ...] = ()
    product_codes: tuple[NonEmptyString, ...] = ()
    document_version_ids: tuple[UUID, ...] = ()
    historical_audit: bool = False

    @model_validator(mode="after")
    def normal_mode_is_active_and_indexed(self) -> "RetrievalFilters":
        if not self.historical_audit:
            if self.lifecycle_statuses != (LifecycleStatus.ACTIVE,):
                raise ValueError("normal retrieval permits only ACTIVE lifecycle")
            if self.processing_statuses != (ProcessingStatus.INDEXED,):
                raise ValueError("normal retrieval permits only INDEXED processing state")
        return self


class RetrievalResult(VersionedContract):
    result_id: UUID
    rank: int = Field(ge=1)
    document_id: UUID
    document_version_id: UUID
    runtime_fact_id: UUID
    chunk_id: UUID
    runtime_document_locator: RuntimeDocumentLocator
    score: DecimalString
    score_type: NonEmptyString
    evidence_eligible: bool
    exclusion_reason_codes: tuple[NonEmptyString, ...] = ()

    @model_validator(mode="after")
    def validate_result_identity(self) -> "RetrievalResult":
        locator = self.runtime_document_locator
        if locator.document_id != self.document_id:
            raise ValueError("result and locator document IDs differ")
        if locator.document_version_id != self.document_version_id:
            raise ValueError("result and locator document version IDs differ")
        if locator.chunk_id != self.chunk_id:
            raise ValueError("result and locator chunk IDs differ")
        if self.evidence_eligible == bool(self.exclusion_reason_codes):
            raise ValueError("eligible result has no exclusions; ineligible result requires them")
        return self


class RetrievalRun(VersionedContract):
    retrieval_run_id: UUID
    case_id: UUID
    query_sha256: Sha256
    as_of: UtcDatetime
    metadata_snapshot_hash: Sha256
    lifecycle_policy_version: NonEmptyString
    filters: RetrievalFilters
    top_k: int = Field(ge=1)
    index_snapshot_id: UUID
    index_version: NonEmptyString
    retrieval_config_version: NonEmptyString
    environment_namespace: EnvironmentNamespace
    results: tuple[RetrievalResult, ...] = ()

    @model_validator(mode="after")
    def validate_ranked_results(self) -> "RetrievalRun":
        if len(self.results) > self.top_k:
            raise ValueError("retrieval results cannot exceed top_k")
        ids = [item.result_id for item in self.results]
        ranks = [item.rank for item in self.results]
        if len(ids) != len(set(ids)) or len(ranks) != len(set(ranks)):
            raise ValueError("retrieval result IDs and ranks must be unique")
        if ranks != sorted(ranks):
            raise ValueError("retrieval results must be ordered by ascending rank")
        return self
