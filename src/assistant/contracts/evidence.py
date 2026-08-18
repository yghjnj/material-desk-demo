"""Separate evidence contracts for technical facts and customer statements."""

from __future__ import annotations

from pydantic import model_validator

from .base import ContractModel, NonEmptyString, Sha256Digest, UUIDString
from .customer_locators import GoldCustomerLocator, RuntimeCustomerLocator
from .document_locators import RuntimeDocumentLocator
from .facts import RuntimeKnowledgeFact
from .enums import (
    AttributionStatus,
    CitationRelation,
    FactKind,
    SupportLevel,
    VerificationStatus,
)
from .retrieval import RetrievalResult


class TechnicalCitation(ContractModel):
    technical_citation_id: UUIDString
    claim_id: UUIDString
    case_id: UUIDString
    runtime_fact_id: UUIDString
    fact_kind: FactKind
    retrieval_run_id: UUIDString
    result_id: UUIDString
    document_id: UUIDString
    document_version_id: UUIDString
    chunk_id: UUIDString
    runtime_document_locator: RuntimeDocumentLocator
    # Eligible evidence is serialized with the exact runtime objects that
    # produced the citation.  IDs alone cannot prove that a citation came from
    # a real retrieval result or runtime fact.
    retrieval_result: RetrievalResult | None = None
    runtime_fact: RuntimeKnowledgeFact | None = None
    relation: CitationRelation
    support_level: SupportLevel
    verification_status: VerificationStatus
    evidence_eligible: bool

    @model_validator(mode="after")
    def validate_runtime_binding(self) -> "TechnicalCitation":
        locator = self.runtime_document_locator
        if (
            locator.document_id != self.document_id
            or locator.document_version_id != self.document_version_id
            or locator.chunk_id != self.chunk_id
        ):
            raise ValueError("technical citation IDs must match runtime locator")
        eligible = (
            self.relation is CitationRelation.SUPPORTS
            and self.support_level is SupportLevel.DIRECT
            and self.verification_status is VerificationStatus.EXACT_MATCH
        )
        if self.evidence_eligible and not eligible:
            raise ValueError(
                "eligible technical citation must be SUPPORTS + DIRECT + EXACT_MATCH"
            )
        # Every citation, including an excluded/failed one, must carry the
        # immutable runtime snapshots that prove where its IDs came from.
        result = self.retrieval_result
        fact = self.runtime_fact
        if result is None or fact is None:
            raise ValueError("technical citation requires retrieval_result and runtime_fact snapshots")
        if result.result_id != self.result_id:
            raise ValueError("citation result_id must match retrieval_result")
        if result.runtime_fact_id != self.runtime_fact_id:
            raise ValueError("citation runtime_fact_id must match retrieval_result")
        if fact.runtime_fact_id != self.runtime_fact_id:
            raise ValueError("citation runtime_fact_id must match runtime_fact")
        if fact.fact_kind is not self.fact_kind:
            raise ValueError("citation fact_kind must match runtime_fact")
        if self.evidence_eligible and not result.evidence_eligible:
            raise ValueError("citation cannot promote an ineligible retrieval result")
        if (
            result.document_id != self.document_id
            or result.document_version_id != self.document_version_id
            or result.chunk_id != self.chunk_id
            or result.runtime_document_locator != locator
        ):
            raise ValueError("citation locator and IDs must match retrieval_result")
        if (
            fact.document_id != self.document_id
            or fact.document_version_id != self.document_version_id
            or fact.statement != locator.quote
            or fact.statement_sha256 != locator.quote_sha256
            or locator not in fact.runtime_document_locators
        ):
            raise ValueError("citation locator and quote must match runtime_fact")
        return self


class CustomerAttribution(ContractModel):
    customer_attribution_id: UUIDString
    case_id: UUIDString
    message_id: UUIDString
    message_sha256: Sha256Digest
    runtime_customer_locator: RuntimeCustomerLocator
    field_path: NonEmptyString | None = None
    claim_id: UUIDString | None = None
    attribution_status: AttributionStatus

    @model_validator(mode="after")
    def validate_runtime_binding(self) -> "CustomerAttribution":
        locator = self.runtime_customer_locator
        if locator.message_id != self.message_id or locator.message_sha256 != self.message_sha256:
            raise ValueError("customer attribution must match runtime message locator")
        if self.field_path is None and self.claim_id is None:
            raise ValueError("customer attribution requires field_path or claim_id")
        return self


class ExpectedCustomerAttribution(ContractModel):
    expected_customer_attribution_id: UUIDString
    case_id: UUIDString
    message_id: UUIDString
    message_sha256: Sha256Digest
    gold_customer_locator: GoldCustomerLocator
    field_path: NonEmptyString | None = None
    claim_id: UUIDString | None = None

    @model_validator(mode="after")
    def validate_gold_binding(self) -> "ExpectedCustomerAttribution":
        locator = self.gold_customer_locator
        if locator.message_id != self.message_id or locator.message_sha256 != self.message_sha256:
            raise ValueError("expected attribution must match gold message locator")
        if self.field_path is None and self.claim_id is None:
            raise ValueError("expected attribution requires field_path or claim_id")
        return self
