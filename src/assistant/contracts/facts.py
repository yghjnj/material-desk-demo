"""Gold document facts and independently produced runtime knowledge facts."""

from __future__ import annotations

from pydantic import Field, model_validator

from .base import ContractModel, NonEmptyString, Sha256Digest, UUIDString, sha256_text
from .document_locators import GoldDocumentLocator, RuntimeDocumentLocator
from .documents import ApplicabilityScope, Authority
from .enums import AuthorityClass, FactKind, LocatorStatus, RevisionStatus
from .measurements import Measurement


class GoldDocumentFact(ContractModel):
    gold_fact_id: UUIDString
    gold_revision_status: RevisionStatus
    document_id: UUIDString
    document_version_id: UUIDString
    fact_key: NonEmptyString
    fact_kind: FactKind
    statement: NonEmptyString
    statement_sha256: Sha256Digest
    measurement: Measurement | None = None
    test_method: NonEmptyString | None = None
    test_conditions: tuple[NonEmptyString, ...] = ()
    authority: Authority
    precedence: int
    precedence_policy_version: NonEmptyString
    applicability_scope: ApplicabilityScope
    gold_document_locators: tuple[GoldDocumentLocator, ...] = Field(min_length=1)
    is_synthetic: bool
    provenance: tuple[NonEmptyString, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_gold_fact(self) -> "GoldDocumentFact":
        if sha256_text(self.statement) != self.statement_sha256:
            raise ValueError("statement_sha256 does not match statement")
        for locator in self.gold_document_locators:
            if (
                locator.document_id != self.document_id
                or locator.document_version_id != self.document_version_id
            ):
                raise ValueError("gold fact locator belongs to another document version")
        if self.gold_revision_status is RevisionStatus.FROZEN and any(
            locator.locator_status is not LocatorStatus.VERIFIED
            for locator in self.gold_document_locators
        ):
            raise ValueError("frozen gold fact requires VERIFIED locators")
        if self.is_synthetic:
            if self.authority.authority_class is not AuthorityClass.SYNTHETIC_DEMO:
                raise ValueError("synthetic fact requires SYNTHETIC_DEMO authority")
        elif self.authority.authority_class is AuthorityClass.SYNTHETIC_DEMO:
            raise ValueError("SYNTHETIC_DEMO authority requires is_synthetic=true")
        return self


class RuntimeKnowledgeFact(ContractModel):
    runtime_fact_id: UUIDString
    runtime_ingestion_revision_id: UUIDString
    index_snapshot_id: UUIDString
    document_id: UUIDString
    document_version_id: UUIDString
    fact_key: NonEmptyString
    fact_kind: FactKind
    statement: NonEmptyString
    statement_sha256: Sha256Digest
    measurement: Measurement | None = None
    test_method: NonEmptyString | None = None
    test_conditions: tuple[NonEmptyString, ...] = ()
    authority: Authority
    precedence: int
    precedence_policy_version: NonEmptyString
    applicability_scope: ApplicabilityScope
    runtime_document_locators: tuple[RuntimeDocumentLocator, ...] = Field(min_length=1)
    is_synthetic: bool
    provenance: tuple[NonEmptyString, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_runtime_fact(self) -> "RuntimeKnowledgeFact":
        if sha256_text(self.statement) != self.statement_sha256:
            raise ValueError("statement_sha256 does not match statement")
        for locator in self.runtime_document_locators:
            if (
                locator.document_id != self.document_id
                or locator.document_version_id != self.document_version_id
            ):
                raise ValueError("runtime fact locator belongs to another document version")
        if self.is_synthetic:
            if self.authority.authority_class is not AuthorityClass.SYNTHETIC_DEMO:
                raise ValueError("synthetic fact requires SYNTHETIC_DEMO authority")
        elif self.authority.authority_class is AuthorityClass.SYNTHETIC_DEMO:
            raise ValueError("SYNTHETIC_DEMO authority requires is_synthetic=true")
        return self


class FactMatch(ContractModel):
    fact_match_id: UUIDString
    gold_fact_id: UUIDString
    runtime_fact_id: UUIDString
    matches: bool
    reason_codes: tuple[NonEmptyString, ...]

    @model_validator(mode="after")
    def require_reason_for_mismatch(self) -> "FactMatch":
        if not self.matches and not self.reason_codes:
            raise ValueError("fact mismatch requires reason_codes")
        return self
