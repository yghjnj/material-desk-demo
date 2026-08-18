"""Evidence-gated QA response contract."""

from __future__ import annotations

from uuid import UUID

from pydantic import model_validator

from .base import NonEmptyString, Provenance, Sha256, UtcDatetime, VersionedContract
from .claims import (
    Answer,
    Claim,
    render_claim_statements,
    validate_claim_evidence_graph,
)
from .enums import (
    ClaimType,
    EvidenceRelation,
    QAOutcome,
    SupportLevel,
    SupportStatus,
    VerificationStatus,
)
from .errors import RefusalReason
from .evidence import CustomerAttribution, TechnicalCitation
from .limitations import LimitationDecision


class QAResponse(VersionedContract):
    response_id: UUID
    case_id: UUID
    query_sha256: Sha256
    retrieval_run_id: UUID
    outcome: QAOutcome
    answer: Answer | None = None
    claims: tuple[Claim, ...] = ()
    technical_citations: tuple[TechnicalCitation, ...] = ()
    customer_attributions: tuple[CustomerAttribution, ...] = ()
    refusal_reason: RefusalReason | None = None
    unanswered_aspects: tuple[NonEmptyString, ...] = ()
    clarification_questions: tuple[NonEmptyString, ...] = ()
    warnings: tuple[NonEmptyString, ...] = ()
    limitation_decisions: tuple[LimitationDecision, ...] = ()
    input_snapshot_hash: Sha256
    provenance: Provenance
    created_at: UtcDatetime

    @model_validator(mode="after")
    def validate_outcome_and_evidence(self) -> "QAResponse":
        if self.outcome == QAOutcome.ANSWERED:
            if self.answer is None or self.refusal_reason is not None:
                raise ValueError("ANSWERED requires an answer and no refusal reason")
            if self.unanswered_aspects or self.clarification_questions:
                raise ValueError("ANSWERED cannot carry unresolved aspects or questions")
        elif self.outcome == QAOutcome.PARTIAL:
            if self.answer is None or not self.unanswered_aspects:
                raise ValueError("PARTIAL requires an answer and unanswered aspects")
            if self.refusal_reason is not None:
                raise ValueError("PARTIAL is not a refusal")
        elif self.outcome == QAOutcome.NEEDS_CLARIFICATION:
            if self.answer is not None or not self.clarification_questions:
                raise ValueError("NEEDS_CLARIFICATION requires questions and no answer")
            if self.refusal_reason is not None:
                raise ValueError("NEEDS_CLARIFICATION is not a refusal")
        elif self.outcome == QAOutcome.REFUSED:
            if self.answer is not None or self.refusal_reason is None:
                raise ValueError("REFUSED requires refusal_reason and no answer")

        claims = {claim.claim_id: claim for claim in self.claims}
        citations = {item.technical_citation_id: item for item in self.technical_citations}
        attributions = {
            item.customer_attribution_id: item for item in self.customer_attributions
        }
        validate_claim_evidence_graph(
            self.claims,
            self.technical_citations,
            self.customer_attributions,
        )

        for citation in self.technical_citations:
            if citation.case_id != self.case_id or citation.retrieval_run_id != self.retrieval_run_id:
                raise ValueError("technical citation is outside the QA case or retrieval run")
        for attribution in self.customer_attributions:
            if attribution.case_id != self.case_id:
                raise ValueError("customer attribution is outside the QA case")
            if attribution.claim_id is not None and attribution.claim_id not in claims:
                raise ValueError("customer attribution references an unknown claim")

        if self.answer is not None:
            unknown = set(self.answer.claim_ids) - set(claims)
            if unknown:
                raise ValueError("answer references unknown claims")
            answer_claims = tuple(claims[claim_id] for claim_id in self.answer.claim_ids)
            if self.answer.text != render_claim_statements(answer_claims):
                raise ValueError("answer text must use the deterministic claim template")
            # An ANSWERED/PARTIAL response must contain at least one technical
            # fact.  Customer statements are contextual inputs, never a
            # substitute for a retrievable technical source.
            fact_claims = tuple(
                claim for claim in answer_claims if claim.claim_type is ClaimType.FACT
            )
            if not fact_claims:
                raise ValueError("answer requires at least one technical FACT claim")
            qualified_claim_ids: set[UUID] = set()
            for claim in answer_claims:
                if claim.support_status in {
                    SupportStatus.UNSUPPORTED,
                    SupportStatus.CONFLICTED,
                }:
                    raise ValueError("unsupported or conflicted claim cannot enter answer text")
                if claim.support_status is SupportStatus.QUALIFIED:
                    qualified_claim_ids.add(claim.claim_id)
                if claim.claim_type is ClaimType.FACT and not claim.technical_citation_ids:
                    raise ValueError("technical FACT in answer requires technical evidence")
                for citation_id in claim.technical_citation_ids:
                    citation = citations[citation_id]
                    if not (
                        citation.evidence_eligible
                        and citation.relation == EvidenceRelation.SUPPORTS
                        and citation.support_level == SupportLevel.DIRECT
                        and citation.verification_status == VerificationStatus.EXACT_MATCH
                    ):
                        raise ValueError("answer evidence must be direct, exact, and eligible")
            affected_by_limitation = {
                claim_id
                for decision in self.limitation_decisions
                for claim_id in decision.affected_claim_ids
            }
            if not qualified_claim_ids.issubset(affected_by_limitation):
                raise ValueError("QUALIFIED answer claim requires a limitation decision")
        known_claim_ids = set(claims)
        if any(
            not set(decision.affected_claim_ids).issubset(known_claim_ids)
            for decision in self.limitation_decisions
        ):
            raise ValueError("limitation decision references an unknown QA claim")
        return self
