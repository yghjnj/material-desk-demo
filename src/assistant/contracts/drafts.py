"""Evidence-based reply drafts and human review decisions."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import Field, model_validator

from .base import (
    NonEmptyString,
    Provenance,
    Sha256,
    UtcDatetime,
    VersionedContract,
    sha256_text,
)
from .claims import Claim, render_claim_statements, validate_claim_evidence_graph
from .enums import (
    ClaimType,
    DraftPurpose,
    EvidenceRelation,
    ReviewDecisionType,
    ReviewStatus,
    SupportLevel,
    SupportStatus,
    VerificationStatus,
)
from .evidence import CustomerAttribution, TechnicalCitation
from .limitations import LimitationDecision


class DraftNextAction(VersionedContract):
    action_type: Literal["ASK_CUSTOMER", "INTERNAL_REVIEW", "MANUAL_FOLLOW_UP"]
    description: NonEmptyString
    owner_role: NonEmptyString


class ReplyDraft(VersionedContract):
    draft_id: UUID
    revision: int = Field(ge=1)
    case_id: UUID
    purpose: DraftPurpose
    requirement_revision: int = Field(ge=1)
    retrieval_run_ids: tuple[UUID, ...] = ()
    input_snapshot_hash: Sha256
    subject: NonEmptyString
    body_render_mode: Literal["CLAIM_STATEMENTS_V1", "STRUCTURED_QUESTIONS_V1"]
    body: NonEmptyString
    body_sha256: Sha256
    claims: tuple[Claim, ...] = ()
    technical_citations: tuple[TechnicalCitation, ...] = ()
    customer_attributions: tuple[CustomerAttribution, ...] = ()
    assumptions: tuple[NonEmptyString, ...] = ()
    questions_to_confirm: tuple[NonEmptyString, ...] = ()
    next_actions: tuple[DraftNextAction, ...] = ()
    limitations: tuple[NonEmptyString, ...] = ()
    limitation_decisions: tuple[LimitationDecision, ...] = ()
    blocking_field_paths: tuple[NonEmptyString, ...] = ()
    review_status: ReviewStatus = ReviewStatus.REQUIRES_HUMAN_REVIEW
    provenance: Provenance
    created_at: UtcDatetime

    @model_validator(mode="after")
    def validate_draft_gate(self) -> "ReplyDraft":
        if sha256_text(self.body) != self.body_sha256:
            raise ValueError("draft body_sha256 does not match body")
        if self.review_status != ReviewStatus.REQUIRES_HUMAN_REVIEW:
            raise ValueError("machine-generated draft must require human review")
        if self.blocking_field_paths and self.purpose != DraftPurpose.CLARIFICATION:
            raise ValueError("blocking fields permit only a clarification draft")
        if self.purpose == DraftPurpose.CLARIFICATION and not self.questions_to_confirm:
            raise ValueError("clarification draft requires questions to confirm")
        if self.purpose == DraftPurpose.TECHNICAL_RESPONSE and not self.technical_citations:
            raise ValueError("technical response draft requires technical citations")

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
            if citation.case_id != self.case_id:
                raise ValueError("technical citation is outside the draft claim graph")
            if citation.retrieval_run_id not in self.retrieval_run_ids:
                raise ValueError("technical citation uses an undeclared retrieval run")
            if not (
                citation.evidence_eligible
                and citation.relation == EvidenceRelation.SUPPORTS
                and citation.support_level == SupportLevel.DIRECT
                and citation.verification_status == VerificationStatus.EXACT_MATCH
            ):
                raise ValueError("draft technical evidence must be direct, exact, and eligible")
        for attribution in self.customer_attributions:
            if attribution.case_id != self.case_id:
                raise ValueError("customer attribution is outside the draft case")
            if attribution.claim_id is not None and attribution.claim_id not in claims:
                raise ValueError("customer attribution references an unknown draft claim")
        if self.purpose is DraftPurpose.TECHNICAL_RESPONSE:
            if self.body_render_mode != "CLAIM_STATEMENTS_V1" or not self.claims:
                raise ValueError("technical draft requires claim-template body and claims")
            if self.body != render_claim_statements(self.claims):
                raise ValueError("technical draft body must exactly render ordered claims")
            if not any(claim.claim_type is ClaimType.FACT for claim in self.claims):
                raise ValueError("technical draft requires at least one technical FACT claim")
            if any(
                claim.claim_type is ClaimType.FACT and not claim.technical_citation_ids
                for claim in self.claims
            ):
                raise ValueError("technical FACT in draft requires technical evidence")
            if any(
                claim.support_status in {SupportStatus.UNSUPPORTED, SupportStatus.CONFLICTED}
                for claim in self.claims
            ):
                raise ValueError("unsupported or conflicted claim cannot enter draft body")
        else:
            if self.body_render_mode != "STRUCTURED_QUESTIONS_V1":
                raise ValueError("clarification draft requires structured-question body")
            if self.claims or self.technical_citations:
                raise ValueError("clarification body cannot contain factual claims")
            expected_body = "\n".join(self.questions_to_confirm)
            if self.body != expected_body:
                raise ValueError("clarification body must exactly render structured questions")

        qualified_claim_ids = {
            claim.claim_id
            for claim in self.claims
            if claim.support_status is SupportStatus.QUALIFIED
        }
        affected_by_limitation = {
            claim_id
            for decision in self.limitation_decisions
            for claim_id in decision.affected_claim_ids
        }
        if not qualified_claim_ids.issubset(affected_by_limitation):
            raise ValueError("QUALIFIED draft claim requires a limitation decision")
        if any(
            not set(decision.affected_claim_ids).issubset(claims)
            for decision in self.limitation_decisions
        ):
            raise ValueError("limitation decision references an unknown draft claim")
        return self


class ReviewDecision(VersionedContract):
    review_decision_id: UUID
    case_id: UUID
    draft_id: UUID
    draft_revision: int = Field(ge=1)
    draft_content_sha256: Sha256
    decision: ReviewDecisionType
    reviewer_id: UUID
    reviewer_role: NonEmptyString
    reviewed_at: UtcDatetime
    comment: NonEmptyString | None = None
    unresolved_blockers: tuple[NonEmptyString, ...] = ()
    audit_event_id: UUID

    @model_validator(mode="after")
    def validate_human_decision(self) -> "ReviewDecision":
        if self.decision == ReviewDecisionType.APPROVE and self.unresolved_blockers:
            raise ValueError("a draft with unresolved blockers cannot be approved")
        if self.decision in {
            ReviewDecisionType.REJECT,
            ReviewDecisionType.REQUEST_CHANGES,
        } and self.comment is None:
            raise ValueError("rejection or change request requires a comment")
        return self
