"""Claims and answer text shared by QA and drafting."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal
from uuid import UUID

from pydantic import Field, model_validator

from .base import NonEmptyString, Sha256, VersionedContract, sha256_text
from .enums import AttributionStatus, ClaimType, SupportStatus

if TYPE_CHECKING:
    from .evidence import CustomerAttribution, TechnicalCitation


CLAIM_TEMPLATE_SEPARATOR = "\n"


class Claim(VersionedContract):
    claim_id: UUID
    statement: NonEmptyString
    statement_sha256: Sha256
    claim_type: ClaimType
    support_status: SupportStatus
    technical_citation_ids: tuple[UUID, ...] = ()
    customer_attribution_ids: tuple[UUID, ...] = ()

    @model_validator(mode="after")
    def evidence_shape(self) -> "Claim":
        if sha256_text(self.statement) != self.statement_sha256:
            raise ValueError("claim statement_sha256 does not match statement")
        if len(self.technical_citation_ids) != len(set(self.technical_citation_ids)):
            raise ValueError("claim technical citation IDs must be unique")
        if len(self.customer_attribution_ids) != len(set(self.customer_attribution_ids)):
            raise ValueError("claim customer attribution IDs must be unique")
        if self.technical_citation_ids and self.customer_attribution_ids:
            raise ValueError("mixed technical/customer facts must be split into separate claims")
        if self.claim_type is ClaimType.FACT:
            if self.customer_attribution_ids:
                raise ValueError("technical FACT accepts only TechnicalCitation evidence")
            if self.support_status in {SupportStatus.SUPPORTED, SupportStatus.QUALIFIED}:
                if not self.technical_citation_ids:
                    raise ValueError("supported or qualified FACT requires technical evidence")
        elif self.claim_type is ClaimType.CUSTOMER_STATEMENT:
            if self.technical_citation_ids:
                raise ValueError("customer statement accepts only CustomerAttribution evidence")
            if self.support_status in {SupportStatus.SUPPORTED, SupportStatus.QUALIFIED}:
                if not self.customer_attribution_ids:
                    raise ValueError("supported customer statement requires customer evidence")
        elif self.technical_citation_ids or self.customer_attribution_ids:
            raise ValueError("assumption/unknown claims cannot masquerade as sourced facts")
        elif self.support_status in {SupportStatus.SUPPORTED, SupportStatus.QUALIFIED}:
            raise ValueError("assumption/unknown claims cannot be marked supported")
        if self.support_status == SupportStatus.UNSUPPORTED and (
            self.technical_citation_ids or self.customer_attribution_ids
        ):
            raise ValueError("unsupported claim cannot cite supporting evidence")
        return self


class Answer(VersionedContract):
    answer_id: UUID
    render_mode: Literal["CLAIM_STATEMENTS_V1"] = "CLAIM_STATEMENTS_V1"
    text: NonEmptyString
    text_sha256: Sha256
    claim_ids: tuple[UUID, ...] = Field(min_length=1)
    limitations: tuple[NonEmptyString, ...] = ()
    next_questions: tuple[NonEmptyString, ...] = ()

    @model_validator(mode="after")
    def answer_hash_and_claim_order(self) -> "Answer":
        if sha256_text(self.text) != self.text_sha256:
            raise ValueError("answer text_sha256 does not match text")
        if len(self.claim_ids) != len(set(self.claim_ids)):
            raise ValueError("answer claim IDs must be unique")
        return self


def render_claim_statements(claims: tuple[Claim, ...]) -> str:
    return CLAIM_TEMPLATE_SEPARATOR.join(claim.statement for claim in claims)


def validate_claim_evidence_graph(
    claims: tuple[Claim, ...],
    technical_citations: tuple["TechnicalCitation", ...],
    customer_attributions: tuple["CustomerAttribution", ...],
) -> None:
    citations = {item.technical_citation_id: item for item in technical_citations}
    attributions = {
        item.customer_attribution_id: item for item in customer_attributions
    }
    claim_map = {claim.claim_id: claim for claim in claims}
    if len(claim_map) != len(claims):
        raise ValueError("claim IDs must be unique")
    if len(citations) != len(technical_citations):
        raise ValueError("technical citation IDs must be unique")
    if len(attributions) != len(customer_attributions):
        raise ValueError("customer attribution IDs must be unique")

    for citation in technical_citations:
        claim = claim_map.get(citation.claim_id)
        if claim is None or citation.technical_citation_id not in claim.technical_citation_ids:
            raise ValueError("technical citation must be bound in both graph directions")
        if claim.claim_type is not ClaimType.FACT:
            raise ValueError("TechnicalCitation may support only FACT claims")
    for attribution in customer_attributions:
        if attribution.claim_id is None:
            continue
        claim = claim_map.get(attribution.claim_id)
        if claim is None or attribution.customer_attribution_id not in claim.customer_attribution_ids:
            raise ValueError("customer attribution must be bound in both graph directions")
        if claim.claim_type is not ClaimType.CUSTOMER_STATEMENT:
            raise ValueError("CustomerAttribution may support only CUSTOMER_STATEMENT claims")

    for claim in claims:
        for citation_id in claim.technical_citation_ids:
            citation = citations.get(citation_id)
            if citation is None or citation.claim_id != claim.claim_id:
                raise ValueError("claim technical evidence has a missing or wrong reverse edge")
            locator = citation.runtime_document_locator
            if locator.quote != claim.statement or locator.quote_sha256 != claim.statement_sha256:
                raise ValueError("technical claim must exactly match its evidence quote in v1")
        for attribution_id in claim.customer_attribution_ids:
            attribution = attributions.get(attribution_id)
            if attribution is None or attribution.claim_id != claim.claim_id:
                raise ValueError("claim customer evidence has a missing or wrong reverse edge")
            locator = attribution.runtime_customer_locator
            if attribution.attribution_status is not AttributionStatus.EXACT_MATCH:
                raise ValueError("customer claim requires an EXACT_MATCH attribution")
            if locator.quote != claim.statement or locator.quote_sha256 != claim.statement_sha256:
                raise ValueError("customer claim must exactly match its source quote in v1")
