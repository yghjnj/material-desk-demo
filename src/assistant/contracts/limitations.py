"""Serializable decisions that preserve limitation evidence and resolution."""

from __future__ import annotations

from pydantic import model_validator

from .base import ContractModel, NonEmptyString, Sha256Digest, UTCDateTime, UUIDString
from .enums import (
    DisclosurePlacement,
    LimitationResolutionAction,
    LimitationStatus,
)


class LimitationDisclosure(ContractModel):
    required: bool
    text: NonEmptyString | None = None
    placement: DisclosurePlacement | None = None

    @model_validator(mode="after")
    def validate_disclosure(self) -> "LimitationDisclosure":
        if self.required:
            if self.text is None or self.placement is None:
                raise ValueError("required disclosure needs text and placement")
        elif self.text is not None or self.placement is not None:
            raise ValueError("non-required disclosure must not carry text or placement")
        return self


class LimitationResolution(ContractModel):
    action: LimitationResolutionAction
    resolved_by: UUIDString | None = None
    resolved_at: UTCDateTime | None = None
    notes: NonEmptyString | None = None

    @model_validator(mode="after")
    def validate_resolution_audit(self) -> "LimitationResolution":
        if (self.resolved_by is None) != (self.resolved_at is None):
            raise ValueError("resolved_by and resolved_at must be supplied together")
        return self


class LimitationDecision(ContractModel):
    limitation_decision_id: UUIDString
    status: LimitationStatus
    reason_code: NonEmptyString
    limitation_fact_ids: tuple[UUIDString, ...]
    runtime_locator_refs: tuple[UUIDString, ...]
    technical_citation_refs: tuple[UUIDString, ...]
    affected_claim_ids: tuple[UUIDString, ...]
    disclosure: LimitationDisclosure
    resolution: LimitationResolution
    policy_version: NonEmptyString
    input_snapshot_hash: Sha256Digest

    @model_validator(mode="after")
    def preserve_limitation_evidence(self) -> "LimitationDecision":
        if not (
            self.limitation_fact_ids
            or self.runtime_locator_refs
            or self.technical_citation_refs
        ):
            raise ValueError("limitation decision must preserve structured evidence")
        if self.status is LimitationStatus.APPLIES:
            if not self.disclosure.required:
                raise ValueError("APPLIES limitation must be disclosed")
            if not self.affected_claim_ids:
                raise ValueError("APPLIES limitation requires affected claims")
        elif self.status is LimitationStatus.UNRESOLVED:
            if not self.affected_claim_ids:
                raise ValueError("UNRESOLVED limitation requires affected claims")
            if self.resolution.action is LimitationResolutionAction.QUALIFY:
                raise ValueError("UNRESOLVED limitation cannot be treated as resolved qualification")
        return self
