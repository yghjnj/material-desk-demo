"""B-owned wrappers around the frozen A contract types.

This module intentionally does not redefine any A shared schema.  Gold
locators, gold facts, customer locators, lifecycle results, and runtime
payloads are imported from ``assistant.contracts`` and are carried by the
small B expectation wrappers below where a dataset-only label is needed.
"""

from __future__ import annotations

from typing import ClassVar, Literal
from uuid import UUID

from pydantic import Field, model_validator

from ..contracts.base import (
    ContractModel,
    JsonPointer,
    NonEmptyString,
    Sha256,
    UtcDatetime,
    VersionedContract,
)
from ..contracts.customer_locators import (
    GoldCustomerLocator,
    RuntimeCustomerLocator,
)
from ..contracts.documents import (
    LifecycleAtResult,
    SourceDocumentMetadata,
)
from ..contracts.document_locators import (
    GoldDocumentLocator,
    RuntimeDocumentLocator,
)
from ..contracts.enums import (
    DatasetSplit,
    FieldStatus,
    LifecycleResultStatus,
    LimitationAction,
    LimitationStatus,
    NormalizationStatus,
    ValueState,
)
from ..contracts.evidence import (
    CustomerAttribution,
    ExpectedCustomerAttribution as AExpectedCustomerAttribution,
    TechnicalCitation,
)
from ..contracts.facts import (
    GoldDocumentFact,
    RuntimeKnowledgeFact,
)
from ..contracts.limitations import LimitationDecision
from ..contracts.measurements import Measurement, ValuePayload
from ..contracts.requirements import FieldAnnotation
from ..contracts.runtime import ExecutionNamespaceKey


B_SCHEMA_VERSION = "1.2.0"
B_MIGRATION_FROM = "B-DATA-CONTRACT-v1.1"


class BDatasetExecutionContext(ContractModel):
    """B-private context sidecar keyed by an A namespace hash.

    The fields in this object are dataset bookkeeping only.  In particular,
    none are appended to or substituted for A's nine-field
    ``ExecutionNamespaceKey``.  Consumers must persist the complete A key and
    use ``execution_namespace_key_hash`` as the foreign-key relationship.
    """

    context_schema_version: Literal["1.2.0"] = B_SCHEMA_VERSION
    dataset_id: UUID
    dataset_version: NonEmptyString
    split: DatasetSplit
    connected_component_id: UUID
    fact_family_ids: tuple[UUID, ...] = Field(min_length=1)
    template_family_id: UUID
    source_lineage_ids: tuple[UUID, ...] = Field(min_length=1)
    execution_stage: NonEmptyString
    evidence_mode: NonEmptyString
    protocol_version: NonEmptyString
    run_scope_id: UUID
    execution_namespace_key_hash: Sha256

    @model_validator(mode="after")
    def validate_groups(self) -> "BDatasetExecutionContext":
        for field_name in ("fact_family_ids", "source_lineage_ids"):
            values = getattr(self, field_name)
            if len(values) != len(set(values)):
                raise ValueError(f"{field_name} must contain unique IDs")
        return self

    @classmethod
    def from_namespace_key(
        cls,
        namespace_key: ExecutionNamespaceKey,
        **context: object,
    ) -> "BDatasetExecutionContext":
        """Construct a sidecar bound to ``namespace_key.namespace_hash``."""

        if "execution_namespace_key_hash" in context:
            supplied = context["execution_namespace_key_hash"]
            if str(supplied) != namespace_key.namespace_hash:
                raise ValueError("context namespace hash does not match A key")
        context["execution_namespace_key_hash"] = namespace_key.namespace_hash
        return cls.model_validate(context)

    def assert_namespace_key(self, namespace_key: ExecutionNamespaceKey) -> None:
        """Raise when this sidecar is paired with a different A key."""

        if self.execution_namespace_key_hash != namespace_key.namespace_hash:
            raise ValueError("B context is bound to a different A namespace key")


class ExpectedFieldAnnotation(ContractModel):
    """B gold view of an A ``FieldAnnotation``.

    ``shared_field_annotation`` remains the authoritative A payload.  The
    duplicated ``status`` and ``value_state`` labels are an explicit gold
    projection and are checked for exact agreement, keeping the two axes
    orthogonal while preventing a B-only schema from replacing A's type.
    """

    expected_field_annotation_id: UUID
    shared_field_annotation: FieldAnnotation
    status: FieldStatus
    value_state: ValueState
    gold_source_locator_ids: tuple[UUID, ...] = ()
    gold_customer_attribution_ids: tuple[UUID, ...] = ()
    expected_annotation_revision: int = Field(default=1, ge=1)
    annotation_version: NonEmptyString = "1.2.0"
    supersedes_annotation_id: UUID | None = None
    superseded_by_annotation_id: UUID | None = None

    @model_validator(mode="after")
    def validate_projection(self) -> "ExpectedFieldAnnotation":
        payload = self.shared_field_annotation
        if self.expected_field_annotation_id != payload.annotation_id:
            raise ValueError("expected annotation ID must match A FieldAnnotation")
        if self.status is not payload.status:
            raise ValueError("gold status must match A FieldAnnotation.status")
        if self.value_state is not payload.value.value_state:
            raise ValueError("gold value_state must match A value payload")
        if self.gold_source_locator_ids != payload.source_locator_ids:
            raise ValueError("gold source IDs must match A payload source IDs")
        if self.gold_customer_attribution_ids != payload.customer_attribution_ids:
            raise ValueError("gold attribution IDs must match A payload attribution IDs")
        if self.supersedes_annotation_id != payload.supersedes_annotation_id:
            raise ValueError("supersedes ID must match A FieldAnnotation")
        if self.superseded_by_annotation_id != payload.superseded_by_annotation_id:
            raise ValueError("superseded_by ID must match A FieldAnnotation")
        return self

    @classmethod
    def from_shared(
        cls,
        payload: FieldAnnotation,
        *,
        annotation_version: str = B_SCHEMA_VERSION,
    ) -> "ExpectedFieldAnnotation":
        return cls(
            expected_field_annotation_id=payload.annotation_id,
            shared_field_annotation=payload,
            status=payload.status,
            value_state=payload.value.value_state,
            gold_source_locator_ids=payload.source_locator_ids,
            gold_customer_attribution_ids=payload.customer_attribution_ids,
            annotation_version=annotation_version,
            supersedes_annotation_id=payload.supersedes_annotation_id,
            superseded_by_annotation_id=payload.superseded_by_annotation_id,
        )


class ExpectedLifecycleAtResult(ContractModel):
    """B label wrapper around A's exact ``LifecycleAtResult`` payload."""

    expected_lifecycle_at_result_id: UUID
    shared_lifecycle_at_result: LifecycleAtResult
    scenario_label: NonEmptyString | None = None
    expected_result_status: LifecycleResultStatus
    expected_error_code: NonEmptyString | None = None

    @model_validator(mode="after")
    def validate_projection(self) -> "ExpectedLifecycleAtResult":
        payload = self.shared_lifecycle_at_result
        if self.expected_result_status is not payload.result_status:
            raise ValueError("expected result status must match A LifecycleAtResult")
        if self.expected_error_code != payload.error_code:
            raise ValueError("expected error code must match A LifecycleAtResult")
        return self

    @classmethod
    def from_shared(
        cls,
        payload: LifecycleAtResult,
        *,
        expected_id: UUID | None = None,
        scenario_label: str | None = None,
    ) -> "ExpectedLifecycleAtResult":
        return cls(
            expected_lifecycle_at_result_id=expected_id or payload.document_version_id,
            shared_lifecycle_at_result=payload,
            scenario_label=scenario_label,
            expected_result_status=payload.result_status,
            expected_error_code=payload.error_code,
        )


class ExpectedLimitationDecision(ContractModel):
    """B expectation wrapper mapped to A ``LimitationDecision`` fields."""

    expected_limitation_decision_id: UUID
    shared_limitation_decision: LimitationDecision
    expected_status: LimitationStatus
    expected_action: LimitationAction
    scenario_label: NonEmptyString | None = None

    @model_validator(mode="after")
    def validate_projection(self) -> "ExpectedLimitationDecision":
        payload = self.shared_limitation_decision
        if self.expected_limitation_decision_id != payload.limitation_decision_id:
            raise ValueError("expected limitation ID must match A payload")
        if self.expected_status is not payload.status:
            raise ValueError("expected limitation status must match A payload")
        if self.expected_action is not payload.resolution.action:
            raise ValueError("expected limitation action must match A resolution")
        return self

    @classmethod
    def from_shared(
        cls,
        payload: LimitationDecision,
        *,
        scenario_label: str | None = None,
    ) -> "ExpectedLimitationDecision":
        return cls(
            expected_limitation_decision_id=payload.limitation_decision_id,
            shared_limitation_decision=payload,
            expected_status=payload.status,
            expected_action=payload.resolution.action,
            scenario_label=scenario_label,
        )


# Shared A types are re-exported only as aliases for ergonomic B imports.  No
# B subclass or duplicate schema is created for any of these names.
AExpectedCustomerAttribution = AExpectedCustomerAttribution


__all__ = (
    "AExpectedCustomerAttribution",
    "BDatasetExecutionContext",
    "B_MIGRATION_FROM",
    "B_SCHEMA_VERSION",
    "CustomerAttribution",
    "ExpectedFieldAnnotation",
    "ExpectedLifecycleAtResult",
    "ExpectedLimitationDecision",
    "ExecutionNamespaceKey",
    "FieldAnnotation",
    "GoldCustomerLocator",
    "GoldDocumentFact",
    "GoldDocumentLocator",
    "LifecycleAtResult",
    "LimitationDecision",
    "RuntimeCustomerLocator",
    "RuntimeDocumentLocator",
    "RuntimeKnowledgeFact",
    "Sha256",
    "SourceDocumentMetadata",
    "TechnicalCitation",
)
