"""Customer requirement annotations and aggregate."""

from __future__ import annotations

from uuid import UUID

from pydantic import Field, model_validator

from .base import JsonPointer, NonEmptyString, Provenance, Sha256, UtcDatetime, VersionedContract
from .enums import (
    AttributionStatus,
    FieldStatus,
    NormalizationStatus,
    RequirementStatus,
    ValueState,
)
from .evidence import CustomerAttribution
from .measurements import Measurement, ValuePayload


class FieldAnnotation(VersionedContract):
    annotation_id: UUID
    field_path: JsonPointer
    status: FieldStatus
    value: ValuePayload
    measurement: Measurement | None = None
    source_locator_ids: tuple[UUID, ...] = ()
    customer_attribution_ids: tuple[UUID, ...] = ()
    normalization_status: NormalizationStatus
    derivation_rule_version: str | None = None
    input_annotation_ids: tuple[UUID, ...] = ()
    notes: tuple[str, ...] = ()
    supersedes_annotation_id: UUID | None = None
    superseded_by_annotation_id: UUID | None = None

    @model_validator(mode="after")
    def validate_state_matrix(self) -> "FieldAnnotation":
        state = self.value.value_state
        if len(self.source_locator_ids) != len(set(self.source_locator_ids)):
            raise ValueError("field source locator IDs must be unique")
        if len(self.customer_attribution_ids) != len(set(self.customer_attribution_ids)):
            raise ValueError("field customer attribution IDs must be unique")
        if self.measurement is not None and (
            self.measurement.normalization_status is not self.normalization_status
        ):
            raise ValueError("field and measurement normalization status must match")
        if self.status == FieldStatus.MISSING:
            if state != ValueState.UNSET or self.source_locator_ids or self.customer_attribution_ids:
                raise ValueError("MISSING requires UNSET and no source")
            if self.measurement or self.derivation_rule_version or self.input_annotation_ids:
                raise ValueError("MISSING cannot carry value or derivation")
        if self.status == FieldStatus.EXPLICIT:
            if state == ValueState.UNSET or not self.source_locator_ids or not self.customer_attribution_ids:
                raise ValueError("EXPLICIT requires customer evidence and a non-UNSET value")
            if self.derivation_rule_version or self.input_annotation_ids:
                raise ValueError("unit normalization does not make EXPLICIT derived")
        if self.status == FieldStatus.DERIVED:
            if state not in {ValueState.KNOWN, ValueState.NOT_APPLICABLE, ValueState.UNRESOLVED}:
                raise ValueError("DERIVED has an invalid value state")
            if not self.derivation_rule_version or len(self.input_annotation_ids) < 2:
                raise ValueError("DERIVED requires a rule and at least two explicit inputs")
            if len(set(self.input_annotation_ids)) != len(self.input_annotation_ids):
                raise ValueError("DERIVED input annotations must be distinct")
            if self.source_locator_ids or self.customer_attribution_ids:
                raise ValueError("DERIVED cannot invent direct customer evidence")
        if self.status in {FieldStatus.AMBIGUOUS, FieldStatus.CONFLICTING} and state != ValueState.UNRESOLVED:
            raise ValueError("ambiguous/conflicting fields must be UNRESOLVED")
        if self.status == FieldStatus.SUPERSEDED and self.superseded_by_annotation_id is None:
            raise ValueError("SUPERSEDED requires a replacement annotation")
        return self


class CustomerRequirement(VersionedContract):
    requirement_id: UUID
    revision: int = Field(ge=1)
    case_id: UUID
    source_message_ids: tuple[UUID, ...] = Field(min_length=1)
    field_annotations: tuple[FieldAnnotation, ...] = Field(min_length=1)
    customer_attributions: tuple[CustomerAttribution, ...]
    extraction_status: RequirementStatus
    blocking_field_paths: tuple[JsonPointer, ...] = ()
    input_snapshot_hash: Sha256
    provenance: Provenance
    extracted_at: UtcDatetime

    @model_validator(mode="after")
    def validate_graph(self) -> "CustomerRequirement":
        annotation_ids = [item.annotation_id for item in self.field_annotations]
        if len(annotation_ids) != len(set(annotation_ids)):
            raise ValueError("field annotation IDs must be unique")
        annotations = {item.annotation_id: item for item in self.field_annotations}
        if len(self.source_message_ids) != len(set(self.source_message_ids)):
            raise ValueError("source message IDs must be unique")
        paths = [item.field_path for item in self.field_annotations if item.status != FieldStatus.SUPERSEDED]
        if len(paths) != len(set(paths)):
            raise ValueError("active field paths must be unique")
        attributions = {
            item.customer_attribution_id: item for item in self.customer_attributions
        }
        if len(attributions) != len(self.customer_attributions):
            raise ValueError("customer attribution IDs must be unique")
        if any(item.case_id != self.case_id for item in self.customer_attributions):
            raise ValueError("customer attribution belongs to another case")
        if any(
            item.message_id not in self.source_message_ids
            for item in self.customer_attributions
        ):
            raise ValueError("customer attribution uses an undeclared source message")
        if any(
            item.attribution_status is not AttributionStatus.EXACT_MATCH
            for item in self.customer_attributions
        ):
            raise ValueError("requirement accepts only EXACT_MATCH customer attribution")

        referenced_attribution_ids: set[UUID] = set()
        for annotation in self.field_annotations:
            if annotation.status is FieldStatus.DERIVED:
                inputs = [annotations.get(item) for item in annotation.input_annotation_ids]
                if any(item is None for item in inputs) or any(
                    item.status is not FieldStatus.EXPLICIT
                    for item in inputs
                    if item is not None
                ):
                    raise ValueError("DERIVED inputs must reference existing EXPLICIT annotations")
            annotation_attributions: list[CustomerAttribution] = []
            for attribution_id in annotation.customer_attribution_ids:
                attribution = attributions.get(attribution_id)
                if attribution is None:
                    raise ValueError("field annotation has an unknown customer attribution")
                if attribution.field_path != annotation.field_path:
                    raise ValueError("customer attribution field_path does not match annotation")
                annotation_attributions.append(attribution)
                referenced_attribution_ids.add(attribution_id)
            locator_ids = {
                item.runtime_customer_locator.runtime_customer_locator_id
                for item in annotation_attributions
            }
            if locator_ids != set(annotation.source_locator_ids):
                raise ValueError("field source locators must exactly match attribution locators")
        if referenced_attribution_ids != set(attributions):
            raise ValueError("every requirement attribution must support exactly declared fields")
        valid_paths = set(paths)
        if not set(self.blocking_field_paths).issubset(valid_paths):
            raise ValueError("blocking field path is not an active annotation")
        blocking_allowed = {
            item.field_path
            for item in self.field_annotations
            if item.status in {
                FieldStatus.MISSING,
                FieldStatus.AMBIGUOUS,
                FieldStatus.CONFLICTING,
            }
        }
        if not set(self.blocking_field_paths).issubset(blocking_allowed):
            raise ValueError("only unresolved fields may block requirement completion")
        return self

    @property
    def missing_field_paths(self) -> list[str]:
        return [str(item.field_path) for item in self.field_annotations if item.status == FieldStatus.MISSING]
