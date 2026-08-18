"""Evaluation protocol and immutable, derivable formal results."""

from __future__ import annotations

from decimal import Decimal
from typing import Literal
from uuid import UUID

from pydantic import model_validator

from .base import (
    DecimalString,
    NonEmptyString,
    NonNegativeInt,
    Sha256,
    UtcDatetime,
    VersionedContract,
)
from .enums import DatasetSplit, MeasurementStatus, ThresholdStatus, Verdict


def _threshold_satisfied(
    value: Decimal,
    operator: Literal[">=", "<=", "="],
    threshold: Decimal,
) -> bool:
    if operator == ">=":
        return value >= threshold
    if operator == "<=":
        return value <= threshold
    return value == threshold


class MetricDefinition(VersionedContract):
    metric_id: NonEmptyString
    metric_kind: Literal["RATIO", "COUNT"] = "RATIO"
    numerator_definition: NonEmptyString
    denominator_definition: NonEmptyString
    exclusion_reason_codes: tuple[NonEmptyString, ...]
    operator: Literal[">=", "<=", "="]
    threshold: DecimalString
    threshold_status: ThresholdStatus = ThresholdStatus.SET
    scope: NonEmptyString
    confidence_interval_method: NonEmptyString
    critical: bool = True


class EvaluationProtocol(VersionedContract):
    protocol_id: UUID
    protocol_version: NonEmptyString
    protocol_sha256: Sha256
    dataset_id: UUID
    dataset_version: NonEmptyString
    threshold_registry_version: NonEmptyString
    development_policy: NonEmptyString
    sealed_holdout_policy: NonEmptyString
    metrics: tuple[MetricDefinition, ...]
    reason_codes: tuple[NonEmptyString, ...]
    failure_case_ids: tuple[UUID, ...] = ()
    evaluator_version: NonEmptyString
    run_status: Literal["NOT_RUN", "VALID", "INVALID"]

    @model_validator(mode="after")
    def metric_registry_is_unique(self) -> "EvaluationProtocol":
        if not self.metrics:
            raise ValueError("evaluation protocol requires at least one metric")
        metric_ids = [metric.metric_id for metric in self.metrics]
        if len(metric_ids) != len(set(metric_ids)):
            raise ValueError("evaluation protocol metric IDs must be unique")
        return self


class MetricResult(VersionedContract):
    metric_id: NonEmptyString
    metric_kind: Literal["RATIO", "COUNT"] = "RATIO"
    numerator: NonNegativeInt
    denominator: NonNegativeInt
    excluded_count: NonNegativeInt
    exclusion_reason_counts: dict[NonEmptyString, NonNegativeInt]
    value: DecimalString | None = None
    operator: Literal[">=", "<=", "="]
    threshold: DecimalString
    threshold_status: ThresholdStatus = ThresholdStatus.SET
    threshold_met: bool | None = None
    subclass_counts: dict[NonEmptyString, NonNegativeInt]
    confidence_interval_low: DecimalString | None = None
    confidence_interval_high: DecimalString | None = None

    @model_validator(mode="after")
    def count_and_threshold_semantics(self) -> "MetricResult":
        if sum(self.exclusion_reason_counts.values()) != self.excluded_count:
            raise ValueError("excluded_count must equal exclusion reason counts")
        if self.subclass_counts and sum(self.subclass_counts.values()) != self.denominator:
            raise ValueError("subclass counts must partition the denominator")
        if (self.confidence_interval_low is None) != (
            self.confidence_interval_high is None
        ):
            raise ValueError("confidence interval bounds must be supplied together")

        if self.metric_kind == "RATIO":
            if self.numerator > self.denominator:
                raise ValueError("ratio numerator cannot exceed denominator")
            if self.denominator == 0:
                if self.numerator != 0 or self.value is not None or self.threshold_met is not None:
                    raise ValueError("zero-denominator ratio must be NOT_APPLICABLE")
            else:
                expected_value = Decimal(self.numerator) / Decimal(self.denominator)
                if self.value is None or Decimal(self.value) != expected_value:
                    raise ValueError("ratio value must exactly equal numerator / denominator")
                expected_gate = _threshold_satisfied(
                    expected_value, self.operator, Decimal(self.threshold)
                )
                if self.threshold_met is not expected_gate:
                    raise ValueError("threshold_met does not match the ratio threshold")
        else:
            if self.denominator != 0:
                raise ValueError("COUNT metrics use numerator as value and denominator=0")
            expected_value = Decimal(self.numerator)
            if self.value is None or Decimal(self.value) != expected_value:
                raise ValueError("count value must exactly equal numerator")
            expected_gate = _threshold_satisfied(
                expected_value, self.operator, Decimal(self.threshold)
            )
            if self.threshold_met is not expected_gate:
                raise ValueError("threshold_met does not match the count threshold")

        if self.confidence_interval_low is not None:
            low = Decimal(self.confidence_interval_low)
            high = Decimal(self.confidence_interval_high)
            if low > high:
                raise ValueError("confidence interval lower bound exceeds upper bound")
            if self.value is not None and not low <= Decimal(self.value) <= high:
                raise ValueError("confidence interval must contain the metric value")
            if self.metric_kind == "RATIO" and (low < 0 or high > 1):
                raise ValueError("ratio confidence interval must stay within [0, 1]")
        return self


class EvaluationResult(VersionedContract):
    evaluation_id: UUID
    protocol_id: UUID
    protocol_version: NonEmptyString
    protocol_sha256: Sha256
    run_manifest_sha256: Sha256 | None = None
    split: DatasetSplit
    measurement_status: MeasurementStatus
    required_metric_ids: tuple[NonEmptyString, ...] = ()
    critical_metric_ids: tuple[NonEmptyString, ...] = ()
    metrics: tuple[MetricResult, ...] = ()
    verdict: Verdict
    failure_case_ids: tuple[UUID, ...]
    limitations: tuple[NonEmptyString, ...]
    evaluated_at: UtcDatetime | None = None

    @model_validator(mode="after")
    def formal_gate(self) -> "EvaluationResult":
        if self.measurement_status in {MeasurementStatus.PLANNED, MeasurementStatus.NOT_RUN}:
            if (
                self.verdict is not Verdict.NOT_RUN
                or self.run_manifest_sha256 is not None
                or self.evaluated_at is not None
                or self.metrics
                or self.required_metric_ids
                or self.critical_metric_ids
            ):
                raise ValueError("unrun evaluation cannot carry metrics, run evidence, or verdict")
            return self

        if self.measurement_status is MeasurementStatus.INVALID:
            if self.verdict is not Verdict.FAIL:
                raise ValueError("INVALID evaluation must have FAIL verdict")
            if self.metrics or self.required_metric_ids or self.critical_metric_ids:
                raise ValueError("INVALID evaluation cannot publish formal metric percentages")
            if self.evaluated_at is None or not self.limitations:
                raise ValueError("INVALID evaluation requires time and failure limitation")
            return self

        if self.split is not DatasetSplit.SEALED_HOLDOUT:
            raise ValueError("formal measured verdict requires SEALED_HOLDOUT")
        if self.run_manifest_sha256 is None or self.evaluated_at is None:
            raise ValueError("measured evaluation requires run manifest and time")
        if not self.metrics or not self.required_metric_ids:
            raise ValueError("MEASURED evaluation requires a complete non-empty metric set")

        metric_ids = [metric.metric_id for metric in self.metrics]
        if len(metric_ids) != len(set(metric_ids)):
            raise ValueError("metric results must have unique IDs")
        if set(metric_ids) != set(self.required_metric_ids) or len(metric_ids) != len(
            self.required_metric_ids
        ):
            raise ValueError("metric results must exactly match required_metric_ids")
        if not set(self.critical_metric_ids).issubset(self.required_metric_ids):
            raise ValueError("critical metrics must be part of the required metric set")
        if any(metric.threshold_met is None for metric in self.metrics):
            raise ValueError("measured evaluation cannot include non-applicable metrics")

        failed = {metric.metric_id for metric in self.metrics if metric.threshold_met is False}
        if not failed:
            expected_verdict = Verdict.PASS
        elif failed & set(self.critical_metric_ids):
            expected_verdict = Verdict.FAIL
        else:
            expected_verdict = Verdict.CONDITIONAL_PASS
        if self.verdict is not expected_verdict:
            raise ValueError("verdict does not match complete threshold results")
        if expected_verdict is Verdict.PASS and self.failure_case_ids:
            raise ValueError("PASS evaluation cannot carry failure_case_ids")
        if expected_verdict is not Verdict.PASS and not self.failure_case_ids:
            raise ValueError("non-PASS measured result requires failure_case_ids")
        return self
