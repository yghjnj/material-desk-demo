"""Execution namespaces, sidecars, manifests, and freeze receipts."""

from __future__ import annotations

import hashlib
import json
from typing import Literal
from uuid import UUID

from pydantic import Field, model_validator

from .base import CONTRACT_VERSION, NonEmptyString, PositiveInt, Sha256, UtcDatetime, VersionedContract
from .enums import DatasetSplit, EnvironmentNamespace, FreezeStatus, RunStatus


def _canonical_hash(payload: dict[str, str]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


class ExecutionNamespaceKey(VersionedContract):
    environment: EnvironmentNamespace
    corpus_manifest_hash: Sha256
    split_manifest_hash: Sha256
    document_version_set_hash: Sha256
    source_hash_set_hash: Sha256
    contract_bundle_hash: Sha256
    configuration_hash: Sha256
    code_hash: Sha256
    run_id: UUID
    namespace_hash: Sha256

    @model_validator(mode="after")
    def verify_namespace_hash(self) -> "ExecutionNamespaceKey":
        payload = {
            "environment": self.environment.value,
            "corpus_manifest_hash": self.corpus_manifest_hash,
            "split_manifest_hash": self.split_manifest_hash,
            "document_version_set_hash": self.document_version_set_hash,
            "source_hash_set_hash": self.source_hash_set_hash,
            "contract_bundle_hash": self.contract_bundle_hash,
            "configuration_hash": self.configuration_hash,
            "code_hash": self.code_hash,
            "run_id": str(self.run_id),
        }
        if _canonical_hash(payload) != self.namespace_hash:
            raise ValueError("namespace_hash does not match compound namespace")
        return self


class IndexBuildReceipt(VersionedContract):
    index_snapshot_id: UUID
    source_hash_set_hash: Sha256
    build_config_hash: Sha256
    code_hash: Sha256
    query_input_count: Literal[0] = 0
    gold_input_count: Literal[0] = 0
    label_input_count: Literal[0] = 0
    build_log_hash: Sha256
    verification_status: Literal["VERIFIED"] = "VERIFIED"


class PartialRunSidecar(VersionedContract):
    run_id: UUID
    attempt_id: UUID
    retry_of_attempt_id: UUID | None = None
    retry_of_run_id: UUID | None = None
    idempotency_key_hash: Sha256
    sidecar_sequence: int = Field(ge=1)
    previous_sidecar_sha256: Sha256 | None = None
    sidecar_sha256: Sha256
    input_manifest_hash: Sha256
    input_hash: Sha256
    configuration_hash: Sha256
    contract_versions: dict[str, str]
    code_version: NonEmptyString
    code_hash: Sha256
    config_versions: dict[str, str]
    completed_stage_ids: tuple[NonEmptyString, ...]
    pending_stage_ids: tuple[NonEmptyString, ...]
    failed_stage_ids: tuple[NonEmptyString, ...]
    output_refs: tuple[NonEmptyString, ...]
    output_hashes: tuple[Sha256, ...]
    error_codes: tuple[NonEmptyString, ...]
    status: RunStatus
    created_at: UtcDatetime

    @model_validator(mode="after")
    def stage_partition(self) -> "PartialRunSidecar":
        groups = [set(self.completed_stage_ids), set(self.pending_stage_ids), set(self.failed_stage_ids)]
        if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError("sidecar stage groups must be disjoint")
        if self.status == RunStatus.COMPLETE and (self.pending_stage_ids or self.failed_stage_ids):
            raise ValueError("COMPLETE sidecar cannot have pending or failed stages")
        if self.status == RunStatus.FAILED and not self.failed_stage_ids:
            raise ValueError("FAILED sidecar requires a failed stage")
        if self.status == RunStatus.PARTIAL and not (self.pending_stage_ids or self.failed_stage_ids):
            raise ValueError("PARTIAL sidecar requires unfinished work")
        if len(self.output_refs) != len(self.output_hashes):
            raise ValueError("sidecar output references and hashes must pair exactly")
        if self.retry_of_attempt_id == self.attempt_id:
            raise ValueError("retry_of_attempt_id cannot reference the current attempt")
        if self.previous_sidecar_sha256 == self.sidecar_sha256:
            raise ValueError("a sidecar cannot name itself as its previous sidecar")
        if self.sidecar_sequence == 1:
            if (
                self.previous_sidecar_sha256 is not None
                or self.retry_of_attempt_id is not None
                or self.retry_of_run_id is not None
            ):
                raise ValueError("first sidecar cannot carry retry lineage")
        else:
            if (
                self.previous_sidecar_sha256 is None
                or self.retry_of_attempt_id is None
                or self.retry_of_run_id is None
            ):
                raise ValueError("subsequent sidecar requires complete retry lineage")
            if self.retry_of_run_id != self.run_id:
                raise ValueError("retry lineage cannot cross run IDs")
        if self.status == RunStatus.COMPLETE and self.error_codes:
            raise ValueError("COMPLETE sidecar cannot carry error codes")
        if self.status == RunStatus.FAILED and not self.error_codes:
            raise ValueError("FAILED sidecar requires an error code")
        return self


class PerCaseOutput(VersionedContract):
    root_scenario_id: UUID
    case_id: UUID
    task_instance_id: UUID
    input_hash: Sha256
    output_id: UUID
    output_hash: Sha256
    outcome: NonEmptyString
    latency_ms: int = Field(ge=0)
    reason_codes: tuple[NonEmptyString, ...]


class RunManifest(VersionedContract):
    run_id: UUID
    selected_attempt_id: UUID
    run_manifest_version: NonEmptyString
    run_manifest_sha256: Sha256
    namespace: ExecutionNamespaceKey
    split: DatasetSplit
    input_manifest_hash: Sha256
    corpus_manifest_hash: Sha256
    dataset_hash: Sha256
    split_manifest_hash: Sha256
    document_version_set_hash: Sha256
    source_hash_set_hash: Sha256
    contract_bundle_hash: Sha256
    contract_versions: dict[str, str]
    code_version: NonEmptyString
    code_hash: Sha256
    configuration_hash: Sha256
    parser_version: NonEmptyString
    chunking_version: NonEmptyString
    index_version: NonEmptyString
    model_version: NonEmptyString
    embedding_version: NonEmptyString
    reranker_version: NonEmptyString
    prompt_versions: dict[str, str]
    threshold_version: NonEmptyString
    unit_catalog_version: NonEmptyString
    random_seed: int
    sidecar_hashes: tuple[Sha256, ...]
    expected_case_count: PositiveInt
    per_case_outputs: tuple[PerCaseOutput, ...]
    output_status: Literal["INCOMPLETE", "COMPLETE", "FAILED"]
    output_manifest_hash: Sha256 | None = None
    failure_case_ids: tuple[UUID, ...]
    limitation_decision_ids: tuple[UUID, ...]
    run_status: RunStatus
    started_at: UtcDatetime
    completed_at: UtcDatetime | None = None

    @model_validator(mode="after")
    def manifest_consistency(self) -> "RunManifest":
        bound_hashes = {
            "corpus_manifest_hash": self.corpus_manifest_hash,
            "split_manifest_hash": self.split_manifest_hash,
            "document_version_set_hash": self.document_version_set_hash,
            "source_hash_set_hash": self.source_hash_set_hash,
            "contract_bundle_hash": self.contract_bundle_hash,
            "configuration_hash": self.configuration_hash,
            "code_hash": self.code_hash,
        }
        if self.namespace.run_id != self.run_id:
            raise ValueError("run manifest and namespace run IDs differ")
        if self.contract_versions.get("A") != CONTRACT_VERSION:
            raise ValueError("run manifest must declare the current A contract version")
        if self.namespace.environment.value != self.split.value:
            raise ValueError("run split and namespace environment differ")
        for field_name, expected in bound_hashes.items():
            if getattr(self.namespace, field_name) != expected:
                raise ValueError(f"namespace {field_name} differs from run manifest")
        if self.completed_at is not None and self.completed_at < self.started_at:
            raise ValueError("run completion precedes start")
        if len(self.sidecar_hashes) != len(set(self.sidecar_hashes)):
            raise ValueError("sidecar hashes must be unique")
        task_ids = [item.task_instance_id for item in self.per_case_outputs]
        output_ids = [item.output_id for item in self.per_case_outputs]
        if len(task_ids) != len(set(task_ids)) or len(output_ids) != len(set(output_ids)):
            raise ValueError("per-case task and output IDs must be unique")
        output_case_ids = {item.case_id for item in self.per_case_outputs}
        if not set(self.failure_case_ids).issubset(output_case_ids):
            raise ValueError("failure cases must reference a per-case output")

        if self.run_status is RunStatus.COMPLETE:
            if self.completed_at is None or self.output_status != "COMPLETE":
                raise ValueError("COMPLETE run requires completed time and output status")
            if self.output_manifest_hash is None or not self.sidecar_hashes:
                raise ValueError("COMPLETE run requires sidecars and output manifest hash")
            if len(self.per_case_outputs) != self.expected_case_count:
                raise ValueError("COMPLETE run output count must match expected_case_count")
        elif self.run_status is RunStatus.PARTIAL:
            if (
                self.completed_at is not None
                or self.output_status != "INCOMPLETE"
                or self.output_manifest_hash is not None
            ):
                raise ValueError("PARTIAL run cannot claim complete output")
        else:
            if self.output_status != "FAILED" or self.completed_at is None:
                raise ValueError("FAILED/INVALID run requires failed output and end time")
        return self


class DevelopmentFreezeReceipt(VersionedContract):
    receipt_id: UUID
    component: NonEmptyString
    code_hash: Sha256
    content_hash: Sha256
    configuration_hash: Sha256
    contract_versions: dict[str, str]
    contract_bundle_hash: Sha256
    data_manifest_hash: Sha256
    target_threshold_version: NonEmptyString
    target_threshold_snapshot_hash: Sha256
    development_run_manifest_hash: Sha256
    zero_tolerance_failures: int = Field(ge=0)
    open_issues: tuple[NonEmptyString, ...]
    approved_by: tuple[UUID, ...] = Field(min_length=2)
    approved_at: UtcDatetime
    status: FreezeStatus
    invalidated_at: UtcDatetime | None = None
    invalidation_reason: NonEmptyString | None = None

    @model_validator(mode="after")
    def freeze_shape(self) -> "DevelopmentFreezeReceipt":
        if self.status == FreezeStatus.FROZEN and (self.zero_tolerance_failures or self.open_issues):
            raise ValueError("frozen receipt cannot have failures or open issues")
        if self.status == FreezeStatus.INVALIDATED and (
            self.invalidated_at is None or self.invalidation_reason is None
        ):
            raise ValueError("invalidated receipt requires audit fields")
        if self.status != FreezeStatus.INVALIDATED and (
            self.invalidated_at is not None or self.invalidation_reason is not None
        ):
            raise ValueError("only INVALIDATED receipt may carry invalidation fields")
        if len(set(self.approved_by)) != len(self.approved_by):
            raise ValueError("freeze receipt approvers must be distinct")
        return self


class DevelopmentUnfreezeEvent(VersionedContract):
    event_id: UUID
    receipt_id: UUID
    reason_code: Literal[
        "SECURITY_DEFECT",
        "CONTRACT_CHANGE",
        "DATA_CHANGE",
        "CODE_CHANGE",
        "CONFIG_CHANGE",
        "MODEL_OR_PROMPT_CHANGE",
        "INDEX_CHANGE",
        "UNIT_RULE_CHANGE",
        "REPRODUCIBILITY_FAILURE",
        "HOLDOUT_INVALIDATION",
    ]
    requested_by: UUID
    approved_by: UUID
    affected_receipt_ids: tuple[UUID, ...]
    affected_run_ids: tuple[UUID, ...]
    occurred_at: UtcDatetime
    audit_notes: NonEmptyString
