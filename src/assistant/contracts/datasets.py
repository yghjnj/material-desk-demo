"""Evaluation dataset, root scenario, case graph, and development fixtures."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import Field, model_validator

from .base import NonEmptyString, Sha256, UtcDatetime, VersionedContract
from .document_locators import GoldDocumentLocator
from .enums import DatasetReleaseStatus, DatasetSplit, TaskType

PLANNED_ROOT_SCENARIO_COUNT = 60
PLANNED_TASK_INSTANCE_COUNT = 180
PLANNED_DEVELOPMENT_ROOT_COUNT = 36
PLANNED_HOLDOUT_ROOT_COUNT = 24
PLANNED_DEVELOPMENT_TASK_COUNT = 108
PLANNED_HOLDOUT_TASK_COUNT = 72
PLANNED_END_TO_END_CHAIN_COUNT = 30
# Frozen planning allocation only; it is not an observed dataset or measured result.
PLANNED_TASK_TYPE_COUNTS: dict[TaskType, int] = {
    TaskType.QA: 60,
    TaskType.REQUIREMENT_EXTRACTION: 40,
    TaskType.REPLY_DRAFT: 30,
    TaskType.REFUSAL: 20,
    TaskType.SECURITY: 30,
}


class DatasetPlanningCounts(VersionedContract):
    root_scenario_count: Literal[60] = PLANNED_ROOT_SCENARIO_COUNT
    task_instance_count: Literal[180] = PLANNED_TASK_INSTANCE_COUNT
    development_root_count: Literal[36] = PLANNED_DEVELOPMENT_ROOT_COUNT
    holdout_root_count: Literal[24] = PLANNED_HOLDOUT_ROOT_COUNT
    development_task_count: Literal[108] = PLANNED_DEVELOPMENT_TASK_COUNT
    holdout_task_count: Literal[72] = PLANNED_HOLDOUT_TASK_COUNT
    end_to_end_chain_count: Literal[30] = PLANNED_END_TO_END_CHAIN_COUNT
    end_to_end_chains_are_additive: Literal[False] = False
    task_type_counts: dict[TaskType, int] = Field(default_factory=lambda: dict(PLANNED_TASK_TYPE_COUNTS))

    @model_validator(mode="after")
    def counts(self) -> "DatasetPlanningCounts":
        if set(self.task_type_counts) != set(TaskType) or sum(self.task_type_counts.values()) != 180:
            raise ValueError("five task type planning counts must sum to 180")
        return self


class TaskInstance(VersionedContract):
    task_instance_id: UUID
    task_type: TaskType
    split: DatasetSplit
    eligible: bool = True
    exclusion_reason_code: str | None = None

    @model_validator(mode="after")
    def eligibility_reason(self) -> "TaskInstance":
        if self.eligible == (self.exclusion_reason_code is not None):
            raise ValueError("only ineligible task instances require exclusion reason")
        return self


class RootScenario(VersionedContract):
    root_scenario_id: UUID
    scenario_version: NonEmptyString
    split: DatasetSplit
    fact_family_ids: tuple[UUID, ...] = Field(min_length=1)
    source_lineage_ids: tuple[UUID, ...] = Field(min_length=1)
    template_family_id: UUID
    task_labels: frozenset[TaskType] = Field(min_length=1)
    case_graph_ids: tuple[UUID, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_groups(self) -> "RootScenario":
        if len(self.fact_family_ids) != len(set(self.fact_family_ids)):
            raise ValueError("fact family IDs must be unique")
        if len(self.source_lineage_ids) != len(set(self.source_lineage_ids)):
            raise ValueError("source lineage IDs must be unique")
        return self


class CaseGraph(VersionedContract):
    case_graph_id: UUID
    root_scenario_id: UUID
    case_id: UUID
    fact_family_ids: tuple[UUID, ...] = Field(min_length=1)
    source_lineage_ids: tuple[UUID, ...] = Field(min_length=1)
    template_family_id: UUID
    message_ids: tuple[UUID, ...] = Field(min_length=1)
    expected_requirement_id: UUID | None = None
    expected_retrieval_evidence_ids: tuple[UUID, ...] = ()
    expected_qa_response_id: UUID | None = None
    expected_reply_draft_id: UUID | None = None
    expected_refusal_labels: tuple[NonEmptyString, ...] = ()
    expected_security_labels: tuple[NonEmptyString, ...] = ()
    task_instances: tuple[TaskInstance, ...] = Field(min_length=1)
    split: DatasetSplit

    @model_validator(mode="after")
    def atomic_split(self) -> "CaseGraph":
        if any(item.split != self.split for item in self.task_instances):
            raise ValueError("root scenario task instances cannot cross split")
        return self


class AssetManifestEntry(VersionedContract):
    asset_id: UUID
    logical_ref: NonEmptyString
    asset_type: NonEmptyString
    sha256: Sha256
    size_bytes: int = Field(ge=0)
    media_type: NonEmptyString
    license: NonEmptyString
    provenance: NonEmptyString
    data_classification: NonEmptyString
    split: DatasetSplit | None = None
    fact_family_id: UUID | None = None
    source_lineage_id: UUID | None = None
    template_family_id: UUID | None = None


class LeakageReport(VersionedContract):
    report_id: UUID
    root_scenario_overlap: int = Field(default=0, ge=0)
    fact_family_overlap: int = Field(default=0, ge=0)
    source_lineage_overlap: int = Field(default=0, ge=0)
    template_family_overlap: int = Field(default=0, ge=0)
    unresolved_near_duplicates: int = Field(default=0, ge=0)
    status: Literal["PASS", "FAIL", "NOT_RUN"]

    @model_validator(mode="after")
    def zero_for_pass(self) -> "LeakageReport":
        total = (
            self.root_scenario_overlap
            + self.fact_family_overlap
            + self.source_lineage_overlap
            + self.template_family_overlap
            + self.unresolved_near_duplicates
        )
        if self.status == "PASS" and total:
            raise ValueError("leakage report cannot pass with unresolved overlap")
        return self


class EvaluationDatasetManifest(VersionedContract):
    dataset_id: UUID
    dataset_version: NonEmptyString
    dataset_sha256: Sha256 | None = None
    hash_status: Literal["NOT_COMPUTED", "COMPUTED"]
    annotation_version: NonEmptyString
    release_status: DatasetReleaseStatus
    assets: tuple[AssetManifestEntry, ...]
    root_scenarios: tuple[RootScenario, ...]
    case_graphs: tuple[CaseGraph, ...]
    planned_root_scenario_count: Literal[60] = 60
    planned_task_instance_count: Literal[180] = 180
    planned_development_root_count: Literal[36] = 36
    planned_holdout_root_count: Literal[24] = 24
    planned_development_task_count: Literal[108] = 108
    planned_holdout_task_count: Literal[72] = 72
    planned_task_counts_by_type: dict[TaskType, int]
    planned_end_to_end_chain_count: Literal[30] = 30
    end_to_end_chains_are_additive: Literal[False] = False
    leakage_report: LeakageReport
    created_at: UtcDatetime

    @model_validator(mode="after")
    def planned_counts(self) -> "EvaluationDatasetManifest":
        if set(self.planned_task_counts_by_type) != set(TaskType):
            raise ValueError("planned task counts must cover exactly five task types")
        if self.planned_task_counts_by_type != PLANNED_TASK_TYPE_COUNTS:
            raise ValueError("planned task counts must preserve the frozen B allocation")
        if self.hash_status == "COMPUTED" and self.dataset_sha256 is None:
            raise ValueError("computed dataset requires a real hash")
        if self.hash_status == "NOT_COMPUTED" and self.dataset_sha256 is not None:
            raise ValueError("unbuilt dataset cannot carry a hash")
        if self.release_status == DatasetReleaseStatus.FROZEN:
            if not self.assets or not self.root_scenarios or not self.case_graphs:
                raise ValueError("frozen dataset requires real assets and scenarios")
            if self.hash_status != "COMPUTED" or self.leakage_report.status != "PASS":
                raise ValueError("frozen dataset requires computed hash and clean leakage")
            self._validate_frozen_objects()
        return self

    def _validate_frozen_objects(self) -> None:
        roots = {item.root_scenario_id: item for item in self.root_scenarios}
        graphs = {item.case_graph_id: item for item in self.case_graphs}
        if len(roots) != len(self.root_scenarios):
            raise ValueError("frozen root_scenario_id values must be unique")
        if len(graphs) != len(self.case_graphs):
            raise ValueError("frozen case_graph_id values must be unique")
        asset_ids = [item.asset_id for item in self.assets]
        if len(asset_ids) != len(set(asset_ids)):
            raise ValueError("frozen asset IDs must be unique")
        case_ids = [item.case_id for item in self.case_graphs]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("frozen case IDs must be unique")

        referenced_graph_ids: list[UUID] = []
        all_tasks: list[TaskInstance] = []
        message_split: dict[UUID, DatasetSplit] = {}
        expected_reference_ids: dict[str, set[UUID]] = {
            "requirement": set(),
            "retrieval": set(),
            "qa": set(),
            "draft": set(),
        }
        for root in self.root_scenarios:
            if len(root.case_graph_ids) != len(set(root.case_graph_ids)):
                raise ValueError("root case graph references must be unique")
            referenced_graph_ids.extend(root.case_graph_ids)
            for graph_id in root.case_graph_ids:
                graph = graphs.get(graph_id)
                if graph is None:
                    raise ValueError("root references a missing case graph")
                if graph.root_scenario_id != root.root_scenario_id:
                    raise ValueError("case graph references the wrong root scenario")
                if graph.split != root.split:
                    raise ValueError("root scenario and case graph split must match")
                if (
                    set(graph.fact_family_ids) != set(root.fact_family_ids)
                    or set(graph.source_lineage_ids) != set(root.source_lineage_ids)
                    or graph.template_family_id != root.template_family_id
                ):
                    raise ValueError("case graph lineage must exactly match its root")
                graph_task_types = {task.task_type for task in graph.task_instances}
                if graph_task_types != set(root.task_labels):
                    raise ValueError("root task labels must exactly match case graph task types")
                for message_id in graph.message_ids:
                    previous_split = message_split.setdefault(message_id, graph.split)
                    if previous_split is not graph.split:
                        raise ValueError("message IDs leak across development and holdout")
                if graph.expected_requirement_id is not None:
                    expected_reference_ids["requirement"].add(graph.expected_requirement_id)
                expected_reference_ids["retrieval"].update(graph.expected_retrieval_evidence_ids)
                if graph.expected_qa_response_id is not None:
                    expected_reference_ids["qa"].add(graph.expected_qa_response_id)
                if graph.expected_reply_draft_id is not None:
                    expected_reference_ids["draft"].add(graph.expected_reply_draft_id)
                all_tasks.extend(graph.task_instances)

        if len(referenced_graph_ids) != len(set(referenced_graph_ids)):
            raise ValueError("a case graph cannot belong to multiple roots")
        if set(referenced_graph_ids) != set(graphs):
            raise ValueError("every frozen case graph must be referenced by one root")

        task_ids = [item.task_instance_id for item in all_tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("frozen task_instance_id values must be unique")
        if len(self.root_scenarios) != PLANNED_ROOT_SCENARIO_COUNT:
            raise ValueError("FROZEN requires exactly 60 actual root scenarios")
        if len(all_tasks) != PLANNED_TASK_INSTANCE_COUNT:
            raise ValueError("FROZEN requires exactly 180 actual task instances")

        # Every expected artifact/evidence ID is a graph-level reference.  A
        # duplicate would make one gold output ambiguous and would invalidate
        # per-task metric denominators.
        for kind, values in expected_reference_ids.items():
            expected_count = sum(
                1
                if (
                    (kind == "requirement" and graph.expected_requirement_id is not None)
                    or (kind == "qa" and graph.expected_qa_response_id is not None)
                    or (kind == "draft" and graph.expected_reply_draft_id is not None)
                )
                else len(graph.expected_retrieval_evidence_ids)
                if kind == "retrieval"
                else 0
                for graph in self.case_graphs
            )
            if len(values) != expected_count:
                raise ValueError(f"frozen {kind} reference IDs must be unique")

        root_split_counts = {
            split: sum(root.split is split for root in self.root_scenarios)
            for split in DatasetSplit
        }
        if root_split_counts != {
            DatasetSplit.DEVELOPMENT: PLANNED_DEVELOPMENT_ROOT_COUNT,
            DatasetSplit.SEALED_HOLDOUT: PLANNED_HOLDOUT_ROOT_COUNT,
        }:
            raise ValueError("actual frozen root split must be 36/24")
        task_split_counts = {
            split: sum(task.split is split for task in all_tasks) for split in DatasetSplit
        }
        if task_split_counts != {
            DatasetSplit.DEVELOPMENT: PLANNED_DEVELOPMENT_TASK_COUNT,
            DatasetSplit.SEALED_HOLDOUT: PLANNED_HOLDOUT_TASK_COUNT,
        }:
            raise ValueError("actual frozen task split must be 108/72")
        task_type_counts = {
            task_type: sum(task.task_type is task_type for task in all_tasks)
            for task_type in TaskType
        }
        if task_type_counts != PLANNED_TASK_TYPE_COUNTS:
            raise ValueError("actual frozen task types must preserve 60/40/30/20/30")

        end_to_end_count = sum(
            graph.expected_requirement_id is not None
            and bool(graph.expected_retrieval_evidence_ids)
            and graph.expected_qa_response_id is not None
            and graph.expected_reply_draft_id is not None
            for graph in self.case_graphs
        )
        if end_to_end_count != PLANNED_END_TO_END_CHAIN_COUNT:
            raise ValueError("FROZEN requires exactly 30 non-additive end-to-end relationships")

        development = [
            root for root in self.root_scenarios if root.split is DatasetSplit.DEVELOPMENT
        ]
        holdout = [
            root for root in self.root_scenarios if root.split is DatasetSplit.SEALED_HOLDOUT
        ]
        grouped_fields = ("fact_family_ids", "source_lineage_ids")
        for field_name in grouped_fields:
            development_ids = {
                item for root in development for item in getattr(root, field_name)
            }
            holdout_ids = {item for root in holdout for item in getattr(root, field_name)}
            if development_ids & holdout_ids:
                raise ValueError(f"{field_name} leaks across development and holdout")
        if {root.template_family_id for root in development} & {
            root.template_family_id for root in holdout
        }:
            raise ValueError("template families leak across development and holdout")


class FixtureResult(VersionedContract):
    fixture_result_id: UUID
    document_version_id: UUID
    gold_document_locator: GoldDocumentLocator
    evidence_text: NonEmptyString
    relevance_label: NonEmptyString


class DeterministicRetrievalFixture(VersionedContract):
    fixture_id: UUID
    fixture_version: NonEmptyString
    fixture_sha256: Sha256
    provenance: NonEmptyString
    source_dataset_id: UUID
    usage: Literal["DEVELOPMENT_ONLY"] = "DEVELOPMENT_ONLY"
    query_id: UUID
    ordered_results: tuple[FixtureResult, ...]
