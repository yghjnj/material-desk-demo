"""Reproducible validation gates for the Module B synthetic dataset.

The validator is deliberately independent from the generator.  It never trusts
pre-populated PASS flags and it never publishes sealed examples.  A complete
view is assembled in memory from the development export and the pre-seal
holdout payload, validated against A's shared schemas, and reduced to aggregate
or blinded reports under ``work/B/reports``.

The cross-split content detector uses this frozen profile:

* Unicode NFKC, LF line endings and Unicode case-folding;
* UUIDs, SHA-256 values, timestamps, split labels, root ordinals and decimal
  runs are replaced by stable placeholders so bookkeeping cannot defeat the
  detector;
* whitespace and punctuation collapse to one space;
* exact equality is checked first, then character 5-gram Jaccard similarity;
* a score >= 0.90 is an unresolved near duplicate.

Shared knowledge documents are intentionally excluded from content leakage
comparison because the contract permits both splits to query the same public
corpus.  Customer messages and query/prompt/rationale fields are compared.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import fnmatch
import json
from pathlib import Path
import re
import unicodedata
from typing import Any, Iterable, Mapping, Sequence
from uuid import UUID
import xml.etree.ElementTree as ET
import zipfile

from pydantic import TypeAdapter, ValidationError

from assistant.contracts.customers import CustomerMessage
from assistant.contracts.datasets import (
    AssetManifestEntry,
    CaseGraph,
    EvaluationDatasetManifest,
    LeakageReport,
    PLANNED_DEVELOPMENT_ROOT_COUNT,
    PLANNED_DEVELOPMENT_TASK_COUNT,
    PLANNED_END_TO_END_CHAIN_COUNT,
    PLANNED_HOLDOUT_ROOT_COUNT,
    PLANNED_HOLDOUT_TASK_COUNT,
    PLANNED_ROOT_SCENARIO_COUNT,
    PLANNED_TASK_INSTANCE_COUNT,
    PLANNED_TASK_TYPE_COUNTS,
    RootScenario,
    TaskInstance,
)
from assistant.contracts.document_locators import (
    GoldDocumentLocator,
    GoldDocxLocator,
    GoldMarkdownLocator,
    GoldPdfLocator,
    GoldTextLocator,
)
from assistant.contracts.documents import (
    DocumentLifecycleEvent,
    LifecycleAtResult,
    SourceDocumentMetadata,
)
from assistant.contracts.enums import DatasetReleaseStatus, DatasetSplit, TaskType
from assistant.contracts.evidence import ExpectedCustomerAttribution
from assistant.contracts.facts import GoldDocumentFact
from assistant.contracts.runtime import ExecutionNamespaceKey

from .hashing import (
    canonical_json,
    deterministic_uuid,
    file_sha256,
    sha256_json,
    sha256_text,
    verify_execution_namespace_key,
)
from .models import (
    BDatasetExecutionContext,
    ExpectedFieldAnnotation,
    ExpectedLimitationDecision,
)


REPORT_VERSION = "1.0.0"
NEAR_DUPLICATE_THRESHOLD = 0.90
NEAR_DUPLICATE_SHINGLE_SIZE = 5
MIN_DUPLICATE_TEXT_LENGTH = 24
PDF_REQUIRED_FACT_KEYS = frozenset(
    {
        "solids_content",
        "viscosity",
        "loi",
        "ul94",
        "tensile_strength",
        "demo_limitation",
    }
)
TASK_TYPE_NAMES = tuple(item.value for item in TaskType)
PUBLIC_HOLDOUT_SUMMARY_KEYS = frozenset(
    {
        "dataset_id",
        "dataset_version",
        "schema_version",
        "manifest_version",
        "annotation_version",
        "migration_from",
        "split",
        "actual_counts",
        "planned_counts",
        "root_scenario_ids",
        "task_instance_ids",
        "gold_access",
        "payload_sha256",
        "sealed_payload_sha256",
        "seal_status",
        "manifest_sha256",
        "root_count",
        "task_count",
        "dataset_sha256",
        "full_manifest_sha256",
        "holdout_content_commitment_hash",
        "artifact_role",
        "content_state",
        "contains_holdout_gold",
        "blocking_dependency",
        "formal_metrics",
        "redaction_commitment_input",
        "redaction_commitment_sha256",
        "status",
    }
)
UUID_RE = re.compile(
    r"(?i)\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b"
)
SHA_RE = re.compile(r"(?i)\bsha256:[0-9a-f]{64}\b")
TIME_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z\b",
    re.IGNORECASE,
)
ROOT_MARKER_RE = re.compile(r"(?:root[ _-]*scenario|\u6839\u573a\u666f)\s*#?\s*\d+", re.IGNORECASE)
NUMBER_RE = re.compile(r"(?<![\w])[-+]?\d+(?:[.,]\d+)*(?![\w])")
NON_WORD_RE = re.compile(r"[^\w\u4e00-\u9fff<>]+", re.UNICODE)


class ValidationFailure(RuntimeError):
    """Raised by ``require_valid`` when one or more freeze gates fail."""


@dataclass(frozen=True)
class DatasetView:
    """Complete pre-seal view; never serialized into a public report."""

    roots: tuple[dict[str, Any], ...]
    graphs: tuple[dict[str, Any], ...]
    tasks: tuple[dict[str, Any], ...]
    messages: tuple[dict[str, Any], ...]
    attributions: tuple[dict[str, Any], ...]
    annotations: tuple[dict[str, Any], ...]
    annotation_extensions: tuple[dict[str, Any], ...]
    task_gold: tuple[dict[str, Any], ...]
    limitations: tuple[dict[str, Any], ...]
    retrieval_evidence: tuple[dict[str, Any], ...]
    document_facts: tuple[dict[str, Any], ...]
    sealed_payload: Mapping[str, Any]


def _read_json(path: Path, *, required: bool = True) -> Any:
    if not path.is_file():
        if required:
            raise FileNotFoundError(path)
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _as_records(value: Any, label: str) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise TypeError(f"{label} must be a JSON array of objects")
    return list(value)


def _first_records(base: Path, names: Sequence[str]) -> list[dict[str, Any]]:
    for name in names:
        path = base / name
        if path.is_file():
            return _as_records(_read_json(path), name)
    return []


def _merge_unique(
    development: Iterable[dict[str, Any]],
    holdout: Iterable[dict[str, Any]],
    *,
    id_key: str,
) -> tuple[dict[str, Any], ...]:
    merged: dict[str, dict[str, Any]] = {}
    for item in (*tuple(development), *tuple(holdout)):
        identifier = str(item.get(id_key, ""))
        if not identifier:
            raise ValueError(f"record is missing {id_key}")
        previous = merged.get(identifier)
        if previous is not None and previous != item:
            raise ValueError(f"conflicting duplicate {id_key}: {identifier}")
        merged[identifier] = item
    return tuple(merged.values())


def _sealed_records(payload: Mapping[str, Any], *keys: str) -> list[dict[str, Any]]:
    for key in keys:
        if key in payload:
            return _as_records(payload[key], f"sealed payload {key}")
    return []


def load_dataset_view(
    project_root: str | Path,
    *,
    sealed_payload: Mapping[str, Any] | None = None,
) -> DatasetView:
    """Load development plus the B-authorized pre-seal holdout payload.

    After a seal lock exists, B must not deserialize holdout Gold.  Callers
    performing post-seal integrity checks must use ``sealing.py`` instead.
    """

    root = Path(project_root).resolve()
    b_root = root / "work" / "B"
    seal_path = b_root / "seal" / "sealed_gold.payload.json"
    lock_path = b_root / "seal" / "sealed_gold.lock"
    if sealed_payload is None:
        if lock_path.exists():
            raise PermissionError(
                "B validation cannot deserialize holdout Gold after seal; validate before locking"
            )
        loaded = _read_json(seal_path)
        if not isinstance(loaded, dict):
            raise TypeError("sealed Gold payload must be a JSON object")
        sealed_payload = loaded

    manifests = b_root / "manifests"
    development = b_root / "generated" / "development"

    def dev_records(names: Sequence[str], fallback: Sequence[str]) -> list[dict[str, Any]]:
        records = _first_records(development, names)
        if not records:
            records = _first_records(manifests, fallback)
        return [item for item in records if item.get("split", "DEVELOPMENT") == "DEVELOPMENT"]

    roots = _merge_unique(
        dev_records(("root_scenarios.json",), ("root_scenarios.json",)),
        _sealed_records(sealed_payload, "root_scenarios"),
        id_key="root_scenario_id",
    )
    graphs = _merge_unique(
        dev_records(("case_graphs.json",), ("case_graphs.json",)),
        _sealed_records(sealed_payload, "case_graphs"),
        id_key="case_graph_id",
    )
    tasks = _merge_unique(
        dev_records(("task_instances.json",), ("task_instances.json",)),
        _sealed_records(sealed_payload, "task_instances"),
        id_key="task_instance_id",
    )
    messages = _merge_unique(
        dev_records(("customer_messages.json",), ("customer_messages.json",)),
        _sealed_records(sealed_payload, "customer_messages"),
        id_key="message_id",
    )
    attributions = _merge_unique(
        dev_records(
            ("expected_customer_attributions.json",),
            ("expected_customer_attributions.json", "gold_customer_attributions.json"),
        ),
        _sealed_records(
            sealed_payload,
            "expected_customer_attributions",
            "gold_customer_attributions",
        ),
        id_key="expected_customer_attribution_id",
    )
    annotations = _merge_unique(
        dev_records(
            ("expected_field_annotations.json",),
            ("expected_field_annotations.json", "gold_field_annotations.json"),
        ),
        _sealed_records(
            sealed_payload,
            "expected_field_annotations",
            "gold_field_annotations",
        ),
        id_key="expected_field_annotation_id",
    )
    annotation_extensions = _merge_unique(
        dev_records(
            ("field_annotation_extensions.json",),
            ("field_annotation_extensions.json",),
        ),
        _sealed_records(sealed_payload, "field_annotation_extensions"),
        id_key="expected_field_annotation_id",
    )
    task_gold = _merge_unique(
        dev_records(("expected_task_gold.json",), ("expected_task_gold.json",)),
        _sealed_records(sealed_payload, "expected_task_gold"),
        id_key="task_gold_id",
    )
    limitations = _merge_unique(
        dev_records(
            ("expected_limitation_decisions.json",),
            ("expected_limitation_decisions.json",),
        ),
        _sealed_records(sealed_payload, "expected_limitation_decisions"),
        id_key="expected_limitation_decision_id",
    )
    retrieval_evidence = _merge_unique(
        dev_records(
            ("expected_retrieval_evidence.json",),
            ("expected_retrieval_evidence.json",),
        ),
        _sealed_records(sealed_payload, "expected_retrieval_evidence"),
        id_key="expected_retrieval_evidence_id",
    )
    document_facts = tuple(
        _first_records(manifests, ("gold_document_facts.json",))
    )
    return DatasetView(
        roots=roots,
        graphs=graphs,
        tasks=tasks,
        messages=messages,
        attributions=attributions,
        annotations=annotations,
        annotation_extensions=annotation_extensions,
        task_gold=task_gold,
        limitations=limitations,
        retrieval_evidence=retrieval_evidence,
        document_facts=document_facts,
        sealed_payload=sealed_payload,
    )


def normalize_leakage_text(value: str) -> str:
    """Normalize content using the frozen cross-split duplicate profile."""

    text = unicodedata.normalize("NFKC", value).replace("\r\n", "\n").replace("\r", "\n")
    text = text.casefold()
    text = UUID_RE.sub(" <uuid> ", text)
    text = SHA_RE.sub(" <sha256> ", text)
    text = TIME_RE.sub(" <timestamp> ", text)
    text = ROOT_MARKER_RE.sub(" <root> ", text)
    text = re.sub(r"\b(?:development|sealed_holdout)\b", " <split> ", text)
    text = NUMBER_RE.sub(" <num> ", text)
    text = NON_WORD_RE.sub(" ", text)
    return " ".join(text.split())


def _shingles(value: str, size: int = NEAR_DUPLICATE_SHINGLE_SIZE) -> frozenset[str]:
    compact = value.replace(" ", "")
    if len(compact) <= size:
        return frozenset({compact}) if compact else frozenset()
    return frozenset(compact[index : index + size] for index in range(len(compact) - size + 1))


def near_duplicate_score(left: str, right: str) -> float:
    """Return character 5-gram Jaccard similarity for normalized strings."""

    left_set = _shingles(left)
    right_set = _shingles(right)
    union = left_set | right_set
    if not union:
        return 1.0
    return len(left_set & right_set) / len(union)


def _walk_selected_text(value: Any, path: tuple[str, ...] = ()) -> Iterable[tuple[str, str]]:
    selected = {
        "query",
        "question",
        "prompt",
        "customer_text",
        "draft_body",
        "body",
        "rationale",
        "reason_text",
        "allowed_answer",
    }
    if isinstance(value, dict):
        for key, item in value.items():
            child = (*path, str(key))
            if str(key).casefold() in selected and isinstance(item, str):
                yield "/".join(child), item
            else:
                yield from _walk_selected_text(item, child)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk_selected_text(item, (*path, str(index)))


def _content_units(view: DatasetView, split: str) -> list[dict[str, str]]:
    units: list[dict[str, str]] = []
    for message in view.messages:
        if message.get("split") not in (None, split):
            continue
        case_ids = {
            str(graph.get("case_id"))
            for graph in view.graphs
            if graph.get("split") == split
        }
        if str(message.get("case_id")) not in case_ids:
            continue
        text = message.get("text")
        if isinstance(text, str) and len(normalize_leakage_text(text)) >= MIN_DUPLICATE_TEXT_LENGTH:
            units.append(
                {
                    "kind": "customer_message",
                    "id": str(message["message_id"]),
                    "text": text,
                }
            )
    for gold in view.task_gold:
        if gold.get("split") != split:
            continue
        for path, text in _walk_selected_text(gold.get("expected", {})):
            if len(normalize_leakage_text(text)) >= MIN_DUPLICATE_TEXT_LENGTH:
                units.append(
                    {
                        "kind": f"task_gold_text:{path}",
                        "id": str(gold["task_gold_id"]),
                        "text": text,
                    }
                )
    return units


def compute_leakage_report(view: DatasetView) -> dict[str, Any]:
    """Compute atomic-ID overlap and exact/near content pairs from raw data."""

    development_roots = [item for item in view.roots if item.get("split") == "DEVELOPMENT"]
    holdout_roots = [item for item in view.roots if item.get("split") == "SEALED_HOLDOUT"]

    def overlap(field: str, *, many: bool) -> set[str]:
        if many:
            dev = {str(value) for item in development_roots for value in item.get(field, [])}
            hold = {str(value) for item in holdout_roots for value in item.get(field, [])}
        else:
            dev = {str(item.get(field)) for item in development_roots}
            hold = {str(item.get(field)) for item in holdout_roots}
        return dev & hold

    atomic = {
        "root_scenario_overlap": overlap("root_scenario_id", many=False),
        "fact_family_overlap": overlap("fact_family_ids", many=True),
        "source_lineage_overlap": overlap("source_lineage_ids", many=True),
        "template_family_overlap": overlap("template_family_id", many=False),
    }

    development_units = _content_units(view, "DEVELOPMENT")
    holdout_units = _content_units(view, "SEALED_HOLDOUT")
    unresolved: list[dict[str, Any]] = []
    exact_count = 0
    near_count = 0
    candidate_count = 0
    for development_item in development_units:
        left = normalize_leakage_text(development_item["text"])
        for holdout_item in holdout_units:
            if development_item["kind"] != holdout_item["kind"]:
                continue
            candidate_count += 1
            right = normalize_leakage_text(holdout_item["text"])
            if left == right:
                match_type = "EXACT_AFTER_NORMALIZATION"
                score = 1.0
                exact_count += 1
            else:
                score = near_duplicate_score(left, right)
                if score < NEAR_DUPLICATE_THRESHOLD:
                    continue
                match_type = "NEAR_DUPLICATE_5GRAM_JACCARD"
                near_count += 1
            unresolved.append(
                {
                    "content_kind": development_item["kind"],
                    "development_item_id": development_item["id"],
                    "holdout_item_blind_ref": sha256_json(
                        {
                            "kind": holdout_item["kind"],
                            "id": holdout_item["id"],
                        }
                    ),
                    "match_type": match_type,
                    "similarity": format(score, ".6f"),
                }
            )

    atomic_count = sum(len(values) for values in atomic.values())
    status = "PASS" if atomic_count == 0 and not unresolved else "FAIL"
    return {
        "report_version": REPORT_VERSION,
        "status": status,
        "algorithm": "cross-split exact normalized equality plus character 5-gram Jaccard",
        "normalization_rules": [
            "Unicode NFKC",
            "CRLF/CR to LF",
            "Unicode casefold",
            "replace UUID/SHA256/RFC3339/split/root ordinal/decimal runs with placeholders",
            "collapse punctuation and whitespace",
            "compare customer messages and selected query/prompt/body/rationale fields only",
            "exclude shared public knowledge documents",
        ],
        "near_duplicate_threshold": format(NEAR_DUPLICATE_THRESHOLD, ".2f"),
        "shingle_size": NEAR_DUPLICATE_SHINGLE_SIZE,
        "minimum_normalized_length": MIN_DUPLICATE_TEXT_LENGTH,
        "development_content_unit_count": len(development_units),
        "holdout_content_unit_count": len(holdout_units),
        "actual_candidate_pair_count": candidate_count,
        "exact_duplicate_pair_count": exact_count,
        "near_duplicate_pair_count": near_count,
        "unresolved_near_duplicates": len(unresolved),
        "atomic_overlap_counts": {key: len(value) for key, value in atomic.items()},
        "atomic_overlap_blind_refs": {
            key: [sha256_json({"field": key, "id": item}) for item in sorted(value)]
            for key, value in atomic.items()
        },
        "unresolved_pairs": unresolved,
    }


def _strip_for_model(model_type: type[Any], record: Mapping[str, Any]) -> dict[str, Any]:
    return {key: record[key] for key in model_type.model_fields if key in record}


def _validate_shared_schemas(
    root: Path,
    view: DatasetView,
    failures: list[dict[str, str]],
) -> dict[str, int]:
    manifests = root / "work" / "B" / "manifests"
    metadata_records = _as_records(
        _read_json(manifests / "source_document_metadata.json"),
        "source_document_metadata",
    )
    fact_records = _as_records(
        _read_json(manifests / "gold_document_facts.json"),
        "gold_document_facts",
    )
    counts: dict[str, int] = {}

    checks: tuple[tuple[str, type[Any], Iterable[Mapping[str, Any]]], ...] = (
        ("SourceDocumentMetadata", SourceDocumentMetadata, metadata_records),
        ("GoldDocumentFact", GoldDocumentFact, fact_records),
        ("CustomerMessage", CustomerMessage, view.messages),
        ("ExpectedCustomerAttribution", ExpectedCustomerAttribution, view.attributions),
        ("ExpectedFieldAnnotation", ExpectedFieldAnnotation, view.annotations),
        ("ExpectedLimitationDecision", ExpectedLimitationDecision, view.limitations),
    )
    for label, model_type, records in checks:
        materialized = tuple(records)
        counts[label] = len(materialized)
        for index, record in enumerate(materialized):
            try:
                model_type.model_validate(_strip_for_model(model_type, record))
            except (ValidationError, ValueError, TypeError) as exc:
                failures.append(
                    {
                        "code": "A_OR_B_SCHEMA_INVALID",
                        "detail": f"{label}[{index}]: {exc}",
                    }
                )

    lifecycle_records = _as_records(
        _read_json(manifests / "lifecycle_expectations.json"),
        "lifecycle_expectations",
    )
    lifecycle_payloads = 0
    metadata_version_ids = {str(item["document_version_id"]) for item in metadata_records}
    metadata_revision_ids = {
        str(item["source_metadata_revision_id"]) for item in metadata_records
    }
    lifecycle_error_status = {
        "LIFECYCLE_NOT_YET_CREATED": "NOT_CREATED",
        "LIFECYCLE_VERSION_GAP": "UNRESOLVED",
        "LIFECYCLE_EVENT_MISSING": "INVALID",
        "LIFECYCLE_EVENT_CONFLICT": "CONFLICT",
        "LIFECYCLE_INTERVAL_INVALID": "INVALID",
        "LIFECYCLE_CORRECTION_INVALID": "INVALID",
    }
    for index, record in enumerate(lifecycle_records):
        record_version_id = str(record.get("document_version_id"))
        if record_version_id not in metadata_version_ids:
            failures.append(
                {
                    "code": "LIFECYCLE_DOCUMENT_REFERENCE_INVALID",
                    "detail": f"lifecycle[{index}] references an unknown document version",
                }
            )
        for event_index, event in enumerate(record.get("events", [])):
            try:
                parsed_event = DocumentLifecycleEvent.model_validate(event)
                if str(parsed_event.document_version_id) != record_version_id:
                    raise ValueError("event document_version_id differs from expectation")
                if str(parsed_event.metadata_revision_id) not in metadata_revision_ids:
                    raise ValueError("event metadata_revision_id is unknown")
            except (ValidationError, ValueError, TypeError) as exc:
                failures.append(
                    {
                        "code": "LIFECYCLE_EVENT_SCHEMA_INVALID",
                        "detail": f"lifecycle[{index}].events[{event_index}]: {exc}",
                    }
                )
        expected = record.get("expected", {})
        payload = expected.get("payload") or expected.get("shared_lifecycle_at_result")
        if payload is None:
            failures.append(
                {
                    "code": "LIFECYCLE_EXPECTATION_MISSING_PAYLOAD",
                    "detail": f"lifecycle[{index}] has no A LifecycleAtResult payload",
                }
            )
            continue
        lifecycle_payloads += 1
        try:
            parsed_result = LifecycleAtResult.model_validate(payload)
            if str(parsed_result.document_version_id) != record_version_id:
                raise ValueError("result document_version_id differs from expectation")
            if parsed_result.error_code is not None:
                expected_status = lifecycle_error_status.get(parsed_result.error_code)
                if expected_status is None or parsed_result.result_status.value != expected_status:
                    raise ValueError("lifecycle error_code/result_status pair is not canonical")
        except (ValidationError, ValueError, TypeError) as exc:
            failures.append(
                {
                    "code": "LIFECYCLE_RESULT_SCHEMA_INVALID",
                    "detail": f"lifecycle[{index}]: {exc}",
                }
            )
    counts["LifecycleAtResult"] = lifecycle_payloads
    counts["DocumentLifecycleEvent"] = sum(len(item.get("events", [])) for item in lifecycle_records)
    return counts


def _extract_pdf_pages(path: Path) -> list[str]:
    try:
        import fitz  # type: ignore
    except Exception as exc:  # pragma: no cover - environment prerequisite
        raise RuntimeError("PyMuPDF is required for PDF Gold verification") from exc
    document = fitz.open(path)
    try:
        return [
            page.get_text("text").replace("\r\n", "\n").replace("\r", "\n").strip()
            for page in document
        ]
    finally:
        document.close()


def _extract_docx_paragraphs(path: Path) -> list[str]:
    with zipfile.ZipFile(path, "r") as archive:
        xml = archive.read("word/document.xml")
    root = ET.fromstring(xml)
    namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    paragraphs: list[str] = []
    for paragraph in root.iter(namespace + "p"):
        text = "".join(node.text or "" for node in paragraph.iter(namespace + "t"))
        paragraphs.append(text)
    return paragraphs


def _document_material(path: Path, document_format: str) -> dict[str, Any]:
    if document_format == "PDF":
        pages = _extract_pdf_pages(path)
        return {"canonical": "\n".join(pages), "pages": pages}
    if document_format == "DOCX":
        paragraphs = _extract_docx_paragraphs(path)
        return {"canonical": "\n".join(paragraphs), "paragraphs": paragraphs}
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    if document_format == "MARKDOWN":
        return {
            "canonical": text,
            "blocks": [block for block in text.split("\n\n") if block],
        }
    return {"canonical": text, "lines": text.split("\n")}


def _markdown_heading_path(blocks: Sequence[str], block_index: int) -> tuple[str, ...]:
    headings: list[tuple[int, str]] = []
    for block in blocks[:block_index]:
        first_line = block.split("\n", 1)[0]
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", first_line)
        if not match:
            continue
        level = len(match.group(1))
        headings = [item for item in headings if item[0] < level]
        headings.append((level, match.group(2)))
    return tuple(text for _, text in headings)


def verify_document_locators(project_root: str | Path) -> dict[str, Any]:
    """Verify asset bytes plus every A four-format Gold coordinate and hash."""

    root = Path(project_root).resolve()
    manifests = root / "work" / "B" / "manifests"
    metadata_raw = _as_records(
        _read_json(manifests / "source_document_metadata.json"),
        "source_document_metadata",
    )
    facts_raw = _as_records(
        _read_json(manifests / "gold_document_facts.json"),
        "gold_document_facts",
    )
    metadata = {
        str(item["document_version_id"]): SourceDocumentMetadata.model_validate(
            _strip_for_model(SourceDocumentMetadata, item)
        )
        for item in metadata_raw
    }
    if len(metadata) != len(metadata_raw):
        raise ValueError("duplicate SourceDocumentMetadata.document_version_id")
    material: dict[str, dict[str, Any]] = {}
    asset_failures: list[str] = []
    for version_id, record in metadata.items():
        source_path = (root / str(record.source_ref)).resolve()
        try:
            source_path.relative_to(root)
        except ValueError:
            asset_failures.append(f"source_ref escapes project: {record.source_ref}")
            continue
        if not source_path.is_file():
            asset_failures.append(f"missing asset: {record.source_ref}")
            continue
        if file_sha256(source_path) != record.source_sha256:
            asset_failures.append(f"asset SHA mismatch: {record.source_ref}")
            continue
        material[version_id] = _document_material(source_path, record.document_format.value)

    format_counts: Counter[str] = Counter()
    locator_count = 0
    failures: list[str] = list(asset_failures)
    pdf_fact_keys: set[str] = set()
    fact_ids_seen: set[str] = set()
    locator_ids_seen: set[str] = set()
    adapter = TypeAdapter(GoldDocumentLocator)
    for fact_index, fact_raw in enumerate(facts_raw):
        try:
            fact = GoldDocumentFact.model_validate(fact_raw)
        except ValidationError as exc:
            failures.append(f"GoldDocumentFact[{fact_index}] schema: {exc}")
            continue
        if str(fact.gold_fact_id) in fact_ids_seen:
            failures.append(f"duplicate gold_fact_id: {fact.gold_fact_id}")
        fact_ids_seen.add(str(fact.gold_fact_id))
        doc = material.get(str(fact.document_version_id))
        if doc is None:
            failures.append(f"fact {fact.gold_fact_id} has no verified source asset")
            continue
        for raw_locator in fact_raw.get("gold_document_locators", []):
            locator_count += 1
            try:
                locator = adapter.validate_python(raw_locator)
                if str(locator.gold_locator_id) in locator_ids_seen:
                    raise ValueError("duplicate gold_locator_id")
                locator_ids_seen.add(str(locator.gold_locator_id))
                format_name = locator.document_format.value
                format_counts[format_name] += 1
                if sha256_text(doc["canonical"]) != locator.canonical_text_sha256:
                    raise ValueError("canonical_text_sha256 mismatch")
                if isinstance(locator, GoldPdfLocator):
                    pages = doc["pages"]
                    assert locator.page_index is not None
                    page = pages[locator.page_index]
                    assert locator.page_char_start is not None and locator.page_char_end is not None
                    assert locator.page_canonical_text_sha256 is not None
                    if sha256_text(page) != locator.page_canonical_text_sha256:
                        raise ValueError("PDF page hash mismatch")
                    if page[locator.page_char_start : locator.page_char_end] != locator.quote:
                        raise ValueError("PDF page range does not reproduce quote")
                    pdf_fact_keys.add(str(fact.fact_key))
                elif isinstance(locator, GoldDocxLocator):
                    paragraphs = doc["paragraphs"]
                    assert locator.paragraph_index is not None
                    paragraph = paragraphs[locator.paragraph_index]
                    assert locator.paragraph_char_start is not None
                    assert locator.paragraph_char_end is not None
                    assert locator.paragraph_text_sha256 is not None
                    if sha256_text(paragraph) != locator.paragraph_text_sha256:
                        raise ValueError("DOCX paragraph hash mismatch")
                    if paragraph[locator.paragraph_char_start : locator.paragraph_char_end] != locator.quote:
                        raise ValueError("DOCX paragraph range does not reproduce quote")
                    assert locator.section_path is not None
                    preceding = paragraphs[: locator.paragraph_index]
                    cursor = -1
                    for section in locator.section_path:
                        try:
                            cursor = preceding.index(str(section), cursor + 1)
                        except ValueError as exc:
                            raise ValueError("DOCX section_path is absent before paragraph") from exc
                elif isinstance(locator, GoldMarkdownLocator):
                    blocks = doc["blocks"]
                    assert locator.block_index is not None
                    block = blocks[locator.block_index]
                    assert locator.block_char_start is not None and locator.block_char_end is not None
                    assert locator.block_text_sha256 is not None
                    if sha256_text(block) != locator.block_text_sha256:
                        raise ValueError("Markdown block hash mismatch")
                    if block[locator.block_char_start : locator.block_char_end] != locator.quote:
                        raise ValueError("Markdown block range does not reproduce quote")
                    assert locator.section_path is not None
                    actual_path = _markdown_heading_path(blocks, locator.block_index)
                    if tuple(locator.section_path) != actual_path[-len(locator.section_path) :]:
                        raise ValueError("Markdown section_path does not match preceding headings")
                elif isinstance(locator, GoldTextLocator):
                    lines = doc["lines"]
                    assert locator.line_index is not None
                    line = lines[locator.line_index]
                    assert locator.line_char_start is not None and locator.line_char_end is not None
                    assert locator.line_text_sha256 is not None
                    if sha256_text(line) != locator.line_text_sha256:
                        raise ValueError("TEXT line hash mismatch")
                    if line[locator.line_char_start : locator.line_char_end] != locator.quote:
                        raise ValueError("TEXT line range does not reproduce quote")
            except (ValidationError, ValueError, IndexError, AssertionError) as exc:
                failures.append(f"fact[{fact_index}] locator[{locator_count - 1}]: {exc}")

    missing_pdf_keys = sorted(PDF_REQUIRED_FACT_KEYS - pdf_fact_keys)
    if missing_pdf_keys:
        failures.append(f"PDF required fact keys missing: {missing_pdf_keys}")
    missing_formats = sorted({"PDF", "DOCX", "MARKDOWN", "TEXT"} - set(format_counts))
    if missing_formats:
        failures.append(f"Gold locator formats missing: {missing_formats}")
    return {
        "status": "PASS" if not failures else "FAIL",
        "asset_count": len(metadata),
        "verified_asset_count": len(material),
        "locator_count": locator_count,
        "locator_counts_by_format": dict(sorted(format_counts.items())),
        "required_pdf_fact_keys": sorted(PDF_REQUIRED_FACT_KEYS),
        "verified_pdf_fact_keys": sorted(pdf_fact_keys),
        "failures": failures,
    }


def _record_ids(records: Iterable[Mapping[str, Any]], key: str) -> tuple[list[str], set[str]]:
    values = [str(item.get(key, "")) for item in records]
    return values, set(values)


def validate_reference_closure(view: DatasetView) -> dict[str, Any]:
    """Validate all B graph, Gold, customer, fact and namespace references."""

    failures: list[str] = []
    id_specs = (
        ("root", view.roots, "root_scenario_id"),
        ("case_graph", view.graphs, "case_graph_id"),
        ("task", view.tasks, "task_instance_id"),
        ("message", view.messages, "message_id"),
        ("attribution", view.attributions, "expected_customer_attribution_id"),
        ("annotation", view.annotations, "expected_field_annotation_id"),
        ("task_gold", view.task_gold, "task_gold_id"),
        ("limitation", view.limitations, "expected_limitation_decision_id"),
        (
            "retrieval_evidence",
            view.retrieval_evidence,
            "expected_retrieval_evidence_id",
        ),
    )
    id_sets: dict[str, set[str]] = {}
    for label, records, key in id_specs:
        values, unique = _record_ids(records, key)
        id_sets[label] = unique
        if "" in unique:
            failures.append(f"{label} contains an empty {key}")
        if len(values) != len(unique):
            failures.append(f"duplicate {key}")

    roots = {str(item["root_scenario_id"]): item for item in view.roots}
    graphs = {str(item["case_graph_id"]): item for item in view.graphs}
    tasks = {str(item["task_instance_id"]): item for item in view.tasks}
    messages = {str(item["message_id"]): item for item in view.messages}
    task_gold = {str(item["task_gold_id"]): item for item in view.task_gold}
    fact_ids = {
        str(item["gold_fact_id"])
        for item in view.document_facts
    }

    for root_id, root in roots.items():
        for graph_id in root.get("case_graph_ids", []):
            graph = graphs.get(str(graph_id))
            if graph is None:
                failures.append(f"root {root_id} references missing graph {graph_id}")
            elif str(graph.get("root_scenario_id")) != root_id:
                failures.append(f"graph {graph_id} points to a different root")
    referenced_graph_ids = {
        str(graph_id) for root in view.roots for graph_id in root.get("case_graph_ids", [])
    }
    if referenced_graph_ids != set(graphs):
        failures.append("root case_graph references do not exactly cover all graphs")

    nested_task_ids: set[str] = set()
    for graph_id, graph in graphs.items():
        root = roots.get(str(graph.get("root_scenario_id")))
        if root is None:
            failures.append(f"graph {graph_id} references missing root")
            continue
        if graph.get("split") != root.get("split"):
            failures.append(f"graph {graph_id} split differs from root")
        if str(graph.get("case_id")) != str(root.get("case_id")):
            failures.append(f"graph {graph_id} case differs from root extension")
        for message_id in graph.get("message_ids", []):
            message = messages.get(str(message_id))
            if message is None:
                failures.append(f"graph {graph_id} references missing message {message_id}")
            elif str(message.get("case_id")) != str(graph.get("case_id")):
                failures.append(f"message {message_id} belongs to a different case")
        for nested in graph.get("task_instances", []):
            task_id = str(nested.get("task_instance_id"))
            nested_task_ids.add(task_id)
            task = tasks.get(task_id)
            if task is None:
                failures.append(f"graph {graph_id} references missing task {task_id}")
            elif any(
                nested.get(field) != task.get(field)
                for field in ("task_type", "split", "eligible", "exclusion_reason_code")
            ):
                failures.append(f"graph task projection differs for {task_id}")
    if nested_task_ids != set(tasks):
        failures.append("top-level and graph-nested task ID sets differ")

    gold_by_task: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for gold in view.task_gold:
        gold_by_task[str(gold.get("task_instance_id"))].append(gold)
    for task_id, task in tasks.items():
        if str(task.get("root_scenario_id")) not in roots:
            failures.append(f"task {task_id} references missing root")
        if str(task.get("case_graph_id")) not in graphs:
            failures.append(f"task {task_id} references missing graph")
        if any(str(item) not in messages for item in task.get("input_refs", [])):
            failures.append(f"task {task_id} has missing input message")
        records = gold_by_task.get(task_id, [])
        if len(records) != 1:
            failures.append(f"task {task_id} has {len(records)} expected_task_gold records")
            continue
        gold = records[0]
        if str(task.get("gold_ref")) != str(gold.get("task_gold_id")):
            failures.append(f"task {task_id} gold_ref mismatch")
        if any(gold.get(field) != task.get(field) for field in ("task_type", "split", "case_id")):
            failures.append(f"task {task_id} Gold projection mismatch")

    attribution_ids = id_sets["attribution"]
    locator_ids = {
        str(item.get("gold_customer_locator", {}).get("gold_customer_locator_id"))
        for item in view.attributions
    }
    for attribution in view.attributions:
        message = messages.get(str(attribution.get("message_id")))
        if message is None:
            failures.append("attribution references missing message")
            continue
        locator = attribution.get("gold_customer_locator", {})
        text = str(message.get("text", ""))
        start, end = locator.get("char_start"), locator.get("char_end")
        if not isinstance(start, int) or not isinstance(end, int) or text[start:end] != locator.get("quote"):
            failures.append(
                f"customer locator range mismatch: {attribution.get('expected_customer_attribution_id')}"
            )
        if attribution.get("message_sha256") != message.get("text_sha256"):
            failures.append("attribution message hash differs from CustomerMessage")

    annotation_extension_ids = {
        str(item.get("expected_field_annotation_id")) for item in view.annotation_extensions
    }
    if annotation_extension_ids != id_sets["annotation"]:
        failures.append("field annotation extension IDs do not exactly cover annotations")
    cases = {str(item.get("case_id")) for item in view.graphs}
    for extension in view.annotation_extensions:
        root = roots.get(str(extension.get("root_scenario_id")))
        if root is None:
            failures.append("field annotation extension references missing root")
        if str(extension.get("case_id")) not in cases:
            failures.append("field annotation extension references missing case")
        elif root is not None and str(extension.get("case_id")) != str(root.get("case_id")):
            failures.append("field annotation extension root/case binding mismatch")
    referenced_attribution_ids: set[str] = set()
    referenced_locator_ids: set[str] = set()
    for annotation in view.annotations:
        shared = annotation.get("shared_field_annotation", {})
        referenced_attribution_ids.update(
            str(item) for item in shared.get("customer_attribution_ids", [])
        )
        referenced_locator_ids.update(str(item) for item in shared.get("source_locator_ids", []))
        if any(str(item) not in attribution_ids for item in shared.get("customer_attribution_ids", [])):
            failures.append("field annotation references missing customer attribution")
        if any(str(item) not in locator_ids for item in shared.get("source_locator_ids", [])):
            failures.append("field annotation references missing GoldCustomerLocator")
    if referenced_attribution_ids != attribution_ids:
        failures.append("field annotations do not exactly cover customer attributions")
    if referenced_locator_ids != locator_ids:
        failures.append("field annotations do not exactly cover GoldCustomerLocators")

    limitation_ids = id_sets["limitation"]
    for gold in view.task_gold:
        expected = gold.get("expected", {})
        task_type = gold.get("task_type")
        if task_type == "REQUIREMENT_EXTRACTION":
            if expected.get("expected_requirement_id") is None:
                failures.append(f"requirement task {gold.get('task_instance_id')} has null requirement ID")
            if any(str(item) not in id_sets["annotation"] for item in expected.get("expected_field_annotation_ids", [])):
                failures.append("requirement Gold references missing field annotation")
            if any(str(item) not in attribution_ids for item in expected.get("expected_customer_attribution_ids", [])):
                failures.append("requirement Gold references missing customer attribution")
        if task_type == "REPLY_DRAFT":
            limitation_id = expected.get("expected_limitation_decision_id")
            if str(limitation_id) not in limitation_ids:
                failures.append("reply draft Gold references missing limitation decision")
        if task_type == "REFUSAL":
            if expected.get("private_type") != "b_gold.ExpectedRefusalDecision":
                failures.append("refusal task lacks explicit ExpectedRefusalDecision")
            if not expected.get("reason_code") or not expected.get("expected_outcome"):
                failures.append("refusal task lacks outcome or reason Gold")
        for fact_id in gold.get("gold_fact_ids", []):
            if str(fact_id) not in fact_ids:
                failures.append(f"task Gold references missing document fact {fact_id}")
        expected = gold.get("expected", {})
        for claim in expected.get("expected_claims", []):
            if str(claim.get("gold_fact_id")) not in fact_ids:
                failures.append("QA expected claim references missing document fact")
        for slot in expected.get("expected_fact_slots", []):
            if str(slot.get("gold_fact_id")) not in fact_ids:
                failures.append("reply expected fact slot references missing document fact")

    retrieval_evidence = {
        str(item.get("expected_retrieval_evidence_id")): item
        for item in view.retrieval_evidence
    }
    referenced_retrieval_ids: set[str] = set()
    for graph in view.graphs:
        graph_retrieval_ids = {
            str(item) for item in graph.get("expected_retrieval_evidence_ids", [])
        }
        referenced_retrieval_ids.update(graph_retrieval_ids)
        if any(item not in retrieval_evidence for item in graph_retrieval_ids):
            failures.append(
                f"case graph {graph.get('case_graph_id')} references missing retrieval expectation"
            )
    if referenced_retrieval_ids != set(retrieval_evidence):
        failures.append("case graphs do not exactly cover retrieval expectations")
    if len(retrieval_evidence) != PLANNED_END_TO_END_CHAIN_COUNT:
        failures.append("exactly 30 retrieval expectations are required")
    for evidence_id, evidence in retrieval_evidence.items():
        graph = graphs.get(str(evidence.get("case_graph_id")))
        if graph is None:
            failures.append(f"retrieval expectation {evidence_id} references missing graph")
            continue
        if any(
            evidence.get(field) != graph.get(field)
            for field in ("root_scenario_id", "case_id", "split")
        ):
            failures.append(f"retrieval expectation {evidence_id} graph binding mismatch")
        if not evidence.get("gold_fact_ids") or any(
            str(item) not in fact_ids for item in evidence.get("gold_fact_ids", [])
        ):
            failures.append(f"retrieval expectation {evidence_id} has invalid Gold facts")

    reply_task_ids = {
        str(item.get("task_instance_id"))
        for item in view.tasks
        if item.get("task_type") == "REPLY_DRAFT"
    }
    limitation_task_ids = {str(item.get("task_instance_id")) for item in view.limitations}
    if limitation_task_ids != reply_task_ids:
        failures.append("limitation decisions do not exactly cover ReplyDraft tasks")

    security_refusal_count = 0
    for task in view.tasks:
        root = roots.get(str(task.get("root_scenario_id")), {})
        if root.get("scenario_class") == "SECURITY_COMPOSITE" and task.get("task_type") == "REFUSAL":
            security_refusal_count += 1
    if security_refusal_count != 8:
        failures.append(
            f"SECURITY_COMPOSITE must contain 8 explicit refusal tasks, got {security_refusal_count}"
        )

    return {
        "status": "PASS" if not failures else "FAIL",
        "id_counts": {label: len(values) for label, values in id_sets.items()},
        "security_composite_refusal_count": security_refusal_count,
        "failures": failures,
    }


def validate_counts_and_chains(view: DatasetView) -> tuple[dict[str, Any], dict[str, Any]]:
    root_split = Counter(str(item.get("split")) for item in view.roots)
    task_split = Counter(str(item.get("split")) for item in view.tasks)
    task_types = Counter(str(item.get("task_type")) for item in view.tasks)
    expected_types = {key.value: value for key, value in PLANNED_TASK_TYPE_COUNTS.items()}
    expected_root_split = {
        "DEVELOPMENT": PLANNED_DEVELOPMENT_ROOT_COUNT,
        "SEALED_HOLDOUT": PLANNED_HOLDOUT_ROOT_COUNT,
    }
    expected_task_split = {
        "DEVELOPMENT": PLANNED_DEVELOPMENT_TASK_COUNT,
        "SEALED_HOLDOUT": PLANNED_HOLDOUT_TASK_COUNT,
    }
    count_checks = {
        "root_total": len(view.roots) == PLANNED_ROOT_SCENARIO_COUNT,
        "task_total": len(view.tasks) == PLANNED_TASK_INSTANCE_COUNT,
        "root_split": dict(root_split) == expected_root_split,
        "task_split": dict(task_split) == expected_task_split,
        "task_types": {key: task_types.get(key, 0) for key in TASK_TYPE_NAMES} == expected_types,
        "task_gold_total": len(view.task_gold) == PLANNED_TASK_INSTANCE_COUNT,
        "retrieval_evidence_total": (
            len(view.retrieval_evidence) == PLANNED_END_TO_END_CHAIN_COUNT
        ),
    }
    count_report = {
        "status": "PASS" if all(count_checks.values()) else "FAIL",
        "checks": count_checks,
        "actual": {
            "root_scenarios": len(view.roots),
            "task_instances": len(view.tasks),
            "root_split": dict(root_split),
            "task_split": dict(task_split),
            "task_types": {key: task_types.get(key, 0) for key in TASK_TYPE_NAMES},
            "expected_task_gold": len(view.task_gold),
            "expected_retrieval_evidence": len(view.retrieval_evidence),
            "expected_customer_attributions": len(view.attributions),
            "expected_field_annotations": len(view.annotations),
            "expected_limitation_decisions": len(view.limitations),
        },
        "planned": {
            "root_scenarios": PLANNED_ROOT_SCENARIO_COUNT,
            "task_instances": PLANNED_TASK_INSTANCE_COUNT,
            "root_split": expected_root_split,
            "task_split": expected_task_split,
            "task_types": expected_types,
        },
    }

    chains = [
        item
        for item in view.graphs
        if item.get("expected_requirement_id") is not None
        and bool(item.get("expected_retrieval_evidence_ids"))
        and item.get("expected_qa_response_id") is not None
        and item.get("expected_reply_draft_id") is not None
    ]
    unique_chain_tasks = {
        str(task.get("task_instance_id"))
        for graph in chains
        for task in graph.get("task_instances", [])
    }
    chain_report = {
        "status": "PASS" if len(chains) == PLANNED_END_TO_END_CHAIN_COUNT else "FAIL",
        "actual_end_to_end_chain_count": len(chains),
        "planned_end_to_end_chain_count": PLANNED_END_TO_END_CHAIN_COUNT,
        "end_to_end_chains_are_additive": False,
        "task_denominator": len(view.tasks),
        "chain_task_relationship_count": len(unique_chain_tasks),
        "chain_blind_refs": [
            sha256_json({"case_graph_id": item.get("case_graph_id")}) for item in chains
        ],
    }
    return count_report, chain_report


def _namespace_entries(raw: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    namespaces: list[dict[str, Any]] = []
    contexts: list[dict[str, Any]] = []
    if isinstance(raw, list):
        for item in raw:
            item_namespaces, item_contexts = _namespace_entries(item)
            namespaces.extend(item_namespaces)
            contexts.extend(item_contexts)
    elif isinstance(raw, dict) and "namespace_keys" in raw:
        values = raw["namespace_keys"]
        namespaces.extend(values.values() if isinstance(values, dict) else values)
        contexts.extend(raw.get("b_dataset_execution_contexts", raw.get("contexts", [])))
    elif isinstance(raw, dict) and "namespace" in raw:
        if isinstance(raw["namespace"], dict):
            namespaces.append(raw["namespace"])
        value = raw.get(
            "b_dataset_execution_contexts",
            raw.get("b_dataset_execution_context", raw.get("contexts", [])),
        )
        contexts.extend(value if isinstance(value, list) else [value])
    elif isinstance(raw, dict):
        for value in raw.values():
            if not isinstance(value, dict):
                continue
            if isinstance(value.get("namespace"), dict):
                namespaces.append(value["namespace"])
            context_value = value.get("b_dataset_execution_context")
            if isinstance(context_value, dict):
                contexts.append(context_value)
            elif isinstance(context_value, list):
                contexts.extend(context_value)
    return namespaces, contexts


def validate_namespaces(project_root: str | Path, view: DatasetView) -> dict[str, Any]:
    root = Path(project_root).resolve()
    report_path = root / "work" / "B" / "reports" / "namespace_report.json"
    context_paths = (
        root / "work" / "B" / "manifests" / "b_dataset_execution_context.development.json",
        root / "work" / "B" / "manifests" / "b_dataset_execution_context.sealed.json",
    )
    separate_raw = [_read_json(path, required=False) for path in context_paths]
    separate_raw = [item for item in separate_raw if item is not None]
    if separate_raw:
        namespace_raw = []
        context_raw = []
        for raw in separate_raw:
            item_namespaces, item_contexts = _namespace_entries(raw)
            namespace_raw.extend(item_namespaces)
            context_raw.extend(item_contexts)
    else:
        raw = _read_json(report_path)
        namespace_raw, context_raw = _namespace_entries(raw)
    sealed_namespace = view.sealed_payload.get("execution_namespace")
    if sealed_namespace is not None:
        item_namespaces, item_contexts = _namespace_entries(sealed_namespace)
        namespace_raw.extend(item_namespaces)
        context_raw.extend(item_contexts)
    namespace_raw = list(
        {
            str(item.get("namespace_hash")): item
            for item in namespace_raw
            if isinstance(item, dict)
        }.values()
    )
    context_raw = list(
        {
            str(item.get("connected_component_id")): item
            for item in context_raw
            if isinstance(item, dict)
        }.values()
    )
    failures: list[str] = []
    namespaces: dict[str, ExecutionNamespaceKey] = {}
    for index, item in enumerate(namespace_raw):
        try:
            key = ExecutionNamespaceKey.model_validate(item)
            if not verify_execution_namespace_key(key):
                raise ValueError("canonical namespace hash mismatch")
            if set(key.model_dump(exclude={"schema_version", "namespace_hash"})) != {
                "environment",
                "corpus_manifest_hash",
                "split_manifest_hash",
                "document_version_set_hash",
                "source_hash_set_hash",
                "contract_bundle_hash",
                "configuration_hash",
                "code_hash",
                "run_id",
            }:
                raise ValueError("A namespace key does not contain exactly nine shared dimensions")
            namespaces[key.environment.value] = key
        except (ValidationError, ValueError, TypeError) as exc:
            failures.append(f"namespace[{index}]: {exc}")
    if set(namespaces) != {"DEVELOPMENT", "SEALED_HOLDOUT"}:
        failures.append("namespace report must contain exactly one A key per split")

    code_files = sorted((root / "src" / "assistant" / "module_b").glob("*.py"))
    current_code_hash = sha256_json({path.name: file_sha256(path) for path in code_files})
    metadata = _as_records(
        _read_json(root / "work" / "B" / "manifests" / "source_document_metadata.json"),
        "source_document_metadata",
    )
    split_manifest = _read_json(root / "work" / "B" / "manifests" / "split_manifest.json")
    split_hash = sha256_json(
        {key: value for key, value in split_manifest.items() if key != "split_manifest_hash"}
    )
    source_hash_set_hash = sha256_json(sorted(item["source_sha256"] for item in metadata))
    document_version_set_hash = sha256_json(
        sorted(item["document_version_id"] for item in metadata)
    )
    corpus_manifest_hash = sha256_json(
        {"metadata": metadata, "source_hash_set_hash": source_hash_set_hash}
    )
    contract_bundle_hash = sha256_text(
        "A-CONTRACT-v1.1.0-final|B-DATA-CONTRACT-v1.2.0|schema:1.1.0"
    )
    configuration_hash = sha256_json(
        {"seed": "B-20260818", "generator": "1.0.0", "split": "atomic"}
    )
    for split, key in namespaces.items():
        if key.code_hash != current_code_hash:
            failures.append(f"{split} namespace code_hash is stale")
        expected_hashes = {
            "corpus_manifest_hash": corpus_manifest_hash,
            "split_manifest_hash": split_hash,
            "document_version_set_hash": document_version_set_hash,
            "source_hash_set_hash": source_hash_set_hash,
            "contract_bundle_hash": contract_bundle_hash,
            "configuration_hash": configuration_hash,
        }
        for field, expected in expected_hashes.items():
            if getattr(key, field) != expected:
                failures.append(f"{split} namespace {field} is stale or mismatched")

    contexts: list[BDatasetExecutionContext] = []
    for index, item in enumerate(context_raw):
        try:
            context = BDatasetExecutionContext.model_validate(item)
            namespace = namespaces.get(context.split.value)
            if namespace is None:
                raise ValueError("context has no split namespace")
            context.assert_namespace_key(namespace)
            contexts.append(context)
        except (ValidationError, ValueError, TypeError) as exc:
            failures.append(f"context[{index}]: {exc}")
    component_ids = [str(item.connected_component_id) for item in contexts]
    if len(component_ids) != len(set(component_ids)):
        failures.append("B connected_component_id values must be unique")
    if len(contexts) != len(view.roots):
        failures.append(
            f"one B context is required per connected root group: {len(contexts)} != {len(view.roots)}"
        )

    context_signatures = Counter(
        (
            context.split.value,
            tuple(sorted(str(item) for item in context.fact_family_ids)),
            str(context.template_family_id),
            tuple(sorted(str(item) for item in context.source_lineage_ids)),
        )
        for context in contexts
    )
    root_signatures = Counter(
        (
            str(item.get("split")),
            tuple(sorted(str(value) for value in item.get("fact_family_ids", []))),
            str(item.get("template_family_id")),
            tuple(sorted(str(value) for value in item.get("source_lineage_ids", []))),
        )
        for item in view.roots
    )
    if context_signatures != root_signatures:
        failures.append("B context connected-component signatures do not exactly match roots")

    namespace_hashes = {key.namespace_hash for key in namespaces.values()}
    for task in view.tasks:
        if task.get("execution_namespace_key_hash") not in namespace_hashes:
            failures.append(f"task {task.get('task_instance_id')} references an unknown namespace")
    for gold in view.task_gold:
        if gold.get("execution_namespace_key_hash") not in namespace_hashes:
            failures.append(f"task Gold {gold.get('task_gold_id')} references an unknown namespace")

    return {
        "status": "PASS" if not failures else "FAIL",
        "namespace_key_count": len(namespace_raw),
        "context_count": len(context_raw),
        "current_code_hash": current_code_hash,
        "namespace_hashes": {split: key.namespace_hash for split, key in namespaces.items()},
        "failures": failures,
    }


def validate_fixture(project_root: str | Path, view: DatasetView) -> dict[str, Any]:
    root = Path(project_root).resolve()
    path = root / "work" / "B" / "fixtures" / "deterministic_retrieval_fixture.development.json"
    payload = _read_json(path)
    failures: list[str] = []
    if not isinstance(payload, dict):
        return {"status": "FAIL", "failures": ["fixture must be a JSON object"]}
    results = payload.get("results", [])
    if not isinstance(results, list):
        results = []
        failures.append("fixture results must be a list")
    development_task_ids = {
        str(item.get("task_instance_id"))
        for item in view.tasks
        if item.get("split") == "DEVELOPMENT"
    }
    forbidden_keys = {
        "technical_citation",
        "technical_citation_id",
        "retrieval_result",
        "retrieval_run_id",
        "recall",
        "mrr",
        "metric_result",
        "score",
    }

    def scan_keys(value: Any, path_parts: tuple[str, ...] = ()) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key.casefold() in forbidden_keys:
                    failures.append(f"fixture forbidden runtime/metric key: {'/'.join((*path_parts, key))}")
                scan_keys(item, (*path_parts, key))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                scan_keys(item, (*path_parts, str(index)))

    scan_keys({key: value for key, value in payload.items() if key != "forbidden_outputs"})
    for index, result in enumerate(results):
        if result.get("usage") != "DEVELOPMENT_ONLY":
            failures.append(f"fixture result[{index}] usage is not DEVELOPMENT_ONLY")
        if str(result.get("task_instance_id")) not in development_task_ids:
            failures.append(f"fixture result[{index}] references non-development task")
        ref = result.get("fixture_evidence_ref")
        if not isinstance(ref, str) or not ref.startswith("fixture-evidence:"):
            failures.append(f"fixture result[{index}] lacks fixture_evidence_ref")
    if any(item.get("split") == "SEALED_HOLDOUT" for item in results if isinstance(item, dict)):
        failures.append("fixture contains holdout data")
    return {
        "status": "PASS" if not failures else "FAIL",
        "fixture_result_count": len(results),
        "development_only": True,
        "produces_technical_citation": False,
        "produces_retrieval_metrics": False,
        "failures": failures,
    }


def _sensitive_holdout_strings(view: DatasetView) -> set[str]:
    sensitive: set[str] = set()
    categories = (
        "root_scenarios",
        "case_graphs",
        "task_instances",
        "customer_messages",
        "expected_customer_attributions",
        "gold_customer_attributions",
        "expected_field_annotations",
        "gold_field_annotations",
        "field_annotation_extensions",
        "expected_task_gold",
        "expected_retrieval_evidence",
        "expected_limitation_decisions",
    )

    policy_keys = {
        "private_type",
        "shared_type",
        "provenance",
        "runtime_citation_requirement",
        "runtime_requirement",
        "allowed_answer_policy",
        "version_condition",
        "review_status",
        "purpose",
        "forbidden_commitments",
        "limitations",
        "next_actions",
        "questions_to_confirm",
        "assumptions",
        "notes",
    }

    def collect(value: Any, key: str | None = None, target: set[str] | None = None) -> None:
        destination = sensitive if target is None else target
        if isinstance(value, dict):
            for child_key, item in value.items():
                collect(item, str(child_key), destination)
        elif isinstance(value, list):
            for item in value:
                collect(item, key, destination)
        elif isinstance(value, str):
            if UUID_RE.fullmatch(value):
                destination.add(value)
            elif key and key.endswith("sha256"):
                destination.add(value)
            elif len(value) >= 24 and key not in policy_keys:
                destination.add(value)

    # Keep only holdout values absent from the corresponding development
    # records. Shared templates, schema labels, and public document wording
    # are intentionally present in both partitions and are not leakage.
    development_strings: set[str] = set()
    root_split = {
        str(item.get("root_scenario_id")): str(item.get("split"))
        for item in view.roots
    }
    case_split = {
        str(item.get("case_id")): str(item.get("split"))
        for item in view.roots
    }
    aliases = {
        "root_scenarios": "roots",
        "case_graphs": "graphs",
        "task_instances": "tasks",
        "customer_messages": "messages",
        "expected_customer_attributions": "attributions",
        "gold_customer_attributions": "attributions",
        "expected_field_annotations": "annotations",
        "gold_field_annotations": "annotations",
        "field_annotation_extensions": "annotation_extensions",
        "expected_task_gold": "task_gold",
        "expected_retrieval_evidence": "retrieval_evidence",
        "expected_limitation_decisions": "limitations",
    }

    def record_split(item: Mapping[str, Any]) -> str | None:
        direct = item.get("split")
        if direct is not None:
            return str(direct)
        root_id = item.get("root_scenario_id")
        if root_id is not None and str(root_id) in root_split:
            return root_split[str(root_id)]
        case_id = item.get("case_id")
        if case_id is not None and str(case_id) in case_split:
            return case_split[str(case_id)]
        return None

    # Field annotations carry their root binding in the extension sidecar;
    # recover that split before scanning the shared annotation projection.
    annotation_split = {
        str(item.get("expected_field_annotation_id")): root_split.get(
            str(item.get("root_scenario_id"))
        )
        for item in view.annotation_extensions
    }

    def development_record(item: Mapping[str, Any]) -> bool:
        split = record_split(item)
        if split is None:
            annotation_id = item.get("expected_field_annotation_id")
            split = annotation_split.get(str(annotation_id))
        return split == "DEVELOPMENT"

    for category in categories:
        holdout_records = _sealed_records(view.sealed_payload, category)
        if holdout_records:
            collect(holdout_records)
        source = getattr(view, aliases.get(category, category), ())
        development_records = [
            item
            for item in source
            if isinstance(item, Mapping) and development_record(item)
        ]
        if development_records:
            collect(development_records, target=development_strings)

    # Include every development-side record, including root/task projections
    # whose values are not part of the category-specific public list above.
    for source in (
        view.roots,
        view.graphs,
        view.tasks,
        view.messages,
        view.attributions,
        view.annotations,
        view.annotation_extensions,
        view.task_gold,
        view.retrieval_evidence,
        view.limitations,
    ):
        collect(
            [item for item in source if isinstance(item, Mapping) and development_record(item)],
            target=development_strings,
        )
    shared_knowledge_strings: set[str] = set()
    collect(view.document_facts, target=shared_knowledge_strings)
    for fact in view.document_facts:
        for key in ("document_id", "document_version_id", "statement", "statement_sha256"):
            value = fact.get(key)
            if isinstance(value, str):
                shared_knowledge_strings.add(value)
    collect(view.sealed_payload.get("gold_document_facts", []), target=shared_knowledge_strings)
    collect(
        view.sealed_payload.get("lifecycle_expectations", []),
        target=shared_knowledge_strings,
    )
    sealed_manifest = view.sealed_payload.get("evaluation_dataset_manifest", {})
    if isinstance(sealed_manifest, dict):
        collect(sealed_manifest.get("assets", []), target=shared_knowledge_strings)
    sensitive -= shared_knowledge_strings
    sensitive -= development_strings
    # Root and task IDs are explicitly allowed in a no-Gold split summary.
    sensitive -= {
        str(item.get("root_scenario_id"))
        for item in _sealed_records(view.sealed_payload, "root_scenarios")
    }
    sensitive -= {
        str(item.get("task_instance_id"))
        for item in _sealed_records(view.sealed_payload, "task_instances")
    }
    return sensitive


def validate_public_holdout_absence(project_root: str | Path, view: DatasetView) -> dict[str, Any]:
    """Prove that public B JSON contains no split-specific holdout payload."""

    root = Path(project_root).resolve()
    b_root = root / "work" / "B"
    public_files = sorted(
        {
            *list((b_root / "manifests").glob("*.json")),
            *list((b_root / "generated" / "development").glob("*.json")),
            *list((b_root / "fixtures").glob("*.json")),
        }
    )
    sensitive = _sensitive_holdout_strings(view)

    def string_values(value: Any) -> set[str]:
        if isinstance(value, dict):
            return {item for child in value.values() for item in string_values(child)}
        if isinstance(value, list):
            return {item for child in value for item in string_values(child)}
        return {value} if isinstance(value, str) else set()

    violations: list[dict[str, Any]] = []
    scanned = 0
    for path in public_files:
        payload = _read_json(path)
        scanned += 1
        relative = path.relative_to(root).as_posix()
        if path.name == "split_manifest.json":
            # Root/task IDs and aggregate counts are the only holdout-specific data allowed here.
            holdout = (
                payload.get("SEALED_HOLDOUT", payload.get("sealed_holdout", {}))
                if isinstance(payload, dict)
                else {}
            )
            forbidden_summary_keys = set(holdout) - {
                "root_ids",
                "task_ids",
                "counts",
                "gold_access",
                "payload_sha256",
                "sealed_payload_sha256",
                "seal_status",
                "content_commitment_hash",
                "holdout_content_commitment_hash",
            }
            if forbidden_summary_keys:
                violations.append(
                    {
                        "file": relative,
                        "reason": "split manifest has forbidden holdout summary keys",
                        "keys": sorted(forbidden_summary_keys),
                    }
                )
            continue
        if path.name in {
            "evaluation_dataset_manifest.sealed.json",
            "sealed_manifest.summary.json",
        }:
            if not isinstance(payload, dict):
                violations.append({"file": relative, "reason": "sealed summary is not an object"})
                continue
            forbidden_keys = set(payload) - PUBLIC_HOLDOUT_SUMMARY_KEYS
            if forbidden_keys:
                violations.append(
                    {
                        "file": relative,
                        "reason": "sealed public summary contains non-summary fields",
                        "keys": sorted(forbidden_keys),
                    }
                )
            continue
        serialized = canonical_json(payload)
        public_values = string_values(payload)
        matched = [
            item
            for item in sensitive
            if item
            and item in serialized
            and (
                UUID_RE.fullmatch(item)
                or SHA_RE.fullmatch(item)
                or item not in public_values
            )
        ]
        if matched:
            violations.append(
                {
                    "file": relative,
                    "reason": "sealed holdout content found in public JSON",
                    "matched_value_blind_refs": [sha256_json(item) for item in sorted(matched)],
                    "match_count": len(matched),
                }
            )
        if isinstance(payload, list) and any(
            isinstance(item, dict) and item.get("split") == "SEALED_HOLDOUT" for item in payload
        ):
            violations.append(
                {"file": relative, "reason": "public record array contains SEALED_HOLDOUT objects"}
            )
    return {
        "status": "PASS" if not violations else "FAIL",
        "public_json_file_count": scanned,
        "sealed_sensitive_value_count": len(sensitive),
        "holdout_gold_record_count_in_public_json": sum(
            item.get("match_count", 1) for item in violations
        ),
        "violations": violations,
    }


def validate_allowlist(
    project_root: str | Path,
    *,
    require_all_allowlisted: bool = False,
) -> dict[str, Any]:
    """Compare B's exact allowlist to disk in both directions.

    Cache files are never persistent allowlist members.  Cache directories
    below ``work/B`` are ignored only when an explicit ``/**`` rule covers
    them; caches below source/tests are a hard failure.
    """

    root = Path(project_root).resolve()
    allowlist_path = root / "docs" / "b" / "b-file-allowlist-v1.0.json"
    # Unit/integration fixtures build an isolated project root containing only
    # generated B outputs.  The authoritative project root must carry the
    # allowlist, but an isolated fixture can validate data semantics without
    # copying the control-plane document.  Keep the strict mode available for
    # the real build gate via ``require_all_allowlisted=True``.
    if not allowlist_path.is_file():
        if require_all_allowlisted:
            return {
                "status": "FAIL",
                "require_all_allowlisted": True,
                "allowlisted_exact_file_count": 0,
                "allowlisted_glob_count": 0,
                "required_after_build_file_count": 0,
                "actual_persistent_file_count": 0,
                "unlisted_persistent_files": [],
                "missing_allowlisted_files": [],
                "prohibited_source_test_cache_files": [],
                "ignored_explicit_work_cache_files": [],
                "failures": ["allowlist JSON is missing"],
            }
        return {
            "status": "PASS",
            "applicable": False,
            "require_all_allowlisted": False,
            "allowlisted_exact_file_count": 0,
            "allowlisted_glob_count": 0,
            "required_after_build_file_count": 0,
            "actual_persistent_file_count": 0,
            "unlisted_persistent_files": [],
            "missing_allowlisted_files": [],
            "prohibited_source_test_cache_files": [],
            "ignored_explicit_work_cache_files": [],
            "failures": [],
        }
    payload = _read_json(allowlist_path)
    if not isinstance(payload, dict) or not isinstance(payload.get("owned_paths"), list):
        return {"status": "FAIL", "failures": ["allowlist JSON is invalid"]}
    patterns = [str(item).replace("\\", "/") for item in payload["owned_paths"]]
    ephemeral_patterns = [
        str(item).replace("\\", "/") for item in payload.get("ephemeral_patterns", [])
    ]
    required_after_build = {
        str(item).replace("\\", "/")
        for item in payload.get("required_after_build", payload["owned_paths"])
    }
    exact = {item for item in patterns if not any(char in item for char in "*?[")}
    globs = [item for item in patterns if item not in exact] + ephemeral_patterns
    roots = (
        root / "src" / "assistant" / "module_b",
        root / "tests" / "module_b",
        root / "docs" / "b",
        root / "work" / "B",
    )
    actual: set[str] = set()
    cache_source_files: list[str] = []
    ignored_cache_files: list[str] = []
    for owned_root in roots:
        if not owned_root.exists():
            continue
        for path in owned_root.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            parts = {part.casefold() for part in path.parts}
            is_cache = (
                "__pycache__" in parts
                or ".pytest_cache" in parts
                or path.suffix.casefold() in {".pyc", ".pyo"}
            )
            if is_cache:
                if relative.startswith("src/assistant/module_b/") or relative.startswith("tests/module_b/"):
                    cache_source_files.append(relative)
                elif any(
                    fnmatch.fnmatchcase(relative, pattern)
                    for pattern in ephemeral_patterns
                ):
                    ignored_cache_files.append(relative)
                else:
                    actual.add(relative)
                continue
            if any(fnmatch.fnmatchcase(relative, pattern) for pattern in globs):
                ignored_cache_files.append(relative)
                continue
            actual.add(relative)

    unlisted = sorted(
        item
        for item in actual
        if item not in exact and not any(fnmatch.fnmatchcase(item, pattern) for pattern in globs)
    )
    missing = sorted(item for item in required_after_build if not (root / Path(item)).is_file())
    failures: list[str] = []
    if unlisted:
        failures.append("persistent B files exist outside owned_paths")
    if cache_source_files:
        failures.append("source/tests contain prohibited cache files")
    if require_all_allowlisted and missing:
        failures.append("allowlisted persistent files are missing")
    return {
        "status": "PASS" if not failures else "FAIL",
        "require_all_allowlisted": require_all_allowlisted,
        "allowlisted_exact_file_count": len(exact),
        "allowlisted_glob_count": len(globs),
        "required_after_build_file_count": len(required_after_build),
        "actual_persistent_file_count": len(actual),
        "unlisted_persistent_files": unlisted,
        "missing_allowlisted_files": missing,
        "prohibited_source_test_cache_files": sorted(cache_source_files),
        "ignored_explicit_work_cache_files": sorted(ignored_cache_files),
        "failures": failures,
    }


def _manifest_assets(root: Path, metadata_records: Sequence[Mapping[str, Any]]) -> tuple[AssetManifestEntry, ...]:
    media_types = {
        "PDF": "application/pdf",
        "DOCX": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "MARKDOWN": "text/markdown",
        "TEXT": "text/plain",
    }
    assets: list[AssetManifestEntry] = []
    for raw in metadata_records:
        metadata = SourceDocumentMetadata.model_validate(
            _strip_for_model(SourceDocumentMetadata, raw)
        )
        path = root / str(metadata.source_ref)
        assets.append(
            AssetManifestEntry(
                asset_id=deterministic_uuid("B", "asset", str(metadata.document_version_id)),
                logical_ref=str(metadata.source_ref),
                asset_type=metadata.document_format.value,
                sha256=metadata.source_sha256,
                size_bytes=path.stat().st_size,
                media_type=media_types[metadata.document_format.value],
                license="SELF_AUTHORED_SYNTHETIC",
                provenance="FICTIONAL_DEMO; B generated",
                data_classification=metadata.data_classification,
            )
        )
    return tuple(assets)


def _sorted_records(records: Iterable[Mapping[str, Any]], *id_keys: str) -> list[Mapping[str, Any]]:
    def sort_key(item: Mapping[str, Any]) -> tuple[str, ...]:
        return tuple(str(item.get(key, "")) for key in id_keys)

    return sorted(records, key=sort_key)


def dataset_hash_material(project_root: str | Path, view: DatasetView) -> dict[str, Any]:
    """Return the complete, order-independent Gold/corpus hash material."""

    root = Path(project_root).resolve()
    manifests = root / "work" / "B" / "manifests"
    metadata = _as_records(
        _read_json(manifests / "source_document_metadata.json"),
        "source_document_metadata",
    )
    lifecycle = _as_records(
        _read_json(manifests / "lifecycle_expectations.json"),
        "lifecycle_expectations",
    )
    return {
        "hash_profile": "B-DATASET-CONTENT-v1; canonical JSON; arrays sorted by stable IDs",
        "source_document_metadata": _sorted_records(metadata, "document_version_id"),
        "gold_document_facts": _sorted_records(view.document_facts, "gold_fact_id"),
        "root_scenarios": _sorted_records(view.roots, "root_scenario_id"),
        "case_graphs": _sorted_records(view.graphs, "case_graph_id"),
        "task_instances": _sorted_records(view.tasks, "task_instance_id"),
        "customer_messages": _sorted_records(view.messages, "message_id"),
        "expected_customer_attributions": _sorted_records(
            view.attributions, "expected_customer_attribution_id"
        ),
        "expected_field_annotations": _sorted_records(
            view.annotations, "expected_field_annotation_id"
        ),
        "field_annotation_extensions": _sorted_records(
            view.annotation_extensions, "expected_field_annotation_id"
        ),
        "expected_task_gold": _sorted_records(view.task_gold, "task_gold_id"),
        "expected_retrieval_evidence": _sorted_records(
            view.retrieval_evidence, "expected_retrieval_evidence_id"
        ),
        "expected_limitation_decisions": _sorted_records(
            view.limitations, "expected_limitation_decision_id"
        ),
        "lifecycle_expectations": _sorted_records(
            lifecycle, "document_version_id", "scenario_label", "source_ref"
        ),
    }


def build_evaluation_dataset_manifest(
    project_root: str | Path,
    view: DatasetView,
    leakage: Mapping[str, Any],
) -> EvaluationDatasetManifest:
    """Build and instantiate the complete frozen A manifest in memory."""

    root = Path(project_root).resolve()
    manifests = root / "work" / "B" / "manifests"
    split_manifest = _read_json(manifests / "split_manifest.json")
    metadata_records = _as_records(
        _read_json(manifests / "source_document_metadata.json"),
        "source_document_metadata",
    )
    roots = tuple(
        RootScenario.model_validate(_strip_for_model(RootScenario, item)) for item in view.roots
    )
    graphs = tuple(
        CaseGraph.model_validate(_strip_for_model(CaseGraph, item)) for item in view.graphs
    )
    leakage_model = LeakageReport(
        report_id=deterministic_uuid("B", "leakage-report", split_manifest["dataset_version"]),
        root_scenario_overlap=leakage["atomic_overlap_counts"]["root_scenario_overlap"],
        fact_family_overlap=leakage["atomic_overlap_counts"]["fact_family_overlap"],
        source_lineage_overlap=leakage["atomic_overlap_counts"]["source_lineage_overlap"],
        template_family_overlap=leakage["atomic_overlap_counts"]["template_family_overlap"],
        unresolved_near_duplicates=leakage["unresolved_near_duplicates"],
        status=leakage["status"],
    )
    base = {
        "dataset_id": split_manifest["dataset_id"],
        "dataset_version": split_manifest["dataset_version"],
        "annotation_version": split_manifest["annotation_version"],
        "assets": _manifest_assets(root, metadata_records),
        "root_scenarios": roots,
        "case_graphs": graphs,
        "planned_task_counts_by_type": PLANNED_TASK_TYPE_COUNTS,
        "leakage_report": leakage_model,
        "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
    }
    dataset_sha256 = sha256_json(dataset_hash_material(root, view))
    manifest = EvaluationDatasetManifest(
        **base,
        dataset_sha256=dataset_sha256,
        hash_status="COMPUTED",
        release_status=DatasetReleaseStatus.FROZEN,
    )
    sealed_manifest = view.sealed_payload.get("evaluation_dataset_manifest")
    if sealed_manifest is None:
        raise ValueError("sealed payload lacks the complete A EvaluationDatasetManifest")
    declared = EvaluationDatasetManifest.model_validate(sealed_manifest)
    if declared.dataset_sha256 != dataset_sha256:
        raise ValueError("declared EvaluationDatasetManifest dataset_sha256 is not reproducible")

    def comparable(value: EvaluationDatasetManifest) -> dict[str, Any]:
        payload = value.model_dump(mode="json", exclude={"dataset_sha256"})
        payload["assets"] = sorted(payload["assets"], key=lambda item: item["asset_id"])
        payload["root_scenarios"] = sorted(
            payload["root_scenarios"], key=lambda item: item["root_scenario_id"]
        )
        payload["case_graphs"] = sorted(
            payload["case_graphs"], key=lambda item: item["case_graph_id"]
        )
        return payload

    declared_comparable = comparable(declared)
    rebuilt_comparable = comparable(manifest)
    if declared_comparable != rebuilt_comparable:
        differing_fields = sorted(
            key
            for key in declared_comparable
            if declared_comparable.get(key) != rebuilt_comparable.get(key)
        )
        raise ValueError(
            "declared EvaluationDatasetManifest content differs from rebuilt A model; "
            f"fields={differing_fields}"
        )
    return manifest


def _checksum_report(root: Path, view: DatasetView, manifest: EvaluationDatasetManifest | None) -> dict[str, Any]:
    manifests = root / "work" / "B" / "manifests"
    split = _read_json(manifests / "split_manifest.json")
    split_basis = {key: value for key, value in split.items() if key != "split_manifest_hash"}
    split_hash = sha256_json(split_basis)
    gold_hash = sha256_json(
        {
            "expected_customer_attributions": view.attributions,
            "expected_field_annotations": view.annotations,
            "field_annotation_extensions": view.annotation_extensions,
            "expected_task_gold": view.task_gold,
            "expected_retrieval_evidence": view.retrieval_evidence,
            "expected_limitation_decisions": view.limitations,
        }
    )
    seal_path = root / "work" / "B" / "seal" / "sealed_gold.payload.json"
    return {
        "status": "PASS" if split.get("split_manifest_hash") == split_hash and manifest else "FAIL",
        "hash_profile": "module_b canonical JSON, UTF-8, sorted keys, NFC/LF strings",
        "split_manifest_hash": split_hash,
        "split_manifest_declared_hash": split.get("split_manifest_hash"),
        "gold_bundle_hash": gold_hash,
        "evaluation_dataset_hash": manifest.dataset_sha256 if manifest else None,
        "evaluation_dataset_hash_basis": "complete corpus and Gold bundle; canonical arrays sorted by stable IDs",
        "sealed_payload_file_sha256": file_sha256(seal_path) if seal_path.is_file() else None,
        "formal_metrics": "NOT_RUN",
    }


def validate_all(
    project_root: str | Path,
    *,
    sealed_payload: Mapping[str, Any] | None = None,
    write_reports: bool = True,
    require_all_allowlisted: bool = False,
) -> dict[str, Any]:
    """Run every B pre-seal freeze gate and optionally write public reports."""

    root = Path(project_root).resolve()
    failures: list[dict[str, str]] = []
    try:
        view = load_dataset_view(root, sealed_payload=sealed_payload)
    except Exception as exc:
        result = {
            "report_version": REPORT_VERSION,
            "report_kind": "POST_SEAL_ACCESS_AUDIT",
            "status": "FAIL",
            "formal_metrics": "NOT_RUN",
            "failures": [{"code": "DATASET_LOAD_FAILED", "detail": str(exc)}],
        }
        if write_reports:
            # A post-seal access guard is an audit event, not a new pre-seal
            # validation result.  Keep the canonical PASS receipt immutable;
            # callers must use the sidecar to inspect this expected refusal.
            _write_json(
                root / "work" / "B" / "reports" / "validation_post_seal_report.json",
                result,
            )
        return result

    try:
        shared_counts = _validate_shared_schemas(root, view, failures)
    except Exception as exc:
        shared_counts = {}
        failures.append({"code": "SHARED_SCHEMA_VALIDATION_FAILED", "detail": str(exc)})

    count_report, chain_report = validate_counts_and_chains(view)
    leakage_report = compute_leakage_report(view)
    locator_report = verify_document_locators(root)
    reference_report = validate_reference_closure(view)
    namespace_report = validate_namespaces(root, view)
    fixture_report = validate_fixture(root, view)
    public_scan_report = validate_public_holdout_absence(root, view)
    allowlist_report = validate_allowlist(
        root, require_all_allowlisted=require_all_allowlisted
    )

    component_reports = {
        "counts": count_report,
        "chains": chain_report,
        "leakage": leakage_report,
        "locators": locator_report,
        "references": reference_report,
        "namespaces": namespace_report,
        "fixture": fixture_report,
        "public_holdout_scan": public_scan_report,
        "allowlist": allowlist_report,
    }
    for label, report in component_reports.items():
        if report.get("status") != "PASS":
            detail = f"{label} status={report.get('status')}"
            if label == "leakage":
                detail += f" atomic_overlap_counts={report.get('atomic_overlap_counts', {})}"
            failures.append(
                {
                    "code": f"{label.upper()}_GATE_FAILED",
                    "detail": detail,
                }
            )

    manifest: EvaluationDatasetManifest | None = None
    if leakage_report["status"] == "PASS":
        try:
            manifest = build_evaluation_dataset_manifest(root, view, leakage_report)
        except Exception as exc:
            failures.append(
                {"code": "A_EVALUATION_DATASET_MANIFEST_INVALID", "detail": str(exc)}
            )
    else:
        failures.append(
            {
                "code": "A_EVALUATION_DATASET_MANIFEST_NOT_FROZEN",
                "detail": "computed leakage failure prohibits A FROZEN manifest",
            }
        )

    checksum_report = _checksum_report(root, view, manifest)
    if checksum_report["status"] != "PASS":
        failures.append(
            {"code": "CHECKSUM_GATE_FAILED", "detail": "split or dataset hash validation failed"}
        )

    result = {
        "report_version": REPORT_VERSION,
        "status": "PASS" if not failures else "FAIL",
        "dataset_state": "DESIGN_FROZEN_DATA_BUILT" if not failures else "NOT_FROZEN",
        "formal_metrics": "NOT_RUN",
        "a_evaluation_dataset_manifest": {
            "status": "PASS" if manifest else "FAIL",
            "shared_type": "assistant.contracts.datasets.EvaluationDatasetManifest",
            "release_status": manifest.release_status.value if manifest else None,
            "dataset_sha256": manifest.dataset_sha256 if manifest else None,
            "root_scenario_count": len(manifest.root_scenarios) if manifest else None,
            "task_instance_count": (
                sum(len(graph.task_instances) for graph in manifest.case_graphs)
                if manifest
                else None
            ),
        },
        "shared_schema_instance_counts": shared_counts,
        "component_statuses": {
            label: report.get("status") for label, report in component_reports.items()
        }
        | {"checksums": checksum_report["status"]},
        "failures": failures,
    }
    if write_reports:
        reports = root / "work" / "B" / "reports"
        _write_json(reports / "validation_report.json", result)
        _write_json(reports / "leakage_report.json", leakage_report)
        _write_json(reports / "count_report.json", count_report)
        _write_json(reports / "chain_report.json", chain_report)
        _write_json(reports / "checksum_report.json", checksum_report)
        _write_json(reports / "allowlist_report.json", allowlist_report)
    return result


def require_valid(
    project_root: str | Path,
    *,
    sealed_payload: Mapping[str, Any] | None = None,
    write_reports: bool = True,
    require_all_allowlisted: bool = False,
) -> dict[str, Any]:
    """Run validation and raise unless every B freeze gate passes."""

    result = validate_all(
        project_root,
        sealed_payload=sealed_payload,
        write_reports=write_reports,
        require_all_allowlisted=require_all_allowlisted,
    )
    if result["status"] != "PASS":
        codes = ", ".join(item["code"] for item in result.get("failures", []))
        raise ValidationFailure(f"Module B validation failed: {codes}")
    return result


def validate_dataset(
    project_root: str | Path,
    *,
    sealed_payload: Mapping[str, Any] | None = None,
    write_reports: bool = True,
    require_all_allowlisted: bool = False,
) -> dict[str, Any]:
    """Compatibility entry point for B build, CLI, and sealing callers.

    Pre-seal callers pass the in-memory ``sealed_gold`` returned by
    :func:`generation.generate_all`; the validator never requires a public or
    pre-existing sealed payload file.
    """

    return validate_all(
        project_root,
        sealed_payload=sealed_payload,
        write_reports=write_reports,
        require_all_allowlisted=require_all_allowlisted,
    )


__all__ = (
    "DatasetView",
    "MIN_DUPLICATE_TEXT_LENGTH",
    "NEAR_DUPLICATE_SHINGLE_SIZE",
    "NEAR_DUPLICATE_THRESHOLD",
    "ValidationFailure",
    "build_evaluation_dataset_manifest",
    "compute_leakage_report",
    "dataset_hash_material",
    "load_dataset_view",
    "near_duplicate_score",
    "normalize_leakage_text",
    "require_valid",
    "validate_all",
    "validate_allowlist",
    "validate_counts_and_chains",
    "validate_dataset",
    "validate_fixture",
    "validate_namespaces",
    "validate_public_holdout_absence",
    "validate_reference_closure",
    "verify_document_locators",
)
