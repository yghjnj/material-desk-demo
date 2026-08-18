"""Small, side-effect-free D baseline.

The baseline intentionally uses only customer messages and shared A contracts.
It does not retrieve documents, call a model, or produce technical evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import re
from typing import Iterable
from uuid import UUID, NAMESPACE_URL, uuid5

from assistant.contracts.base import sha256_text
from assistant.contracts.customer_locators import RuntimeCustomerLocator
from assistant.contracts.customers import (
    CustomerMessage,
    CustomerNormalizationMap,
    CustomerNormalizationMappingSegment,
)
from assistant.contracts.enums import (
    AttributionStatus,
    Comparator,
    CustomerMappingKind,
    FieldStatus,
    NormalizationMapStatus,
    NormalizationStatus,
    RequirementStatus,
    ValueState,
)
from assistant.contracts.evidence import CustomerAttribution
from assistant.contracts.measurements import Measurement
from assistant.contracts.requirements import CustomerRequirement, FieldAnnotation
from assistant.contracts.runtime import ExecutionNamespaceKey


FIELD_PATHS = (
    "/application_scenario",
    "/material_or_product",
    "/substrate",
    "/performance_indicators",
    "/test_standards",
    "/process",
    "/quantity_and_usage",
    "/budget_and_currency",
    "/delivery_and_date",
    "/region",
    "/compliance",
)

_UUID_NAMESPACE = NAMESPACE_URL
_MEASUREMENT_RE = re.compile(
    r"(?P<label>固含|solid\s+content|黏度|粘度|viscosity|温度|temperature|"
    r"膜厚|film\s+thickness|拉伸强度|tensile(?:\s+strength)?)"
    r"\s*(?P<op>>=|<=|>|<|≥|≤|不少于|不低于|不超过|至多)?\s*"
    r"(?P<number>\d+(?:\.\d+)?)\s*(?P<unit>%|wt%|vol%|cP|cSt|mPa\s*[·.]?\s*s|Pa\s*[·.]?\s*s|"
    r"mm2/s|mm²/s|°?C|K|°?F|um|μm|µm|mm|MPa|kPa|GPa)",
    re.IGNORECASE,
)
_TEXT_PATTERNS = {
    "/material_or_product": re.compile(
        r"(?:产品|材料|product|material)\s*(?:是|为|:|：)?\s*([^,，。.;；\n]+)", re.I
    ),
    "/substrate": re.compile(
        r"(?:基材|substrate)\s*(?:是|为|:|：|on)?\s*([^,，。.;；\n]+)", re.I
    ),
    "/process": re.compile(
        r"(?:工艺|施工方式|application\s+method|process)\s*(?:是|为|:|：)?\s*([^,，。.;；\n]+)", re.I
    ),
    "/region": re.compile(
        r"(?:地区|区域|region|market)\s*(?:是|为|:|：)?\s*([^,，。.;；\n]+)", re.I
    ),
    "/compliance": re.compile(r"\b(?:RoHS|REACH|UL94(?:\s*[A-Z]-?\d)?)\b", re.I),
    "/test_standards": re.compile(
        r"\b(?:ASTM|ISO|GB/?T|EN)\s*[A-Z]?\s*\d+(?:[.-]\d+)*\b", re.I
    ),
    "/delivery_and_date": re.compile(
        r"(?:交期|交货日期|delivery|deliver)\s*(?:是|为|:|：)?\s*(\d{4}-\d{2}-\d{2})", re.I
    ),
    "/quantity_and_usage": re.compile(
        r"(?:数量|用量|quantity|amount)\s*(?:是|为|:|：)?\s*(\d+(?:\.\d+)?)\s*(kg|g|吨|t|L|mL)\b", re.I
    ),
    "/budget_and_currency": re.compile(
        r"(?:预算|budget)\s*(?:是|为|:|：)?\s*(?:([A-Z]{3}|RMB|人民币|¥|\$)\s*)?(\d+(?:\.\d+)?)", re.I
    ),
}
_SCENARIO_RE = re.compile(
    r"(?:用于|应用于|application|use(?:d)?\s+for)\s*([^,，。.;；\n]+)", re.I
)
_PRODUCT_NAME_RE = re.compile(
    r"(?:我们考虑|we\s+(?:need|consider)|需要|need)\s+([A-Za-z][A-Za-z0-9_-]{2,})",
    re.I,
)


@dataclass(frozen=True)
class DExtractionContext:
    """D-private call context; the namespace object is A's shared type."""

    execution_namespace_key: ExecutionNamespaceKey
    normalizer_version: str = "d-nfc-identity-v1"
    revision: int = 1
    extracted_at: datetime | None = None


@dataclass(frozen=True)
class _Candidate:
    field_path: str
    message: CustomerMessage
    start: int
    end: int
    raw: str
    known_value: str
    measurement: Measurement | None = None
    normalization_status: NormalizationStatus = NormalizationStatus.NOT_REQUESTED


def _id(context: DExtractionContext, *parts: object) -> UUID:
    return uuid5(_UUID_NAMESPACE, ":".join((str(context.execution_namespace_key.namespace_hash), *(str(p) for p in parts))))


def _active_messages(messages: Iterable[CustomerMessage]) -> tuple[CustomerMessage, ...]:
    ordered = tuple(sorted(messages, key=lambda item: (item.sequence_no, str(item.message_id))))
    if not ordered:
        raise ValueError("D requires at least one CustomerMessage")
    case_ids = {item.case_id for item in ordered}
    if len(case_ids) != 1:
        raise ValueError("D messages must belong to one case")
    by_id = {item.message_id: item for item in ordered}
    if len(by_id) != len(ordered) or len({item.sequence_no for item in ordered}) != len(ordered):
        raise ValueError("D message IDs and sequence numbers must be unique")
    for message in ordered:
        parent = message.supersedes_message_id
        if parent is not None:
            if parent not in by_id or by_id[parent].sequence_no >= message.sequence_no:
                raise ValueError("invalid customer message supersedes link")
    superseded = {item.supersedes_message_id for item in ordered if item.supersedes_message_id is not None}
    return tuple(item for item in ordered if item.message_id not in superseded)


def _normalization_map(message: CustomerMessage, version: str) -> CustomerNormalizationMap:
    length = len(message.text)
    segments = tuple(
        CustomerNormalizationMappingSegment(
            normalized_start=index,
            normalized_end=index + 1,
            original_start=index,
            original_end=index + 1,
            mapping_kind=CustomerMappingKind.ONE_TO_ONE,
        )
        for index in range(length)
    )
    verification_hash = sha256_text(
        f"{message.message_id}|{message.text_sha256}|{sha256_text(message.text)}|{version}|{length}"
    )
    return CustomerNormalizationMap(
        message_id=message.message_id,
        original_message_sha256=message.text_sha256,
        normalized_text_sha256=sha256_text(message.text),
        normalizer_version=version,
        original_code_point_length=length,
        normalized_code_point_length=length,
        mapping_segments=segments,
        verification_hash=verification_hash,
        status=NormalizationMapStatus.VERIFIED,
    )


def _decimal(value: str) -> str:
    try:
        result = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError("invalid decimal") from exc
    if not result.is_finite():
        raise ValueError("non-finite decimal")
    result = result.normalize()
    text = format(result, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in {"", "-0"} else text


def _measurement(match: re.Match[str]) -> tuple[str, Measurement, NormalizationStatus]:
    raw_number = match.group("number")
    raw_unit = match.group("unit").replace("·", ".").replace(" ", "")
    label = match.group("label").lower()
    comparator_text = match.group("op") or "="
    comparator = {
        ">": Comparator.GT, ">=": Comparator.GTE, "≥": Comparator.GTE,
        "<": Comparator.LT, "<=": Comparator.LTE, "≤": Comparator.LTE,
        "不少于": Comparator.GTE, "不低于": Comparator.GTE, "不超过": Comparator.LTE,
    }.get(comparator_text, Comparator.EQ)
    value = _decimal(raw_number)
    unit_key = raw_unit.lower()
    normalized = value
    unit_code = raw_unit
    dimension = "unknown"
    normalized_unit = raw_unit
    status = NormalizationStatus.NORMALIZED
    if unit_key in {"%", "wt%", "vol%"}:
        normalized = _decimal(str(Decimal(value) / Decimal("100")))
        unit_code, normalized_unit, dimension = "1", "1", "fraction"
        if unit_key == "%" and label in {"固含", "solid content"}:
            status = NormalizationStatus.UNRESOLVED
            normalized = None
    elif unit_key in {"cp", "mpa.s"}:
        unit_code, normalized_unit, dimension = "mPa.s", "mPa.s", "dynamic_viscosity"
        if unit_key == "cp":
            normalized = value
    elif unit_key == "pa.s":
        unit_code, normalized_unit, dimension = "Pa.s", "mPa.s", "dynamic_viscosity"
        normalized = _decimal(str(Decimal(value) * Decimal("1000")))
    elif unit_key in {"cst", "mm2/s", "mm²/s"}:
        unit_code, normalized_unit, dimension = "mm2/s", "mm2/s", "kinematic_viscosity"
    elif unit_key in {"c", "°c"}:
        unit_code, normalized_unit, dimension = "Cel", "Cel", "temperature"
    elif unit_key == "k":
        unit_code, normalized_unit, dimension = "K", "Cel", "temperature"
        normalized = _decimal(str(Decimal(value) - Decimal("273.15")))
    elif unit_key in {"f", "°f"}:
        unit_code, normalized_unit, dimension = "[degF]", "Cel", "temperature"
        normalized = _decimal(str((Decimal(value) - Decimal("32")) * Decimal("5") / Decimal("9")))
    elif unit_key in {"um", "μm", "µm"}:
        unit_code, normalized_unit, dimension = "um", "um", "length"
    elif unit_key == "mm":
        unit_code, normalized_unit, dimension = "mm", "mm", "length"
    elif unit_key in {"mpa", "kpa", "gpa"}:
        unit_code, normalized_unit, dimension = raw_unit.upper(), "MPa", "stress"
        factor = {"mpa": Decimal("1"), "kpa": Decimal("0.001"), "gpa": Decimal("1000")}[unit_key]
        normalized = _decimal(str(Decimal(value) * factor))
    else:
        status = NormalizationStatus.UNRESOLVED
        unit_code = normalized_unit = dimension = None
        normalized = None
    measurement = Measurement(
        raw_text=match.group(0),
        comparator=comparator,
        value_decimal=value,
        unit_raw=match.group("unit"),
        unit_code=unit_code,
        dimension=dimension,
        normalized_value=normalized,
        normalized_unit=normalized_unit,
        normalization_status=status,
    )
    return value, measurement, status


def _candidates(message: CustomerMessage) -> list[_Candidate]:
    found: list[_Candidate] = []
    for match in _MEASUREMENT_RE.finditer(message.text):
        label = match.group("label").lower()
        if label in {"固含", "solid content"}:
            path = "/performance_indicators"
        else:
            path = "/performance_indicators"
        _, measurement, status = _measurement(match)
        known = match.group(0)
        found.append(_Candidate(path, message, match.start(), match.end(), known, known, measurement, status))
    for path, pattern in _TEXT_PATTERNS.items():
        for match in pattern.finditer(message.text):
            raw = match.group(0)
            value = match.group(1) if path not in {"/compliance", "/test_standards"} else raw
            found.append(_Candidate(path, message, match.start(), match.end(), raw, value))
    for pattern in (_SCENARIO_RE,):
        for match in pattern.finditer(message.text):
            raw = match.group(0)
            found.append(_Candidate("/application_scenario", message, match.start(), match.end(), raw, match.group(1)))
    for match in _PRODUCT_NAME_RE.finditer(message.text):
        raw = match.group(0)
        found.append(_Candidate("/material_or_product", message, match.start(), match.end(), raw, match.group(1)))
    return found


def extract_customer_requirement(
    messages: Iterable[CustomerMessage],
    *,
    context: DExtractionContext,
) -> CustomerRequirement:
    """Extract a deterministic current requirement projection from customer text."""

    current = _active_messages(messages)
    candidates = [candidate for message in current for candidate in _candidates(message)]
    case_id = current[0].case_id
    attributions: list[CustomerAttribution] = []
    annotations: list[FieldAnnotation] = []
    for path in FIELD_PATHS:
        items = [item for item in candidates if item.field_path == path]
        if not items:
            annotations.append(
                FieldAnnotation(
                    annotation_id=_id(context, case_id, path, "missing"),
                    field_path=path,
                    status=FieldStatus.MISSING,
                    value={"value_state": ValueState.UNSET},
                    normalization_status=NormalizationStatus.NOT_REQUESTED,
                )
            )
            continue
        unique_values = {item.known_value for item in items}
        status = FieldStatus.EXPLICIT if len(unique_values) == 1 else FieldStatus.CONFLICTING
        locator_ids: list[UUID] = []
        attribution_ids: list[UUID] = []
        for item in items:
            map_ = _normalization_map(item.message, context.normalizer_version)
            original_start, original_end = map_.original_span_for_normalized(item.start, item.end)
            locator_id = _id(context, item.message.message_id, path, original_start, original_end)
            attribution_id = _id(context, "attr", locator_id)
            locator = RuntimeCustomerLocator(
                message_id=item.message.message_id,
                message_sha256=item.message.text_sha256,
                original_unicode_code_point_length=len(item.message.text),
                char_start=original_start,
                char_end=original_end,
                quote=item.message.text[original_start:original_end],
                quote_sha256=sha256_text(item.message.text[original_start:original_end]),
                runtime_customer_locator_id=locator_id,
                extraction_run_id=context.execution_namespace_key.run_id,
                normalizer_version=context.normalizer_version,
            )
            attributions.append(
                CustomerAttribution(
                    customer_attribution_id=attribution_id,
                    case_id=case_id,
                    message_id=item.message.message_id,
                    message_sha256=item.message.text_sha256,
                    runtime_customer_locator=locator,
                    field_path=path,
                    attribution_status=AttributionStatus.EXACT_MATCH,
                )
            )
            locator_ids.append(locator_id)
            attribution_ids.append(attribution_id)
        if status is FieldStatus.CONFLICTING:
            value = {"value_state": ValueState.UNRESOLVED, "candidate_values": tuple(sorted(unique_values)), "unresolved_reason": "multiple_customer_values"}
            measurement = None
            normalization_status = NormalizationStatus.UNRESOLVED
        else:
            item = items[0]
            value = {"value_state": ValueState.KNOWN, "known_value": item.known_value}
            measurement = item.measurement
            normalization_status = item.normalization_status
        annotations.append(
            FieldAnnotation(
                annotation_id=_id(context, case_id, path, "current"),
                field_path=path,
                status=status,
                value=value,
                measurement=measurement,
                source_locator_ids=tuple(locator_ids),
                customer_attribution_ids=tuple(attribution_ids),
                normalization_status=normalization_status,
            )
        )
    input_hash = sha256_text(
        "|".join([str(case_id), str(context.revision), *[f"{m.message_id}:{m.text_sha256}" for m in current]])
    )
    blocking = tuple(
        str(item.field_path)
        for item in annotations
        if item.status in {FieldStatus.MISSING, FieldStatus.AMBIGUOUS, FieldStatus.CONFLICTING}
    )
    extracted_at = context.extracted_at or datetime.now(timezone.utc)
    return CustomerRequirement(
        requirement_id=_id(context, case_id, "requirement", context.revision),
        revision=context.revision,
        case_id=case_id,
        source_message_ids=tuple(item.message_id for item in current),
        field_annotations=tuple(annotations),
        customer_attributions=tuple(attributions),
        extraction_status=RequirementStatus.NEEDS_CONFIRMATION if blocking else RequirementStatus.COMPLETE,
        blocking_field_paths=blocking,
        input_snapshot_hash=input_hash,
        provenance={"producer": "module-d", "producer_version": "d-rule-baseline-0.1"},
        extracted_at=extracted_at,
    )
