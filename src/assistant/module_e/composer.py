"""Deterministic evidence-bound reply draft composition for module E."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import re
from typing import Iterable
from uuid import NAMESPACE_URL, UUID, uuid5

from assistant.contracts.api import GenerateReplyDraftRequest, ReviewReplyDraftRequest
from assistant.contracts.base import Provenance, sha256_text
from assistant.contracts.claims import Claim, render_claim_statements
from assistant.contracts.drafts import DraftNextAction, ReplyDraft, ReviewDecision
from assistant.contracts.enums import (
    ClaimType,
    DraftPurpose,
    ErrorCode,
    FactKind,
    FieldStatus,
    LimitationAction,
    LimitationStatus,
    NormalizationStatus,
    RequirementStatus,
    RefusalCode,
    ReviewDecisionType,
    ReviewStatus,
    SupportLevel,
    SupportStatus,
    ValueState,
    VerificationStatus,
    CitationRelation,
)
from assistant.contracts.errors import APIError, RefusalReason
from assistant.contracts.evidence import CustomerAttribution, TechnicalCitation
from assistant.contracts.limitations import LimitationDecision
from assistant.contracts.requirements import CustomerRequirement, FieldAnnotation
from assistant.contracts.retrieval import RetrievalRun
from assistant.contracts.runtime import ExecutionNamespaceKey

from .e_internal import (
    ClaimAtom,
    ComposeContext,
    ComposeResult,
    ComposerDecisionTrace,
    FixtureEvidenceReference,
    SafetyFinding,
    ServiceError,
    SupportGateResult,
)


_INJECTION_RE = re.compile(
    r"(?:ignore\s+(?:all\s+)?(?:previous|system|safety)\s+instructions|"
    r"忽略(?:所有|之前的)?(?:规则|指令)|执行命令|读取文件|联网|调用工具|"
    r"execute\s+(?:the\s+)?command|reveal\s+(?:the\s+)?prompt)",
    re.IGNORECASE,
)
_COMMITMENT_RE = re.compile(
    r"(?:我司|我们|本公司|we\s+can|our\s+company).{0,40}"
    r"(?:保证|承诺|报价|交期|发货|下单|发送|写入\s*crm|guarantee|quote|"
    r"delivery|ship|order|send|crm)",
    re.IGNORECASE,
)
_FIXTURE_REF_RE = re.compile(r"^fixture-evidence:[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")

_QUALIFICATION_REQUIRED_FACT_KINDS = {
    FactKind.TYPICAL_VALUE,
    FactKind.SINGLE_TEST_RESULT,
    FactKind.RECOMMENDED_RANGE,
}


def _now(value: datetime | None) -> datetime:
    result = value or datetime.now(timezone.utc)
    if result.tzinfo is None or result.utcoffset() is None or result.utcoffset().total_seconds() != 0:
        raise _api_error(ErrorCode.SEMANTIC_VALIDATION_FAILED, "E requires a timezone-aware UTC timestamp")
    return result.astimezone(timezone.utc)


def _id(*parts: object) -> UUID:
    return uuid5(NAMESPACE_URL, "module-e:" + ":".join(str(part) for part in parts))


def _api_error(code: ErrorCode, message: str, *, retryable: bool = False) -> ServiceError:
    return ServiceError(APIError(code=code, message=message, retryable=retryable))


def _refusal(
    code: RefusalCode,
    *,
    missing: Iterable[str] = (),
    conflicts: Iterable[UUID] = (),
    message: str,
    next_action: str,
) -> RefusalReason:
    return RefusalReason(
        code=code,
        user_message=message,
        missing_information=tuple(str(item) for item in missing),
        conflicting_technical_citation_ids=tuple(conflicts),
        suggested_next_action=next_action,
    )


def _trace(
    context: ComposeContext,
    *,
    stages: list[str],
    gates: list[SupportGateResult],
    findings: list[SafetyFinding],
    limitation_decision_ids: Iterable[UUID] = (),
    accepted: int,
    rejected: int,
    body_sha256: str | None,
) -> ComposerDecisionTrace:
    trace_id = str(
        _id(
            context.execution_namespace_key.namespace_hash,
            context.request.case_id,
            context.request.requirement_id,
            context.request.requirement_revision,
            context.request.input_snapshot_hash,
            _input_fingerprint(context),
            tuple(str(item) for item in context.request.retrieval_run_ids),
            context.template_version,
            context.composer_version,
            "trace",
        )
    )
    return ComposerDecisionTrace(
        trace_id=trace_id,
        input_snapshot_hash=str(context.request.input_snapshot_hash),
        namespace_hash=str(context.execution_namespace_key.namespace_hash),
        stage_decisions=tuple(stages),
        support_gate_results=tuple(gates),
        safety_findings=tuple(findings),
        limitation_decision_ids=tuple(sorted({str(item) for item in limitation_decision_ids})),
        accepted_atom_count=accepted,
        rejected_atom_count=rejected,
        body_sha256=body_sha256,
    )


def _draft_id(context: ComposeContext, purpose: str) -> UUID:
    """Derive an id from the complete immutable composition identity."""

    return _id(
        context.execution_namespace_key.namespace_hash,
        context.request.case_id,
        context.request.requirement_id,
        context.request.requirement_revision,
        context.request.input_snapshot_hash,
        _input_fingerprint(context),
        tuple(str(item) for item in context.request.retrieval_run_ids),
        context.template_version,
        context.composer_version,
        purpose,
    )


def _claim_id(context: ComposeContext, source_kind: str, source_id: object) -> UUID:
    return _id(
        context.execution_namespace_key.namespace_hash,
        context.request.requirement_id,
        context.request.requirement_revision,
        source_kind,
        source_id,
    )


def _input_fingerprint(context: ComposeContext) -> str:
    """Hash the immutable F/C/D input graph without retaining source text."""

    payload = {
        "request": context.request.model_dump(mode="json"),
        "requirement": context.requirement.model_dump(mode="json"),
        "retrieval_runs": [item.model_dump(mode="json") for item in context.retrieval_runs],
        "technical_citations": [
            item.model_dump(mode="json") for item in context.technical_citations
        ],
        "limitation_decisions": [
            item.model_dump(mode="json") for item in context.limitation_decisions
        ],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256_text(canonical)


def _validate_context(context: ComposeContext) -> dict[UUID, RetrievalRun]:
    request = context.request
    requirement = context.requirement
    if context.created_at is not None:
        _now(context.created_at)
    if request.case_id != requirement.case_id:
        raise _api_error(ErrorCode.REVISION_CONFLICT, "request and requirement case IDs differ")
    if request.requirement_id != requirement.requirement_id:
        raise _api_error(ErrorCode.REVISION_CONFLICT, "request requirement ID is stale")
    if request.requirement_revision != requirement.revision:
        raise _api_error(ErrorCode.REVISION_CONFLICT, "request requirement revision is stale")
    if request.input_snapshot_hash != requirement.input_snapshot_hash:
        raise _api_error(ErrorCode.REVISION_CONFLICT, "request input snapshot hash is stale")
    requested_ids = tuple(request.retrieval_run_ids)
    if len(requested_ids) != len(set(requested_ids)):
        raise _api_error(ErrorCode.SEMANTIC_VALIDATION_FAILED, "retrieval run IDs must be unique")
    runs = {run.retrieval_run_id: run for run in context.retrieval_runs}
    if len(runs) != len(context.retrieval_runs):
        raise _api_error(ErrorCode.SEMANTIC_VALIDATION_FAILED, "retrieval runs must have unique IDs")
    if set(runs) != set(requested_ids):
        raise _api_error(ErrorCode.DEPENDENCY_UNAVAILABLE, "F did not provide every requested retrieval run")
    for run in runs.values():
        if run.case_id != request.case_id:
            raise _api_error(ErrorCode.REVISION_CONFLICT, "retrieval run belongs to another case")
        if run.environment_namespace != context.execution_namespace_key.environment:
            raise _api_error(ErrorCode.SEMANTIC_VALIDATION_FAILED, "retrieval run namespace differs from F context")
    citation_ids = [item.technical_citation_id for item in context.technical_citations]
    if len(citation_ids) != len(set(citation_ids)):
        raise _api_error(ErrorCode.SEMANTIC_VALIDATION_FAILED, "technical citation IDs must be unique")
    limitation_ids = [item.limitation_decision_id for item in context.limitation_decisions]
    if len(limitation_ids) != len(set(limitation_ids)):
        raise _api_error(ErrorCode.SEMANTIC_VALIDATION_FAILED, "limitation decision IDs must be unique")
    return runs


def _decision_matches_citation(
    decision: LimitationDecision,
    citation: TechnicalCitation,
) -> bool:
    return bool(
        citation.claim_id in decision.affected_claim_ids
        or citation.technical_citation_id in decision.technical_citation_refs
        or citation.runtime_fact_id in decision.limitation_fact_ids
        or citation.runtime_document_locator.runtime_locator_id in decision.runtime_locator_refs
    )


def _has_qualifying_decision(
    context: ComposeContext,
    citation: TechnicalCitation,
) -> bool:
    return any(
        decision.status is LimitationStatus.APPLIES
        and decision.resolution.action is LimitationAction.QUALIFY
        and _decision_matches_citation(decision, citation)
        for decision in context.limitation_decisions
    )


def _eligible_citations(
    context: ComposeContext,
    runs: dict[UUID, RetrievalRun],
) -> tuple[list[TechnicalCitation], list[SupportGateResult], int, list[SafetyFinding]]:
    eligible: list[TechnicalCitation] = []
    gates: list[SupportGateResult] = []
    findings: list[SafetyFinding] = []
    seen_fact_ids: set[UUID] = set()
    eligible_run_ids: set[UUID] = set()
    rejected = 0
    if context.technical_citations and len(runs) != 1:
        raise _api_error(
            ErrorCode.SEMANTIC_VALIDATION_FAILED,
            "a technical reply draft must use exactly one retrieval run",
        )
    for citation in context.technical_citations:
        if citation.case_id != context.request.case_id:
            raise _api_error(ErrorCode.REVISION_CONFLICT, "technical citation belongs to another case")
        run = runs.get(citation.retrieval_run_id)
        if run is None:
            raise _api_error(ErrorCode.SEMANTIC_VALIDATION_FAILED, "citation uses an undeclared retrieval run")
        result = next((item for item in run.results if item.result_id == citation.result_id), None)
        if result is None or result != citation.retrieval_result:
            raise _api_error(ErrorCode.SEMANTIC_VALIDATION_FAILED, "citation is not bound to the supplied retrieval result")
        if citation.runtime_fact is None:
            raise _api_error(ErrorCode.SEMANTIC_VALIDATION_FAILED, "citation is missing its runtime fact snapshot")
        if citation.runtime_fact.index_snapshot_id != run.index_snapshot_id:
            raise _api_error(ErrorCode.REVISION_CONFLICT, "citation runtime fact is from a different index snapshot")
        atom_id = str(citation.technical_citation_id)
        quote = citation.runtime_document_locator.quote
        if _COMMITMENT_RE.search(quote):
            rejected += 1
            gates.append(SupportGateResult(atom_id, False, False, "DANGEROUS_COMMITMENT_BLOCKED", atom_id))
            findings.append(
                SafetyFinding(
                    "DANGEROUS_COMMITMENT_TEXT",
                    "REMOVE",
                    atom_id,
                    "source text was excluded; no commitment may enter a machine draft",
                )
            )
            continue
        exact = (
            citation.evidence_eligible
            and result.evidence_eligible
            and citation.relation == CitationRelation.SUPPORTS
            and citation.support_level == SupportLevel.DIRECT
            and citation.verification_status == VerificationStatus.EXACT_MATCH
        )
        if not exact:
            rejected += 1
            reason = "TECHNICAL_EVIDENCE_GATE_FAILED"
            if citation.verification_status is VerificationStatus.STALE:
                reason = "STALE_SOURCE_VERSION"
            elif citation.relation is CitationRelation.CONTRADICTS:
                reason = "CONFLICTING_TECHNICAL_EVIDENCE"
            gates.append(SupportGateResult(atom_id, False, False, reason, atom_id))
            continue
        if citation.fact_kind == FactKind.LIMITATION:
            rejected += 1
            gates.append(SupportGateResult(atom_id, False, False, "LIMITATION_REQUIRES_DECISION", atom_id))
            continue
        if citation.runtime_fact_id in seen_fact_ids:
            rejected += 1
            gates.append(SupportGateResult(atom_id, False, False, "DUPLICATE_RUNTIME_FACT", atom_id))
            continue
        requires_qualification = (
            citation.fact_kind in _QUALIFICATION_REQUIRED_FACT_KINDS
            or bool(citation.runtime_fact.test_conditions)
            or bool(citation.runtime_fact.applicability_scope.conditions)
        )
        if requires_qualification and not _has_qualifying_decision(context, citation):
            rejected += 1
            gates.append(
                SupportGateResult(
                    atom_id,
                    False,
                    False,
                    "FACT_REQUIRES_EXPLICIT_QUALIFICATION",
                    atom_id,
                )
            )
            continue
        seen_fact_ids.add(citation.runtime_fact_id)
        eligible_run_ids.add(citation.retrieval_run_id)
        eligible.append(citation)
        gates.append(SupportGateResult(atom_id, True, False, "TECHNICAL_EVIDENCE_ACCEPTED", atom_id))
        if _INJECTION_RE.search(quote):
            findings.append(SafetyFinding("PROMPT_INJECTION_AS_DATA", "IGNORE_AS_DATA", atom_id, "source text treated as data"))
    if len(eligible_run_ids) > 1:
        raise _api_error(ErrorCode.SEMANTIC_VALIDATION_FAILED, "one E draft cannot combine multiple retrieval runs")
    eligible.sort(key=lambda item: (runs[item.retrieval_run_id].results.index(item.retrieval_result), str(item.technical_citation_id)))
    return eligible, gates, rejected, findings


def _blocking_fields(requirement: CustomerRequirement) -> tuple[str, ...]:
    paths = {str(path) for path in requirement.blocking_field_paths}
    for annotation in requirement.field_annotations:
        if annotation.status in {
            FieldStatus.MISSING,
            FieldStatus.AMBIGUOUS,
            FieldStatus.CONFLICTING,
        }:
            paths.add(str(annotation.field_path))
        if annotation.value.value_state in {
            ValueState.UNSET,
            ValueState.UNKNOWN,
            ValueState.UNRESOLVED,
        }:
            paths.add(str(annotation.field_path))
        if annotation.normalization_status in {
            NormalizationStatus.UNRESOLVED,
            NormalizationStatus.FAILED,
        }:
            paths.add(str(annotation.field_path))
    return tuple(sorted(paths))


def _questions(
    requirement: CustomerRequirement,
    decisions: Iterable[LimitationDecision],
) -> tuple[str, ...]:
    blockers = _blocking_fields(requirement)
    questions = [f"请补充或确认客户需求字段：{path}。" for path in blockers]
    if requirement.extraction_status is not RequirementStatus.COMPLETE and not blockers:
        questions.append("请确认客户需求提取结果后再继续技术回复。")
    for decision in decisions:
        if decision.status is LimitationStatus.UNRESOLVED or decision.resolution.action in {
            LimitationAction.ASK_CLARIFICATION,
            LimitationAction.HUMAN_REVIEW,
        }:
            questions.append(f"请确认限制条件（{decision.reason_code}）后再决定是否适用。")
    return tuple(dict.fromkeys(questions))


def _requirement_assumptions(requirement: CustomerRequirement) -> tuple[str, ...]:
    """Keep derived/uncertain states explicit without turning them into claims."""

    assumptions: list[str] = []
    for annotation in sorted(requirement.field_annotations, key=lambda item: str(item.field_path)):
        if annotation.status is FieldStatus.DERIVED:
            assumptions.append(f"字段 {annotation.field_path} 为派生值，需人工确认后使用。")
        elif annotation.status is FieldStatus.SUPERSEDED:
            assumptions.append(f"字段 {annotation.field_path} 已被替代，未作为当前事实使用。")
    return tuple(assumptions)


def _customer_claims(
    context: ComposeContext,
) -> tuple[list[Claim], list[CustomerAttribution], dict[str, UUID]]:
    requirement = context.requirement
    by_id = {item.customer_attribution_id: item for item in requirement.customer_attributions}
    claims: list[Claim] = []
    attributions: list[CustomerAttribution] = []
    old_to_new: dict[str, UUID] = {}
    selected: set[UUID] = set()
    annotations = sorted(requirement.field_annotations, key=lambda item: str(item.field_path))
    for annotation in annotations:
        if annotation.status != FieldStatus.EXPLICIT:
            continue
        if annotation.value.value_state not in {ValueState.KNOWN, ValueState.NOT_APPLICABLE}:
            continue
        for attribution_id in annotation.customer_attribution_ids:
            attribution = by_id[attribution_id]
            if attribution.customer_attribution_id in selected:
                continue
            selected.add(attribution.customer_attribution_id)
            locator = attribution.runtime_customer_locator
            claim_id = _claim_id(context, "customer", attribution.customer_attribution_id)
            claims.append(
                Claim(
                    claim_id=claim_id,
                    statement=locator.quote,
                    statement_sha256=locator.quote_sha256,
                    claim_type=ClaimType.CUSTOMER_STATEMENT,
                    support_status=SupportStatus.SUPPORTED,
                    customer_attribution_ids=(attribution.customer_attribution_id,),
                )
            )
            attributions.append(attribution.model_copy(update={"claim_id": claim_id}))
            old_to_new[str(attribution.customer_attribution_id)] = claim_id
            if attribution.claim_id is not None:
                old_to_new[str(attribution.claim_id)] = claim_id
    return claims, attributions, old_to_new


def _apply_limitations(
    context: ComposeContext,
    citations: list[TechnicalCitation],
    old_to_new: dict[str, UUID],
) -> tuple[
    tuple[LimitationDecision, ...],
    set[UUID],
    set[UUID],
    tuple[str, ...],
    bool,
    bool,
]:
    """Map upstream limitation references onto this draft's claim IDs.

    A limitation may refer to a fact, locator, citation, or the upstream claim
    ID.  Every reference must resolve against the supplied runtime objects;
    silently dropping an unresolved decision would turn a constrained fact into
    an apparently unconditional one.
    """

    for citation in citations:
        old_to_new.setdefault(str(citation.claim_id), _claim_id(context, "technical", citation.runtime_fact_id))
    all_citations = tuple(context.technical_citations)
    citation_ids = {str(item.technical_citation_id) for item in all_citations}
    fact_ids = {str(item.runtime_fact_id) for item in all_citations}
    locator_ids = {
        str(item.runtime_document_locator.runtime_locator_id) for item in all_citations
    }
    copied: list[LimitationDecision] = []
    qualified: set[UUID] = set()
    removed: set[UUID] = set()
    disclosures: list[str] = []
    requires_clarification = False
    requires_refusal = False
    for decision in context.limitation_decisions:
        affected_old = {str(item) for item in decision.affected_claim_ids}
        mapped_ids = {old_to_new[item] for item in affected_old if item in old_to_new}
        matched_any_evidence = False
        for citation in all_citations:
            if _decision_matches_citation(decision, citation):
                matched_any_evidence = True
                mapped = old_to_new.get(str(citation.claim_id))
                if mapped is not None:
                    mapped_ids.add(mapped)
        unknown_claims = affected_old - set(old_to_new)
        unknown_citation_refs = {
            str(item) for item in decision.technical_citation_refs
        } - citation_ids
        unknown_fact_refs = {str(item) for item in decision.limitation_fact_ids} - fact_ids
        unknown_locator_refs = {str(item) for item in decision.runtime_locator_refs} - locator_ids
        if unknown_claims or unknown_citation_refs or unknown_fact_refs or unknown_locator_refs:
            raise _api_error(
                ErrorCode.REVISION_CONFLICT,
                "limitation decision references stale or undeclared evidence",
            )
        if (
            decision.status in {LimitationStatus.APPLIES, LimitationStatus.UNRESOLVED}
            and not mapped_ids
        ):
            raise _api_error(
                ErrorCode.SEMANTIC_VALIDATION_FAILED,
                "limitation decision does not bind an E draft claim",
            )
        mapped_tuple = tuple(sorted(mapped_ids, key=str))
        action = decision.resolution.action
        if decision.status is LimitationStatus.APPLIES:
            if action is LimitationAction.QUALIFY:
                qualified.update(mapped_ids)
            elif action is LimitationAction.REMOVE_CLAIM:
                removed.update(mapped_ids)
            elif action is LimitationAction.REFUSE:
                requires_refusal = True
            elif action in {LimitationAction.ASK_CLARIFICATION, LimitationAction.HUMAN_REVIEW}:
                requires_clarification = True
        elif decision.status is LimitationStatus.UNRESOLVED:
            if action is LimitationAction.REFUSE:
                requires_refusal = True
            elif action is LimitationAction.REMOVE_CLAIM:
                removed.update(mapped_ids)
                requires_clarification = True
            else:
                requires_clarification = True
        if decision.disclosure.required and decision.disclosure.text:
            disclosures.append(decision.disclosure.text)
        copied.append(decision.model_copy(update={"affected_claim_ids": mapped_tuple}))
    return (
        tuple(copied),
        qualified,
        removed,
        tuple(dict.fromkeys(disclosures)),
        requires_clarification,
        requires_refusal,
    )


def _provenance(context: ComposeContext) -> Provenance:
    return Provenance(
        producer="module-e",
        producer_version=context.composer_version,
        config_hash=sha256_text(context.template_version),
    )


def _build_clarification(
    context: ComposeContext,
    questions: tuple[str, ...],
    trace_data: dict[str, object],
    *,
    assumptions: tuple[str, ...] = (),
    limitations: tuple[str, ...] = (),
    limitation_decision_ids: Iterable[UUID] = (),
    additional_blockers: Iterable[str] = (),
) -> ComposeResult:
    if not questions:
        refusal = _refusal(
            RefusalCode.INSUFFICIENT_EVIDENCE,
            message="当前没有足够的客户或技术证据生成安全草稿。",
            next_action="请补充客户需求或提供可验证的技术资料。",
        )
        return ComposeResult(
            None,
            refusal,
            _trace(
                context,
                body_sha256=None,
                limitation_decision_ids=limitation_decision_ids,
                **trace_data,
            ),
        )
    now = _now(context.created_at or context.requirement.extracted_at)
    draft_id = _draft_id(context, "clarification")
    body = "\n".join(questions)
    blockers = tuple(
        dict.fromkeys((*_blocking_fields(context.requirement), *additional_blockers))
    )
    draft = ReplyDraft(
        draft_id=draft_id,
        revision=context.request.requirement_revision,
        case_id=context.request.case_id,
        purpose=DraftPurpose.CLARIFICATION,
        requirement_revision=context.request.requirement_revision,
        retrieval_run_ids=(),
        input_snapshot_hash=context.request.input_snapshot_hash,
        subject="客户需求澄清草稿",
        body_render_mode="STRUCTURED_QUESTIONS_V1",
        body=body,
        body_sha256=sha256_text(body),
        assumptions=assumptions,
        questions_to_confirm=questions,
        next_actions=(DraftNextAction(action_type="INTERNAL_REVIEW", description="由人工审核客户原文和澄清问题后再决定后续沟通。", owner_role="TECHNICAL_SUPPORT"),),
        limitations=limitations,
        limitation_decisions=(),
        blocking_field_paths=blockers,
        review_status=ReviewStatus.REQUIRES_HUMAN_REVIEW,
        provenance=_provenance(context),
        created_at=now,
    )
    return ComposeResult(
        draft,
        None,
        _trace(
            context,
            body_sha256=draft.body_sha256,
            limitation_decision_ids=limitation_decision_ids,
            **trace_data,
        ),
    )


def compose_reply_draft(context: ComposeContext) -> ComposeResult:
    """Compose an A `ReplyDraft` from real C/D runtime objects only."""

    runs = _validate_context(context)
    stages = ["RECEIVED", "INPUT_LOCKED", "ATOMIZED"]
    gates: list[SupportGateResult] = []
    findings: list[SafetyFinding] = []
    citations, citation_gates, rejected, citation_findings = _eligible_citations(context, runs)
    gates.extend(citation_gates)
    findings.extend(citation_findings)
    stages.append("SUPPORT_GATED")

    questions = _questions(context.requirement, context.limitation_decisions)
    blockers = _blocking_fields(context.requirement)
    assumptions = _requirement_assumptions(context.requirement)
    limitation_decisions: tuple[LimitationDecision, ...] = ()
    qualified: set[UUID] = set()
    removed: set[UUID] = set()
    disclosure_texts: tuple[str, ...] = ()
    requires_clarification = False
    requires_refusal = False
    customer_claims: list[Claim] = []
    customer_attributions: list[CustomerAttribution] = []
    customer_old_to_new: dict[str, UUID] = {}

    if citations:
        customer_claims, customer_attributions, customer_old_to_new = _customer_claims(context)
        (
            limitation_decisions,
            qualified,
            removed,
            disclosure_texts,
            requires_clarification,
            requires_refusal,
        ) = _apply_limitations(context, citations, customer_old_to_new)
    stages.append("LIMITATIONS_APPLIED")

    limitation_blockers = tuple(
        f"/limitation_decisions/{decision.limitation_decision_id}"
        for decision in context.limitation_decisions
        if decision.status is LimitationStatus.UNRESOLVED
        or decision.resolution.action
        in {LimitationAction.ASK_CLARIFICATION, LimitationAction.HUMAN_REVIEW}
    )
    requirement_status_blocker = (
        ("/requirement",)
        if context.requirement.extraction_status is not RequirementStatus.COMPLETE
        and not blockers
        else ()
    )
    limitation_question_needed = bool(limitation_blockers)
    if blockers or requirement_status_blocker or requires_clarification or limitation_question_needed:
        stages.append("NEEDS_CLARIFICATION")
        clarification_limits = disclosure_texts
        return _build_clarification(
            context,
            questions,
            dict(
                stages=stages,
                gates=gates,
                findings=findings,
                accepted=0,
                rejected=rejected,
            ),
            assumptions=assumptions,
            limitations=clarification_limits,
            limitation_decision_ids=(
                item.limitation_decision_id for item in context.limitation_decisions
            ),
            additional_blockers=(*limitation_blockers, *requirement_status_blocker),
        )
    if requires_refusal:
        stages.append("REFUSED")
        refusal = _refusal(
            RefusalCode.INSUFFICIENT_EVIDENCE,
            message="限制条件不允许生成确定性技术回复。",
            next_action="请由技术支持人工处理限制条件后再继续。",
        )
        return ComposeResult(
            None,
            refusal,
            _trace(
                context,
                stages=stages,
                gates=gates,
                findings=findings,
                limitation_decision_ids=(
                    item.limitation_decision_id for item in context.limitation_decisions
                ),
                accepted=0,
                rejected=rejected,
                body_sha256=None,
            ),
        )
    if not citations:
        reasons = {item.reason for item in gates if not item.accepted}
        if "DANGEROUS_COMMITMENT_BLOCKED" in reasons:
            refusal_code = RefusalCode.PROHIBITED_ACTION
            message = "检测到不可自动承诺的内容，未生成技术回复。"
            next_action = "请由人工审核后决定是否需要单独回复。"
        elif "CONFLICTING_TECHNICAL_EVIDENCE" in reasons:
            refusal_code = RefusalCode.CONFLICTING_EVIDENCE
            message = "技术证据存在冲突，不能生成确定性技术回复。"
            next_action = "请由技术支持核对有效版本和冲突来源。"
        elif "STALE_SOURCE_VERSION" in reasons:
            refusal_code = RefusalCode.NO_ACTIVE_SOURCE_VERSION
            message = "技术来源版本已失效或过期，不能生成确定性技术回复。"
            next_action = "请提供当前有效版本并重新检索。"
        else:
            refusal_code = RefusalCode.INSUFFICIENT_EVIDENCE
            message = "当前没有可直接、精确且有效的技术证据，不能生成确定性技术回复。"
            next_action = "请补充可验证资料或由技术支持人工复核。"
        stages.append("REFUSED")
        refusal = _refusal(
            refusal_code,
            conflicts=(
                item.technical_citation_id
                for item in context.technical_citations
                if item.technical_citation_id
                and any(
                    gate.atom_id == str(item.technical_citation_id)
                    and gate.reason == "CONFLICTING_TECHNICAL_EVIDENCE"
                    for gate in gates
                )
            ),
            message=message,
            next_action=next_action,
        )
        return ComposeResult(
            None,
            refusal,
            _trace(
                context,
                stages=stages,
                gates=gates,
                findings=findings,
                limitation_decision_ids=(
                    item.limitation_decision_id for item in context.limitation_decisions
                ),
                accepted=0,
                rejected=rejected,
                body_sha256=None,
            ),
        )

    technical_claims: list[Claim] = []
    technical_attributions: list[TechnicalCitation] = []
    for citation in citations:
        claim_id = _claim_id(context, "technical", citation.runtime_fact_id)
        if claim_id in removed:
            continue
        statement = citation.runtime_document_locator.quote
        claim = Claim(
            claim_id=claim_id,
            statement=statement,
            statement_sha256=citation.runtime_document_locator.quote_sha256,
            claim_type=ClaimType.FACT,
            support_status=SupportStatus.QUALIFIED if claim_id in qualified else SupportStatus.SUPPORTED,
            technical_citation_ids=(citation.technical_citation_id,),
        )
        technical_claims.append(claim)
        technical_attributions.append(citation.model_copy(update={"claim_id": claim_id}))
    customer_claims = [item for item in customer_claims if item.claim_id not in removed]
    customer_attributions = [item for item in customer_attributions if item.claim_id not in removed]
    claims = tuple(technical_claims + customer_claims)
    if limitation_decisions:
        surviving_ids = {claim.claim_id for claim in claims}
        retained_decisions: list[LimitationDecision] = []
        for decision in limitation_decisions:
            retained = tuple(
                claim_id
                for claim_id in decision.affected_claim_ids
                if claim_id in surviving_ids
            )
            if decision.status is LimitationStatus.DOES_NOT_APPLY and not retained:
                retained_decisions.append(decision.model_copy(update={"affected_claim_ids": ()}))
            elif retained:
                retained_decisions.append(
                    decision.model_copy(update={"affected_claim_ids": retained})
                )
        limitation_decisions = tuple(retained_decisions)
    if not technical_claims:
        stages.append("REFUSED")
        refusal = _refusal(
            RefusalCode.INSUFFICIENT_EVIDENCE,
            message="限制条件移除了全部可用技术事实，不能生成确定性技术回复。",
            next_action="请先由技术支持人工处理限制条件。",
        )
        return ComposeResult(
            None,
            refusal,
            _trace(
                context,
                stages=stages,
                gates=gates,
                findings=findings,
                limitation_decision_ids=(
                    item.limitation_decision_id for item in context.limitation_decisions
                ),
                accepted=0,
                rejected=rejected + len(citations),
                body_sha256=None,
            ),
        )

    stages.extend(["ASSEMBLED", "SAFETY_SCANNED", "MATERIALIZED", "REQUIRES_HUMAN_REVIEW"])
    body = render_claim_statements(claims)
    if disclosure_texts:
        findings.append(SafetyFinding("LIMITATION_DISCLOSURE", "REQUIRE_REVIEW", None, "structured limitation retained"))
    draft = ReplyDraft(
        draft_id=_draft_id(context, "technical"),
        revision=context.request.requirement_revision,
        case_id=context.request.case_id,
        purpose=DraftPurpose.TECHNICAL_RESPONSE,
        requirement_revision=context.request.requirement_revision,
        retrieval_run_ids=(citations[0].retrieval_run_id,),
        input_snapshot_hash=context.request.input_snapshot_hash,
        subject="技术资料回复草稿",
        body_render_mode="CLAIM_STATEMENTS_V1",
        body=body,
        body_sha256=sha256_text(body),
        claims=claims,
        technical_citations=tuple(technical_attributions),
        customer_attributions=tuple(customer_attributions),
        assumptions=assumptions,
        questions_to_confirm=questions,
        next_actions=(DraftNextAction(action_type="INTERNAL_REVIEW", description="由人工审核技术证据、客户原文和限制后再决定后续沟通。", owner_role="TECHNICAL_SUPPORT"),),
        limitations=disclosure_texts + (("部分检索结果未通过证据门，未进入草稿正文。",) if rejected else ()),
        limitation_decisions=limitation_decisions,
        blocking_field_paths=(),
        review_status=ReviewStatus.REQUIRES_HUMAN_REVIEW,
        provenance=_provenance(context),
        created_at=_now(context.created_at or context.requirement.extracted_at),
    )
    return ComposeResult(
        draft,
        None,
        _trace(
            context,
            stages=stages,
            gates=gates,
            findings=findings,
            limitation_decision_ids=(
                item.limitation_decision_id for item in context.limitation_decisions
            ),
            accepted=len(claims),
            rejected=rejected,
            body_sha256=draft.body_sha256,
        ),
    )


def fixture_evidence_only(
    fixture_evidence_ref: str,
    *,
    execution_namespace_key: ExecutionNamespaceKey,
) -> FixtureEvidenceReference:
    """Return the only E-visible output allowed for B's development fixture."""

    if execution_namespace_key.environment.value != "DEVELOPMENT":
        raise _api_error(
            ErrorCode.SEMANTIC_VALIDATION_FAILED,
            "deterministic fixture evidence is restricted to DEVELOPMENT",
        )
    if not _FIXTURE_REF_RE.fullmatch(fixture_evidence_ref or ""):
        raise _api_error(ErrorCode.SEMANTIC_VALIDATION_FAILED, "invalid fixture_evidence_ref")
    return FixtureEvidenceReference(fixture_evidence_ref=fixture_evidence_ref)


def review_reply_draft(
    current_draft: ReplyDraft,
    request: ReviewReplyDraftRequest,
    *,
    now: datetime | None = None,
) -> ReviewDecision:
    """Apply a human review decision while rejecting stale or unresolved drafts."""

    if request.case_id != current_draft.case_id or request.draft_id != current_draft.draft_id:
        raise _api_error(ErrorCode.NOT_FOUND, "draft does not belong to the requested case")
    if request.expected_revision != current_draft.revision:
        raise _api_error(ErrorCode.REVISION_CONFLICT, "draft revision is stale")
    if request.draft_content_sha256 != current_draft.body_sha256:
        raise _api_error(ErrorCode.REVISION_CONFLICT, "draft content hash is stale")
    if current_draft.review_status != ReviewStatus.REQUIRES_HUMAN_REVIEW:
        raise _api_error(ErrorCode.REVISION_CONFLICT, "draft is no longer awaiting human review")
    if request.decision in {ReviewDecisionType.REJECT, ReviewDecisionType.REQUEST_CHANGES} and not request.comment:
        raise _api_error(
            ErrorCode.SEMANTIC_VALIDATION_FAILED,
            "reject or request-changes requires a human comment",
        )
    unresolved = list(current_draft.blocking_field_paths)
    unresolved.extend(
        decision.reason_code
        for decision in current_draft.limitation_decisions
        if decision.status is LimitationStatus.UNRESOLVED
        or decision.resolution.action
        in {
            LimitationAction.ASK_CLARIFICATION,
            LimitationAction.HUMAN_REVIEW,
            LimitationAction.REFUSE,
        }
    )
    unresolved = list(dict.fromkeys(unresolved))
    if request.decision == ReviewDecisionType.APPROVE and unresolved:
        raise _api_error(ErrorCode.SEMANTIC_VALIDATION_FAILED, "draft has unresolved blockers and cannot be approved")
    reviewed_at = _now(now)
    decision_id = _id(
        current_draft.draft_id,
        current_draft.revision,
        request.idempotency_key,
        request.reviewer_id,
        request.decision,
        sha256_text(request.comment or ""),
    )
    return ReviewDecision(
        review_decision_id=decision_id,
        case_id=current_draft.case_id,
        draft_id=current_draft.draft_id,
        draft_revision=current_draft.revision,
        draft_content_sha256=current_draft.body_sha256,
        decision=request.decision,
        reviewer_id=request.reviewer_id,
        reviewer_role=request.reviewer_role,
        reviewed_at=reviewed_at,
        comment=request.comment,
        unresolved_blockers=tuple(unresolved),
        audit_event_id=_id(decision_id, "audit"),
    )
