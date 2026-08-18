"""Private E implementation objects.

Only A-owned contract models cross the module boundary.  These dataclasses
carry deterministic decisions and diagnostics for F/H without becoming a
second public contract surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from assistant.contracts.api import GenerateReplyDraftRequest
from assistant.contracts.drafts import ReplyDraft
from assistant.contracts.errors import APIError, RefusalReason
from assistant.contracts.evidence import CustomerAttribution, TechnicalCitation
from assistant.contracts.facts import RuntimeKnowledgeFact
from assistant.contracts.limitations import LimitationDecision
from assistant.contracts.requirements import CustomerRequirement
from assistant.contracts.retrieval import RetrievalRun
from assistant.contracts.runtime import ExecutionNamespaceKey


@dataclass(frozen=True)
class ComposeContext:
    """F-provided immutable inputs for one E composition attempt."""

    execution_namespace_key: ExecutionNamespaceKey
    request: GenerateReplyDraftRequest
    requirement: CustomerRequirement
    retrieval_runs: tuple[RetrievalRun, ...] = ()
    technical_citations: tuple[TechnicalCitation, ...] = ()
    limitation_decisions: tuple[LimitationDecision, ...] = ()
    created_at: datetime | None = None
    template_version: str = "e-template-1.0.0"
    composer_version: str = "e-composer-1.0.0"


@dataclass(frozen=True)
class ClaimAtom:
    """A private, source-bound atom before A Claim materialization."""

    atom_id: str
    kind: Literal[
        "TECHNICAL_FACT",
        "CUSTOMER_STATEMENT",
        "ASSUMPTION",
        "UNKNOWN",
        "LIMITATION",
        "CLARIFICATION",
        "NEXT_ACTION",
    ]
    statement: str
    source_id: str | None
    source_kind: Literal["TECHNICAL_CITATION", "CUSTOMER_ATTRIBUTION", "NONE"]
    source_fact: RuntimeKnowledgeFact | None = None


@dataclass(frozen=True)
class SupportGateResult:
    atom_id: str
    accepted: bool
    qualified: bool
    reason: str
    source_id: str | None = None


@dataclass(frozen=True)
class SafetyFinding:
    category: str
    action: Literal["IGNORE_AS_DATA", "REMOVE", "REFUSE", "REQUIRE_REVIEW"]
    atom_id: str | None
    detail: str


@dataclass(frozen=True)
class ComposerDecisionTrace:
    """Redacted decision trace; it never stores source text or PII."""

    trace_id: str
    input_snapshot_hash: str
    namespace_hash: str
    stage_decisions: tuple[str, ...]
    support_gate_results: tuple[SupportGateResult, ...]
    safety_findings: tuple[SafetyFinding, ...]
    limitation_decision_ids: tuple[str, ...]
    accepted_atom_count: int
    rejected_atom_count: int
    body_sha256: str | None


@dataclass(frozen=True)
class ComposeResult:
    reply_draft: ReplyDraft | None
    refusal: RefusalReason | None
    trace: ComposerDecisionTrace


@dataclass(frozen=True)
class FixtureEvidenceReference:
    fixture_evidence_ref: str
    usage: Literal["DEVELOPMENT_FIXTURE_ONLY"] = "DEVELOPMENT_FIXTURE_ONLY"


class ServiceError(RuntimeError):
    """Raised with an A-owned APIError; no E-specific public error exists."""

    def __init__(self, api_error: APIError) -> None:
        super().__init__(str(api_error))
        self.api_error = api_error


PROVIDER_ADAPTER_STATUS: Literal["DISABLED/NOT_IMPLEMENTED"] = (
    "DISABLED/NOT_IMPLEMENTED"
)
