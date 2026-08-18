"""Module E: deterministic, evidence-bound reply draft composition."""

from .composer import compose_reply_draft, fixture_evidence_only, review_reply_draft
from .e_internal import ComposeContext, ComposeResult, ComposerDecisionTrace, ServiceError

__all__ = [
    "ComposeContext",
    "ComposeResult",
    "ComposerDecisionTrace",
    "ServiceError",
    "compose_reply_draft",
    "fixture_evidence_only",
    "review_reply_draft",
]
