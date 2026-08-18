"""Public Module B build, validation, and H-only sealing surface."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .generation import DATASET_ID, DATASET_VERSION, PLANNED_COUNTS, generate_all
from .sealing import (
    AReleaseGateError,
    SealError,
    SealIntegrityError,
    check_a_release_gate,
    read_sealed_gold,
    seal_dataset,
    verify_seal,
)


def validate_all(
    project_root: str | Path,
    *,
    write_reports: bool = True,
) -> dict[str, Any]:
    """Load the validator lazily so package imports remain side-effect free."""

    from .validation import validate_all as implementation

    return implementation(project_root, write_reports=write_reports)


def validate_dataset(
    project_root: str | Path,
    *,
    write_reports: bool = True,
) -> dict[str, Any]:
    """Backward-compatible name for :func:`validate_all`."""

    return validate_all(project_root, write_reports=write_reports)


__all__ = (
    "AReleaseGateError",
    "DATASET_ID",
    "DATASET_VERSION",
    "PLANNED_COUNTS",
    "SealError",
    "SealIntegrityError",
    "check_a_release_gate",
    "generate_all",
    "read_sealed_gold",
    "seal_dataset",
    "validate_all",
    "validate_dataset",
    "verify_seal",
)
