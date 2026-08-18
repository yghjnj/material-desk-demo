"""Command-line entry points for Module B build, validation, and sealing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

from .generation import generate_all
from .sealing import (
    AReleaseGateError,
    SealError,
    check_a_release_gate,
    seal_dataset,
    verify_seal,
)


EXIT_OK = 0
EXIT_ERROR = 1
EXIT_BLOCKED = 3


def _emit(payload: Any, *, stream: Any = None) -> None:
    target = stream if stream is not None else sys.stdout
    target.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _result_exit_code(payload: Any) -> int:
    """Map a structured command result to a truthful process exit code."""

    if not isinstance(payload, dict):
        return EXIT_OK
    status = str(payload.get("status", "")).upper()
    if status in {"BLOCKED", "BLOCKED_BY_A"}:
        return EXIT_BLOCKED
    if status in {"FAIL", "FAILED", "INVALID"}:
        return EXIT_ERROR
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m assistant.module_b.cli")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("generate", "build development data and holdout Gold in memory"),
        ("validate", "validate generated Module B data"),
        ("seal", "create the A-gated immutable H-only seal"),
        ("status", "show A gate and seal integrity status"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument(
            "--project-root",
            type=Path,
            default=Path.cwd(),
            help="project root (default: current directory)",
        )
    return parser


def _run_generate(root: Path) -> dict[str, Any]:
    generated = generate_all(root)
    return {
        "status": "GENERATED",
        "project_root": str(root.resolve()),
        "actual_counts": generated["status"]["actual_counts"],
        "seal_status": "PENDING",
        "sealed_payload_written": False,
    }


def _run_validate(root: Path) -> dict[str, Any]:
    from .validation import validate_all

    return validate_all(root, write_reports=True)


def _run_status(root: Path) -> dict[str, Any]:
    gate = check_a_release_gate(root)
    seal = verify_seal(root)
    if seal["valid"]:
        status = "SEALED"
    elif gate["status"] != "PASS":
        status = "BLOCKED_BY_A"
    else:
        status = "READY_TO_SEAL"
    return {"status": status, "a_release_gate": gate, "seal": seal}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.project_root.resolve()
    try:
        if args.command == "generate":
            result = _run_generate(root)
        elif args.command == "validate":
            result = _run_validate(root)
        elif args.command == "seal":
            result = seal_dataset(root)
        else:
            result = _run_status(root)
    except AReleaseGateError as exc:
        _emit({"status": "BLOCKED_BY_A", "error": str(exc)}, stream=sys.stderr)
        return EXIT_BLOCKED
    except (SealError, FileNotFoundError, ValueError) as exc:
        _emit(
            {"status": "FAIL", "error": str(exc), "error_type": type(exc).__name__},
            stream=sys.stderr,
        )
        return EXIT_ERROR
    _emit(result)
    return _result_exit_code(result)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "EXIT_BLOCKED",
    "EXIT_ERROR",
    "EXIT_OK",
    "build_parser",
    "main",
)
