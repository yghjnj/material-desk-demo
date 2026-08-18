"""A-gated, tamper-evident sealing for Module B holdout Gold.

The seal is a logical delivery boundary, not an operating-system ACL. Only
the H reader exposed here may return sealed payload contents. Other stages
can inspect the irreversible verification report and digests only.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any
from uuid import uuid4

from .hashing import file_sha256, sha256_bytes


SEAL_SCHEMA_VERSION = "1.0.0"
REQUIRED_A_CONTRACT_VERSION = "1.1.0"
_SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_SEAL_FILENAMES = (
    "sealed_gold.payload.json",
    "sealed_gold.sha256",
    "sealed_gold.lock",
    "h_handoff_manifest.json",
)


class SealError(RuntimeError):
    """Base error for a refused or invalid seal operation."""


class AReleaseGateError(SealError):
    """Raised when A's current control-plane evidence does not release B."""


class SealIntegrityError(SealError):
    """Raised when immutable seal artifacts are incomplete or inconsistent."""


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _recompute_a_aggregate(project_root: Path) -> str | None:
    """Recompute the real A-owned content hash when its registry is present."""

    allowlist_path = project_root / "docs" / "control" / "module-a-file-allowlist-v1.0.json"
    receipt_path = project_root / "docs" / "control" / "implementation-receipt-v1.0.json"
    if not allowlist_path.is_file() or not receipt_path.is_file():
        # Isolated Module B fixtures intentionally do not carry A's registry.
        return None
    allowlist = _read_json(allowlist_path)
    receipt = _read_json(receipt_path)
    manifest = receipt.get("content_manifest", {})
    persistent = allowlist.get("persistent_files") if isinstance(allowlist, Mapping) else None
    if not isinstance(persistent, list) or not isinstance(manifest, Mapping):
        raise SealIntegrityError("A content manifest or allowlist is malformed")
    excluded = set(manifest.get("excluded_self_reporting_files", ())) | set(
        manifest.get("excluded_control_plane_files", ())
    )
    paths = sorted(str(item).replace("\\", "/") for item in persistent if item not in excluded)
    lines: list[str] = []
    for relative in paths:
        path = project_root / Path(relative)
        if not path.is_file():
            raise SealIntegrityError(f"A allowlisted file is missing: {relative}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{relative}|{digest}")
    return "sha256:" + hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _seal_paths(project_root: str | Path) -> dict[str, Path]:
    seal_root = Path(project_root).resolve() / "work" / "B" / "seal"
    return {
        "payload": seal_root / _SEAL_FILENAMES[0],
        "hash": seal_root / _SEAL_FILENAMES[1],
        "lock": seal_root / _SEAL_FILENAMES[2],
        "handoff": seal_root / _SEAL_FILENAMES[3],
    }


def _is_redacted_placeholder(path: Path) -> bool:
    """Recognize the control-plane placeholder that contains no holdout Gold."""

    if not path.is_file():
        return False
    try:
        value = _read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return (
        isinstance(value, Mapping)
        and value.get("artifact_role") == "SEALED_PAYLOAD_PLACEHOLDER"
        and value.get("content_state") == "REDACTED"
        and value.get("contains_holdout_gold") is False
    )


def check_a_release_gate(project_root: str | Path) -> dict[str, Any]:
    """Read the control plane and return whether A 1.1.0 releases B.

    This function is deliberately schema-light: it reads only the existing
    control fields and does not redefine an A-owned model. Missing, stale, or
    contradictory evidence is closed by default.
    """

    root = Path(project_root).resolve()
    stage_path = root / "docs" / "control" / "stage-status-v1.0.json"
    receipt_path = root / "docs" / "control" / "implementation-receipt-v1.0.json"
    reasons: list[str] = []
    stage: Mapping[str, Any] = {}
    receipt: Mapping[str, Any] = {}

    try:
        loaded_stage = _read_json(stage_path)
        if not isinstance(loaded_stage, Mapping):
            raise TypeError("stage status must be a JSON object")
        stage = loaded_stage
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        reasons.append(f"unreadable A stage status: {exc}")

    try:
        loaded_receipt = _read_json(receipt_path)
        if not isinstance(loaded_receipt, Mapping):
            raise TypeError("implementation receipt must be a JSON object")
        receipt = loaded_receipt
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        reasons.append(f"unreadable A implementation receipt: {exc}")

    stages = stage.get("stages", {}) if isinstance(stage, Mapping) else {}
    a_stage = stages.get("A", {}) if isinstance(stages, Mapping) else {}
    b_stage = stages.get("B", {}) if isinstance(stages, Mapping) else {}
    a_status = str(a_stage.get("status", "MISSING")).upper()
    b_status = str(b_stage.get("status", "MISSING")).upper()
    next_stage = str(stage.get("next_stage", "MISSING")).upper()
    if a_status != "PASS":
        reasons.append(f"A stage status is {a_status}, expected PASS")
    if next_stage != "B" and not next_stage.startswith("B_"):
        reasons.append(f"next_stage is {next_stage}, expected B")
    if "A_REMEDIATION" in b_status or b_status.startswith("BLOCKED_BY_A"):
        reasons.append(f"B stage remains blocked by A: {b_status}")

    receipt_status = str(receipt.get("status", "MISSING")).upper()
    contract_version = str(receipt.get("contract_version", "MISSING"))
    hash_status = str(receipt.get("hash_status", "MISSING")).upper()
    manifest = receipt.get("content_manifest", {})
    aggregate_hash = (
        manifest.get("aggregate_sha256") if isinstance(manifest, Mapping) else None
    )
    if receipt_status != "PASS":
        reasons.append(f"A receipt status is {receipt_status}, expected PASS")
    if contract_version != REQUIRED_A_CONTRACT_VERSION:
        reasons.append(
            f"A contract version is {contract_version}, expected "
            f"{REQUIRED_A_CONTRACT_VERSION}"
        )
    if hash_status not in {"CURRENT", "VALID", "VERIFIED"}:
        reasons.append(f"A receipt hash status is {hash_status}, expected current")
    if not isinstance(aggregate_hash, str) or not _SHA256_PATTERN.fullmatch(
        aggregate_hash
    ):
        reasons.append("A receipt aggregate_sha256 is missing or malformed")

    recomputed_aggregate: str | None = None
    try:
        recomputed_aggregate = _recompute_a_aggregate(root)
    except (OSError, TypeError, ValueError, json.JSONDecodeError, SealIntegrityError) as exc:
        reasons.append(f"A aggregate hash cannot be independently verified: {exc}")
    if (
        recomputed_aggregate is not None
        and isinstance(aggregate_hash, str)
        and recomputed_aggregate != aggregate_hash
    ):
        reasons.append(
            "A receipt aggregate_sha256 does not match the current allowlisted files"
        )

    return {
        "status": "PASS" if not reasons else "BLOCKED",
        "required_contract_version": REQUIRED_A_CONTRACT_VERSION,
        "observed": {
            "a_stage_status": a_status,
            "b_stage_status": b_status,
            "next_stage": next_stage,
            "receipt_status": receipt_status,
            "contract_version": contract_version,
            "hash_status": hash_status,
            "aggregate_sha256": aggregate_hash,
            "recomputed_aggregate_sha256": recomputed_aggregate,
        },
        "reasons": reasons,
    }


def _require_a_release(project_root: str | Path) -> dict[str, Any]:
    gate = check_a_release_gate(project_root)
    if gate["status"] != "PASS":
        detail = "; ".join(gate["reasons"])
        raise AReleaseGateError(f"A 1.1.0 release gate is closed: {detail}")
    return gate


def _validate_payload_shape(payload: Any) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise SealIntegrityError("sealed Gold payload must be a JSON object")
    if payload.get("split") != "SEALED_HOLDOUT":
        raise SealIntegrityError("sealed Gold payload must use SEALED_HOLDOUT split")
    if payload.get("gold_access") != "H_ONLY_AFTER_SEAL":
        raise SealIntegrityError("sealed Gold payload must declare H_ONLY_AFTER_SEAL")
    for collection in (
        "root_scenarios",
        "case_graphs",
        "task_instances",
        "customer_messages",
        "expected_task_gold",
    ):
        if not isinstance(payload.get(collection), list):
            raise SealIntegrityError(f"sealed Gold payload requires {collection} list")
    for collection in ("root_scenarios", "case_graphs", "task_instances"):
        if any(
            not isinstance(item, Mapping)
            or item.get("split") != "SEALED_HOLDOUT"
            for item in payload[collection]
        ):
            raise SealIntegrityError(
                f"{collection} contains an item outside SEALED_HOLDOUT"
            )
    return payload


def verify_seal(project_root: str | Path) -> dict[str, Any]:
    """Verify payload bytes, checksum record, immutable lock, and H handoff.

    The payload is intentionally not deserialized here. This verifier is safe
    for B through G because it returns digests and errors only; H performs the
    sole post-seal payload read through :func:`read_sealed_gold`.
    """

    paths = _seal_paths(project_root)
    present = {name for name, path in paths.items() if path.is_file()}
    if not present:
        return {"status": "UNSEALED", "valid": False, "errors": ["seal absent"]}
    if present == {"payload"} and _is_redacted_placeholder(paths["payload"]):
        return {
            "status": "UNSEALED",
            "valid": False,
            "placeholder": True,
            "errors": ["redacted placeholder is not a seal"],
        }
    missing = sorted(set(paths) - present)
    if missing:
        return {
            "status": "FAIL",
            "valid": False,
            "errors": [f"partial seal; missing {', '.join(missing)}"],
        }

    errors: list[str] = []
    hash_record: Any = None
    lock: Any = None
    handoff: Any = None
    for name in ("hash", "lock", "handoff"):
        try:
            value = _read_json(paths[name])
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{name} is unreadable JSON: {exc}")
            continue
        if name == "hash":
            hash_record = value
        elif name == "lock":
            lock = value
        else:
            handoff = value

    payload_sha256 = file_sha256(paths["payload"])
    hash_file_sha256 = file_sha256(paths["hash"])
    lock_sha256 = file_sha256(paths["lock"])

    if not isinstance(hash_record, Mapping):
        errors.append("hash record must be a JSON object")
    else:
        if hash_record.get("algorithm") != "SHA-256":
            errors.append("hash record algorithm mismatch")
        if hash_record.get("payload_file") != paths["payload"].name:
            errors.append("hash record payload_file mismatch")
        if hash_record.get("payload_sha256") != payload_sha256:
            errors.append("payload hash mismatch")

    if not isinstance(lock, Mapping):
        errors.append("lock must be a JSON object")
    else:
        expected_lock = {
            "schema_version": SEAL_SCHEMA_VERSION,
            "status": "SEALED",
            "immutable": True,
            "gold_access": "H_ONLY",
            "payload_file": paths["payload"].name,
            "payload_sha256": payload_sha256,
            "hash_file": paths["hash"].name,
            "hash_file_sha256": hash_file_sha256,
        }
        for field, expected in expected_lock.items():
            if lock.get(field) != expected:
                errors.append(f"lock {field} mismatch")
        a_release = lock.get("a_release")
        if not isinstance(a_release, Mapping):
            errors.append("lock A release binding is missing")
        else:
            if a_release.get("contract_version") != REQUIRED_A_CONTRACT_VERSION:
                errors.append("lock A contract version mismatch")
            aggregate = a_release.get("aggregate_sha256")
            if not isinstance(aggregate, str) or not _SHA256_PATTERN.fullmatch(
                aggregate
            ):
                errors.append("lock A aggregate hash is malformed")

    if not isinstance(handoff, Mapping):
        errors.append("H handoff must be a JSON object")
    else:
        expected_handoff = {
            "schema_version": SEAL_SCHEMA_VERSION,
            "audience": "H",
            "gold_access": "H_ONLY",
            "payload_file": paths["payload"].name,
            "payload_sha256": payload_sha256,
            "hash_file": paths["hash"].name,
            "hash_file_sha256": hash_file_sha256,
            "lock_file": paths["lock"].name,
            "lock_sha256": lock_sha256,
        }
        for field, expected in expected_handoff.items():
            if handoff.get(field) != expected:
                errors.append(f"H handoff {field} mismatch")
        if isinstance(lock, Mapping) and handoff.get("a_release") != lock.get(
            "a_release"
        ):
            errors.append("H handoff A release binding mismatch")

    if errors:
        return {
            "status": "FAIL",
            "valid": False,
            "payload_sha256": payload_sha256,
            "errors": errors,
        }
    return {
        "status": "VERIFIED",
        "valid": True,
        "payload_sha256": payload_sha256,
        "hash_file_sha256": hash_file_sha256,
        "lock_sha256": lock_sha256,
        "errors": [],
    }


def seal_dataset(
    project_root: str | Path,
) -> dict[str, Any]:
    """Create one immutable seal, or return the verified existing seal.

    A closed release gate performs no generation and writes no payload. When
    the generator is invoked in its normal in-memory holdout mode and the
    complete pre-seal validator must pass before any Gold bytes are written.
    """

    root = Path(project_root).resolve()
    gate = _require_a_release(root)
    paths = _seal_paths(root)
    present = {name for name, path in paths.items() if path.exists()}
    placeholder_bytes: bytes | None = None
    if present == {"payload"} and _is_redacted_placeholder(paths["payload"]):
        placeholder_bytes = paths["payload"].read_bytes()
        present.clear()
    if present:
        if present != set(paths):
            missing = ", ".join(sorted(set(paths) - present))
            raise SealIntegrityError(
                f"partial immutable seal cannot be repaired; missing {missing}"
            )
        report = verify_seal(root)
        if not report["valid"]:
            raise SealIntegrityError(
                "existing sealed Gold failed integrity verification: "
                + "; ".join(report["errors"])
            )
        return report | {"status": "SEALED", "idempotent": True}

    from .generation import generate_all
    from .validation import require_valid

    generated = generate_all(root, emit_sealed_payload=False)
    payload = generated["sealed_gold"]
    _validate_payload_shape(payload)
    validation = require_valid(
        root,
        sealed_payload=payload,
        write_reports=True,
        require_all_allowlisted=False,
    )

    payload_bytes = _json_bytes(payload)
    payload_sha256 = sha256_bytes(payload_bytes)
    hash_record = {
        "algorithm": "SHA-256",
        "payload_file": paths["payload"].name,
        "payload_sha256": payload_sha256,
    }
    hash_bytes = _json_bytes(hash_record)
    hash_file_sha256 = sha256_bytes(hash_bytes)
    observed = gate["observed"]
    a_release = {
        "contract_version": observed["contract_version"],
        "aggregate_sha256": observed["aggregate_sha256"],
    }
    lock = {
        "schema_version": SEAL_SCHEMA_VERSION,
        "status": "SEALED",
        "immutable": True,
        "gold_access": "H_ONLY",
        "payload_file": paths["payload"].name,
        "payload_sha256": payload_sha256,
        "hash_file": paths["hash"].name,
        "hash_file_sha256": hash_file_sha256,
        "a_release": a_release,
    }
    lock_bytes = _json_bytes(lock)
    lock_sha256 = sha256_bytes(lock_bytes)
    handoff = {
        "schema_version": SEAL_SCHEMA_VERSION,
        "audience": "H",
        "gold_access": "H_ONLY",
        "payload_file": paths["payload"].name,
        "payload_sha256": payload_sha256,
        "hash_file": paths["hash"].name,
        "hash_file_sha256": hash_file_sha256,
        "lock_file": paths["lock"].name,
        "lock_sha256": lock_sha256,
        "a_release": a_release,
    }

    written: list[Path] = []
    try:
        for name, content in (
            ("payload", payload_bytes),
            ("hash", hash_bytes),
            ("lock", lock_bytes),
            ("handoff", _json_bytes(handoff)),
        ):
            _atomic_write(paths[name], content)
            written.append(paths[name])
    except Exception:
        for path in reversed(written):
            path.unlink(missing_ok=True)
        if placeholder_bytes is not None:
            _atomic_write(paths["payload"], placeholder_bytes)
        raise

    report = verify_seal(root)
    if not report["valid"]:
        raise SealIntegrityError(
            "new seal failed post-write verification: " + "; ".join(report["errors"])
        )
    return report | {
        "status": "SEALED",
        "idempotent": False,
        "validation_status": validation["status"],
    }


def read_sealed_gold(
    project_root: str | Path,
    reader_stage: str,
) -> Any:
    """Return sealed Gold to H only, after complete integrity verification."""

    normalized_stage = str(getattr(reader_stage, "value", reader_stage)).strip().upper()
    if normalized_stage != "H":
        raise PermissionError("sealed Gold reader is restricted to stage H")
    report = verify_seal(project_root)
    if not report["valid"]:
        raise SealIntegrityError(
            "sealed Gold is unavailable or invalid: " + "; ".join(report["errors"])
        )
    payload = _read_json(_seal_paths(project_root)["payload"])
    _validate_payload_shape(payload)
    return payload


__all__ = (
    "AReleaseGateError",
    "REQUIRED_A_CONTRACT_VERSION",
    "SEAL_SCHEMA_VERSION",
    "SealError",
    "SealIntegrityError",
    "check_a_release_gate",
    "read_sealed_gold",
    "seal_dataset",
    "verify_seal",
)
