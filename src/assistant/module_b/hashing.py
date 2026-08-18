"""Deterministic identifiers and hashes for Module B generated assets.

The A-owned :class:`ExecutionNamespaceKey` is constructed exactly as defined
by A-CONTRACT-v1.0.3.  B dataset context is deliberately kept outside that
key and therefore cannot change its canonical hash.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence, Set
from datetime import date, datetime, time, timezone
from decimal import Decimal
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
import unicodedata
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import BaseModel

from ..contracts.enums import EnvironmentNamespace
from ..contracts.runtime import ExecutionNamespaceKey


CANONICAL_JSON_ENCODING = "utf-8"
CANONICAL_JSON_SEPARATORS = (",", ":")


def _normalized_string(value: str) -> str:
    """Return the dataset-wide NFC and LF representation of text."""

    return unicodedata.normalize("NFC", value).replace("\r\n", "\n").replace("\r", "\n")


def _jsonable(value: Any) -> Any:
    """Convert supported values to an unambiguous JSON-compatible tree."""

    if isinstance(value, BaseModel):
        return _jsonable(value.model_dump(mode="python", by_alias=True, exclude_none=False))
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, str):
        return _normalized_string(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical JSON forbids NaN and infinity")
        return value
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("canonical JSON forbids non-finite Decimal values")
        return format(value, "f")
    if isinstance(value, Enum):
        return _jsonable(value.value)
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("canonical JSON requires timezone-aware datetimes")
        utc_value = value.astimezone(timezone.utc)
        return utc_value.isoformat(timespec="microseconds").replace("+00:00", "Z")
    if isinstance(value, (date, time)):
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, bytes):
        return {"__bytes_hex__": value.hex()}
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for raw_key, item in value.items():
            if not isinstance(raw_key, str):
                raise TypeError("canonical JSON mapping keys must be strings")
            key = _normalized_string(raw_key)
            if key in normalized:
                raise ValueError("mapping keys collide after Unicode normalization")
            normalized[key] = _jsonable(item)
        return normalized
    if isinstance(value, Set):
        converted = [_jsonable(item) for item in value]
        return sorted(converted, key=_canonical_sort_key)
    if isinstance(value, Sequence):
        return [_jsonable(item) for item in value]
    raise TypeError(f"unsupported canonical JSON value: {type(value).__qualname__}")


def _canonical_sort_key(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=CANONICAL_JSON_SEPARATORS,
        allow_nan=False,
    ).encode(CANONICAL_JSON_ENCODING)


def canonical_json(value: Any) -> str:
    """Serialize ``value`` using the stable Module B canonical JSON profile."""

    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=CANONICAL_JSON_SEPARATORS,
        allow_nan=False,
    )


def canonical_json_bytes(value: Any) -> bytes:
    return canonical_json(value).encode(CANONICAL_JSON_ENCODING)


def sha256_bytes(value: bytes) -> str:
    """Return an A-compatible, prefixed SHA-256 digest."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    """Hash normalized NFC/LF UTF-8 text."""

    return sha256_bytes(_normalized_string(value).encode(CANONICAL_JSON_ENCODING))


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def model_sha256(value: BaseModel) -> str:
    return sha256_json(value)


def file_sha256(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash exact file bytes without normalizing their contents."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def deterministic_uuid(*parts: Any, namespace: UUID = NAMESPACE_URL) -> UUID:
    """Create a stable UUIDv5 over a canonical, boundary-safe parts array."""

    if not parts:
        raise ValueError("deterministic UUID requires at least one name part")
    return uuid5(namespace, canonical_json(list(parts)))


def execution_namespace_payload(
    *,
    environment: EnvironmentNamespace | str,
    corpus_manifest_hash: str,
    split_manifest_hash: str,
    document_version_set_hash: str,
    source_hash_set_hash: str,
    contract_bundle_hash: str,
    configuration_hash: str,
    code_hash: str,
    run_id: UUID | str,
) -> dict[str, str]:
    """Return only the nine A-frozen canonical namespace dimensions."""

    environment_value = EnvironmentNamespace(environment).value
    run_uuid = run_id if isinstance(run_id, UUID) else UUID(str(run_id))
    return {
        "environment": environment_value,
        "corpus_manifest_hash": corpus_manifest_hash,
        "split_manifest_hash": split_manifest_hash,
        "document_version_set_hash": document_version_set_hash,
        "source_hash_set_hash": source_hash_set_hash,
        "contract_bundle_hash": contract_bundle_hash,
        "configuration_hash": configuration_hash,
        "code_hash": code_hash,
        "run_id": str(run_uuid),
    }


def execution_namespace_hash(**kwargs: Any) -> str:
    """Hash the exact A namespace payload; no B extension is included."""

    return sha256_json(execution_namespace_payload(**kwargs))


def build_execution_namespace_key(
    *,
    environment: EnvironmentNamespace | str,
    corpus_manifest_hash: str,
    split_manifest_hash: str,
    document_version_set_hash: str,
    source_hash_set_hash: str,
    contract_bundle_hash: str,
    configuration_hash: str,
    code_hash: str,
    run_id: UUID | str,
) -> ExecutionNamespaceKey:
    """Validate and construct the A-owned ``ExecutionNamespaceKey``."""

    payload = execution_namespace_payload(
        environment=environment,
        corpus_manifest_hash=corpus_manifest_hash,
        split_manifest_hash=split_manifest_hash,
        document_version_set_hash=document_version_set_hash,
        source_hash_set_hash=source_hash_set_hash,
        contract_bundle_hash=contract_bundle_hash,
        configuration_hash=configuration_hash,
        code_hash=code_hash,
        run_id=run_id,
    )
    return ExecutionNamespaceKey(
        **payload,
        namespace_hash=sha256_json(payload),
    )


def verify_execution_namespace_key(key: ExecutionNamespaceKey) -> bool:
    """Recompute the A canonical hash without inspecting B dataset context."""

    payload = execution_namespace_payload(
        environment=key.environment,
        corpus_manifest_hash=key.corpus_manifest_hash,
        split_manifest_hash=key.split_manifest_hash,
        document_version_set_hash=key.document_version_set_hash,
        source_hash_set_hash=key.source_hash_set_hash,
        contract_bundle_hash=key.contract_bundle_hash,
        configuration_hash=key.configuration_hash,
        code_hash=key.code_hash,
        run_id=key.run_id,
    )
    return sha256_json(payload) == key.namespace_hash


__all__ = (
    "CANONICAL_JSON_ENCODING",
    "CANONICAL_JSON_SEPARATORS",
    "build_execution_namespace_key",
    "canonical_json",
    "canonical_json_bytes",
    "deterministic_uuid",
    "execution_namespace_hash",
    "execution_namespace_payload",
    "file_sha256",
    "model_sha256",
    "sha256_bytes",
    "sha256_json",
    "sha256_text",
    "verify_execution_namespace_key",
)
