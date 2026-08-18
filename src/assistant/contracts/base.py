"""Primitive, side-effect-free contract types."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import unicodedata
from typing import Annotated, Any, Literal, TypeVar
from uuid import UUID

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

CONTRACT_VERSION = "1.1.0"
API_VERSION = "v1"
LOCATOR_PROFILE_VERSION = "1.0.0"
HASH_PROFILE_VERSION = "1.0.0"


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware UTC")
    if value.utcoffset().total_seconds() != 0:
        raise ValueError("timestamp must use UTC")
    return value.astimezone(timezone.utc)


UtcDatetime = Annotated[datetime, AfterValidator(_utc)]
UTCDateTime = UtcDatetime
Sha256 = Annotated[
    str,
    StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$"),
]
Sha256Digest = Sha256
DecimalString = Annotated[
    str,
    StringConstraints(pattern=r"^-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?$"),
]
NonEmptyString = Annotated[str, StringConstraints(min_length=1)]
SemanticVersion = Annotated[
    str,
    StringConstraints(
        pattern=r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$"
    ),
]
LanguageTag = Annotated[
    str,
    StringConstraints(pattern=r"^[A-Za-z]{2,8}(?:-[A-Za-z0-9]{1,8})*$"),
]
UUIDString = UUID
NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(gt=0)]
Confidence = Annotated[float, Field(ge=0.0, le=1.0)]
JsonPointer = Annotated[str, StringConstraints(pattern=r"^/(?:[^/~]|~[01])+(?:/(?:[^/~]|~[01])+)*$")]

_T = TypeVar("_T")


class FrozenDict(dict[str, _T]):
    """A JSON-schema-friendly mapping that rejects every in-place mutation."""

    @staticmethod
    def _immutable(*_args: Any, **_kwargs: Any) -> None:
        raise TypeError("contract mappings are immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable

    def __deepcopy__(self, _memo: dict[int, Any]) -> "FrozenDict[_T]":
        return self


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, FrozenDict):
        return value
    if isinstance(value, dict):
        return FrozenDict({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_deep_freeze(item) for item in value)
    return value


class ContractModel(BaseModel):
    """Base for immutable, strict public payloads."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_default=True,
        revalidate_instances="always",
    )

    @model_validator(mode="after")
    def freeze_nested_containers(self) -> "ContractModel":
        for field_name in type(self).model_fields:
            value = getattr(self, field_name)
            frozen = _deep_freeze(value)
            if frozen is not value:
                object.__setattr__(self, field_name, frozen)
        return self


class VersionedContract(ContractModel):
    schema_version: Literal["1.1.0"] = CONTRACT_VERSION


class Provenance(ContractModel):
    producer: NonEmptyString
    producer_version: NonEmptyString
    model_id: str | None = None
    prompt_version: str | None = None
    config_hash: Sha256 | None = None


class EntityRef(ContractModel):
    entity_id: UUID
    revision: int = Field(ge=1)


class VersionRef(ContractModel):
    version: NonEmptyString
    content_hash: Sha256


class KeyValue(ContractModel):
    key: NonEmptyString
    value: Any


def sha256_text(value: str) -> str:
    """Hash NFC UTF-8 text using the contract representation."""

    normalized = unicodedata.normalize("NFC", value).replace("\r\n", "\n").replace("\r", "\n")
    return "sha256:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def half_open_range_is_valid(start: int, end: int, length: int) -> bool:
    return 0 <= start <= end <= length
