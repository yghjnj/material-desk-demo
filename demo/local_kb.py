"""Local document ingestion and evidence retrieval for the runnable demo.

The service is deliberately local-only. Uploaded bytes stay below ``work`` and
are parsed with Module C before being added to its deterministic FTS5-backed
retriever. No model, network, or external action provider is used.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import re
import sqlite3
from pathlib import Path
from threading import RLock
from uuid import NAMESPACE_URL, UUID, uuid5

from assistant.contracts.base import sha256_bytes, sha256_text
from assistant.contracts.documents import (
    ApplicabilityScope,
    Authority,
    LifecycleAtResult,
    LifecycleEffectiveInterval,
)
from assistant.contracts.enums import (
    ApprovalStatus,
    AuthorityClass,
    DocumentFormat,
    EnvironmentNamespace,
    LifecycleResultStatus,
    LifecycleStatus,
    ProcessingStatus,
)
from assistant.module_c import CDocument, DeterministicRetriever, parse_document
from assistant.module_c.core import tokenize


MAX_UPLOAD_BYTES = 25 * 1024 * 1024
SUPPORTED_SUFFIXES = {".pdf", ".docx", ".md", ".markdown", ".txt"}
_SAFE_NAME = re.compile(r"[^\w.()\- ]", re.UNICODE)


class UploadError(ValueError):
    """A user-correctable upload or parsing error."""


@dataclass(frozen=True)
class DocumentRecord:
    document_id: UUID
    document_version_id: UUID
    filename: str
    title: str
    document_format: DocumentFormat
    source_sha256: str
    version_label: str
    block_count: int
    created_at: str
    path: Path

    def as_json(self) -> dict[str, object]:
        return {
            "document_id": str(self.document_id),
            "document_version_id": str(self.document_version_id),
            "filename": self.filename,
            "title": self.title,
            "document_format": self.document_format.value,
            "source_sha256": self.source_sha256,
            "version_label": self.version_label,
            "block_count": self.block_count,
            "created_at": self.created_at,
            "status": "INDEXED",
            "is_synthetic": False,
        }


def _stable_id(kind: str, value: str) -> UUID:
    return uuid5(NAMESPACE_URL, f"material-desk-local:{kind}:{value}")


def _format_for_suffix(suffix: str) -> DocumentFormat:
    return {
        ".pdf": DocumentFormat.PDF,
        ".docx": DocumentFormat.DOCX,
        ".md": DocumentFormat.MARKDOWN,
        ".markdown": DocumentFormat.MARKDOWN,
        ".txt": DocumentFormat.TEXT,
    }[suffix]


def _safe_filename(filename: str) -> str:
    raw = (filename or "").replace("\x00", "").strip()
    name = Path(raw).name
    if not name or name in {".", ".."}:
        raise UploadError("FILENAME_REQUIRED")
    suffix = Path(name).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise UploadError("UNSUPPORTED_MEDIA_TYPE")
    safe = _SAFE_NAME.sub("_", name)
    return safe[:160]


class LocalKnowledgeBase:
    """Persist local files/metadata and rebuild the Module C index on startup."""

    def __init__(self, storage_dir: str | Path):
        self.storage_dir = Path(storage_dir).resolve()
        self.files_dir = self.storage_dir / "files"
        self.files_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.storage_dir / "index.sqlite3"
        self._lock = RLock()
        self._db = sqlite3.connect(self.db_path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                source_sha256 TEXT PRIMARY KEY,
                document_id TEXT NOT NULL UNIQUE,
                document_version_id TEXT NOT NULL UNIQUE,
                filename TEXT NOT NULL,
                title TEXT NOT NULL,
                document_format TEXT NOT NULL,
                version_label TEXT NOT NULL,
                block_count INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                path TEXT NOT NULL,
                status TEXT NOT NULL
            )
            """
        )
        self._db.commit()
        self._retriever = DeterministicRetriever()
        self._records: dict[str, DocumentRecord] = {}
        self._load_index()

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def _row_to_record(self, row: sqlite3.Row) -> DocumentRecord:
        return DocumentRecord(
            document_id=UUID(row["document_id"]),
            document_version_id=UUID(row["document_version_id"]),
            filename=row["filename"],
            title=row["title"],
            document_format=DocumentFormat(row["document_format"]),
            source_sha256=row["source_sha256"],
            version_label=row["version_label"],
            block_count=row["block_count"],
            created_at=row["created_at"],
            path=Path(row["path"]),
        )

    def _load_index(self) -> None:
        rows = self._db.execute(
            "SELECT * FROM documents WHERE status = 'INDEXED' ORDER BY created_at, filename"
        ).fetchall()
        for row in rows:
            record = self._row_to_record(row)
            if not record.path.is_file():
                continue
            try:
                blocks = parse_document(record.path)
                self._retriever.add(self._to_c_document(record, blocks))
            except (OSError, ValueError, RuntimeError):
                continue
            self._records[record.source_sha256] = record

    @staticmethod
    def _to_c_document(record: DocumentRecord, blocks) -> CDocument:
        now = datetime.now(timezone.utc)
        metadata_hash = sha256_text(
            f"{record.document_id}|{record.document_version_id}|{record.source_sha256}"
        )
        lifecycle = LifecycleAtResult(
            document_version_id=record.document_version_id,
            as_of=now,
            metadata_snapshot_hash=metadata_hash,
            result_status=LifecycleResultStatus.RESOLVED,
            lifecycle_status=LifecycleStatus.ACTIVE,
            effective_interval=LifecycleEffectiveInterval(effective_from=now),
            applied_event_ids=(_stable_id("event", str(record.document_version_id)),),
        )
        return CDocument(
            document_id=record.document_id,
            document_version_id=record.document_version_id,
            runtime_ingestion_revision_id=_stable_id("ingestion", record.source_sha256),
            index_snapshot_id=_stable_id("snapshot", record.source_sha256),
            document_format=record.document_format,
            blocks=tuple(blocks),
            authority=Authority(
                authority_class=AuthorityClass.UNVERIFIED,
                issuer="本地用户上传",
                approval_status=ApprovalStatus.PENDING,
                scope="LOCAL_DEMO_UPLOAD",
            ),
            applicability_scope=ApplicabilityScope(
                languages=("zh", "en"), regions=("LOCAL",), product_refs=("USER_UPLOAD",)
            ),
            precedence=1,
            lifecycle=lifecycle,
            processing_status=ProcessingStatus.INDEXED,
            is_synthetic=False,
        )

    def documents(self) -> list[dict[str, object]]:
        with self._lock:
            return [record.as_json() for record in sorted(self._records.values(), key=lambda x: (x.created_at, x.filename))]

    @staticmethod
    def extract_requirements(text: str) -> dict[str, object]:
        """Return a deterministic, source-located requirement projection."""
        raw = (text or "").strip()
        if not raw:
            raise UploadError("CUSTOMER_MESSAGE_REQUIRED")

        fields: list[dict[str, object]] = []

        def add(key: str, label: str, value: str, start: int, end: int, state: str = "KNOWN") -> None:
            fields.append({
                "key": key,
                "label": label,
                "value": value,
                "state": state,
                "quote": raw[start:end],
                "start": start,
                "end": end,
            })

        def first(pattern: str, key: str, label: str, group: int = 1, state: str = "KNOWN") -> None:
            match = re.search(pattern, raw, re.IGNORECASE)
            if match:
                add(key, label, match.group(group).strip(), match.start(), match.end(), state)

        first(r"(?:用于|应用于|应用场景(?:是|为)?)[\s:：]*([^,，。；;\n]+)", "application_scenario", "应用场景")
        first(r"(?:将|材料(?:是|为)?|产品(?:是|为)?)[\s:：]*([^,，。；;\n]+?)(?=用于|应用于|，|,|。|；|;|$)", "material_or_product", "材料 / 产品")
        first(r"(?:月用量|数量|用量)[^\d]{0,8}(\d+(?:\.\d+)?\s*(?:吨|t|kg|g|L|mL))", "quantity_and_usage", "数量 / 用量")
        first(r"(?:希望|交期|交付|交货)[^\d]{0,10}((?:\d{4}[-/.年])?\d{1,2}\s*月(?:\s*\d{1,2}\s*日)?|\d{4}[-/]\d{1,2}(?:[-/]\d{1,2})?)", "delivery_and_date", "交期")
        first(r"(?:预算)[\s:：]*(待确认|未知|未定)", "budget_and_currency", "预算", state="UNKNOWN")
        first(r"(?:预算)[^\d¥$]*(?:¥|人民币|RMB|\$)?\s*(\d+(?:\.\d+)?(?:\s*[万千百]?元)?)", "budget_and_currency", "预算")
        first(r"(?:工艺|施工方式|工序)[\s:：]*([^,，。；;\n]+)", "process", "工艺")
        first(r"((?:ASTM|ISO|GB/?T|EN)\s*[A-Z]?\s*\d+(?:[.-]\d+)*)", "test_standards", "测试标准")

        performance_terms = ["耐磨", "阻燃", "耐水", "耐候", "附着力", "硬度", "拉伸强度"]
        performance = [term for term in performance_terms if term in raw]
        if performance:
            start = min(raw.index(term) for term in performance)
            end = max(raw.index(term) + len(term) for term in performance)
            add("performance_indicators", "性能指标", "、".join(performance), start, end)

        known = {str(item["key"]): item for item in fields}
        labels = {
            "application_scenario": "应用场景",
            "material_or_product": "材料 / 产品",
            "performance_indicators": "性能指标",
            "quantity_and_usage": "数量 / 用量",
            "delivery_and_date": "交期",
            "budget_and_currency": "预算",
            "process": "工艺",
            "test_standards": "测试标准",
            "compliance": "合规要求",
        }
        for key, label in labels.items():
            if key not in known:
                fields.append({"key": key, "label": label, "value": "未知", "state": "MISSING", "quote": "", "start": None, "end": None})
        missing = [str(item["label"]) for item in fields if item["state"] == "MISSING"]
        risks: list[str] = []
        if "performance_indicators" in known and "test_standards" not in known:
            risks.append("性能指标缺少测试标准，不能直接承诺等级或达标。")
        if "delivery_and_date" in known and re.search(r"\d{1,2}月$", str(known["delivery_and_date"]["value"])):
            risks.append("交期只提供到月份，需确认具体交付日期。")
        if known.get("budget_and_currency", {}).get("state") == "UNKNOWN":
            risks.append("预算仍待客户确认。")
        return {
            "revision": 1,
            "status": "NEEDS_CONFIRMATION" if missing or risks else "COMPLETE",
            "fields": fields,
            "missing": missing,
            "risks": risks,
            "source_text": raw,
        }

    def ingest(self, filename: str, content: bytes) -> dict[str, object]:
        safe_name = _safe_filename(filename)
        if not content:
            raise UploadError("EMPTY_FILE")
        if len(content) > MAX_UPLOAD_BYTES:
            raise UploadError("FILE_TOO_LARGE")
        source_sha256 = sha256_bytes(content)
        with self._lock:
            existing = self._records.get(source_sha256)
            if existing is not None:
                result = existing.as_json()
                result["status"] = "ALREADY_INDEXED"
                return result

            suffix = Path(safe_name).suffix.lower()
            document_id = _stable_id("document", source_sha256)
            version_id = _stable_id("version", source_sha256)
            target = self.files_dir / f"{source_sha256.removeprefix('sha256:')[:16]}-{safe_name}"
            target.write_bytes(content)
            try:
                blocks = parse_document(target)
                record = DocumentRecord(
                    document_id=document_id,
                    document_version_id=version_id,
                    filename=safe_name,
                    title=Path(safe_name).stem,
                    document_format=_format_for_suffix(suffix),
                    source_sha256=source_sha256,
                    version_label="local-1.0",
                    block_count=len(blocks),
                    created_at=datetime.now(timezone.utc).isoformat(),
                    path=target,
                )
                self._retriever.add(self._to_c_document(record, blocks))
                self._db.execute(
                    """
                    INSERT INTO documents
                    (source_sha256, document_id, document_version_id, filename, title,
                     document_format, version_label, block_count, created_at, path, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'INDEXED')
                    """,
                    (
                        record.source_sha256,
                        str(record.document_id),
                        str(record.document_version_id),
                        record.filename,
                        record.title,
                        record.document_format.value,
                        record.version_label,
                        record.block_count,
                        record.created_at,
                        str(record.path),
                    ),
                )
                self._db.commit()
                self._records[source_sha256] = record
                return record.as_json()
            except Exception:
                target.unlink(missing_ok=True)
                raise

    def query(self, question: str) -> dict[str, object]:
        question = (question or "").strip()
        if not question:
            raise UploadError("QUESTION_REQUIRED")
        with self._lock:
            if not self._records:
                return {
                    "status": "REFUSED",
                    "answer": None,
                    "reason": "NO_REGISTERED_DOCUMENTS",
                    "evidence": [],
                }
            expanded = question
            asks_for_name = bool(re.search(r"姓名|名字|主人公|\bname\b", question, re.IGNORECASE))
            if asks_for_name:
                expanded += " 姓名 名字 name"
                # Resume files often put the person's name on the first line
                # without a `姓名:` label. The filename/title gives retrieval
                # a narrow, document-local hint; the answer still comes from
                # the parsed text line, never from the filename itself.
                expanded += " " + " ".join(record.title for record in self._records.values())
            source_hashes = "|".join(sorted(self._records))
            case_id = _stable_id("case", sha256_text(question))
            now = datetime.now(timezone.utc)
            response = self._retriever.answer(
                expanded,
                case_id=case_id,
                as_of=now,
                metadata_snapshot_hash=sha256_text(source_hashes),
                environment_namespace=EnvironmentNamespace.DEVELOPMENT,
            )
            # CJK character-level matching is useful for recall, but common
            # single characters can make an unrelated paragraph look relevant.
            # Keep only evidence sharing a meaningful bigram/Latin token with
            # the question, and require an explicit name label for name asks.
            meaningful_query_tokens = {token for token in tokenize(expanded) if len(token) > 1}
            latin_query_tokens = {
                token for token in tokenize(question)
                if re.fullmatch(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*", token)
            }
            evidence: list[dict[str, object]] = []
            valid_results: list[tuple[object, dict[str, object]]] = []
            by_document = {record.document_id: record for record in self._records.values()}
            for result in response.retrieval_run.results:
                record = by_document.get(result.document_id)
                if record is None:
                    continue
                locator = result.runtime_document_locator
                locator_json = locator.model_dump(mode="json")
                location = self._location_text(record, locator_json)
                quote = str(locator_json["quote"])
                quote_tokens = set(tokenize(quote))
                overlap = meaningful_query_tokens.intersection(quote_tokens)
                latin_overlap = latin_query_tokens.intersection(quote_tokens)
                explicit_name = bool(re.search(r"(?:姓名|名字)\s*[:：]?\s*\S+|\bname\b\s*[:：]?\s*\S+", quote, re.IGNORECASE))
                short_name_line = bool(
                    record.document_format in {DocumentFormat.PDF, DocumentFormat.DOCX}
                    and re.fullmatch(r"[\u3400-\u9fff·]{2,8}", quote.strip())
                )
                if (
                    not overlap
                    or (latin_query_tokens and not latin_overlap)
                    or (asks_for_name and not explicit_name and not short_name_line)
                ):
                    continue
                item = {
                        "filename": record.filename,
                        "title": record.title,
                        "version": record.version_label,
                        "document_id": str(record.document_id),
                        "document_version_id": str(record.document_version_id),
                        "quote": quote,
                        "location": location,
                        "score": result.score,
                        "source_sha256": record.source_sha256,
                        "verification": "EXACT_MATCH",
                    }
                evidence.append(item)
                valid_results.append((result, item))
            if response.outcome != "ANSWERED" or not valid_results:
                return {
                    "status": "REFUSED",
                    "answer": None,
                    "reason": response.reason_code or "INSUFFICIENT_EVIDENCE",
                    "evidence": evidence,
                }
            answer = str(valid_results[0][1]["quote"])
            return {
                "status": "ANSWERED",
                "answer": answer,
                "reason": None,
                "evidence": evidence,
            }

    @staticmethod
    def _location_text(record: DocumentRecord, locator: dict[str, object]) -> str:
        if record.document_format is DocumentFormat.PDF:
            return f"PDF 第 {int(locator['page_index']) + 1} 页"
        if record.document_format is DocumentFormat.DOCX:
            section = " / ".join(locator.get("section_path") or ("ROOT",))
            return f"DOCX 段落 {int(locator['paragraph_index']) + 1} · {section}"
        if record.document_format is DocumentFormat.MARKDOWN:
            section = " / ".join(locator.get("section_path") or ("ROOT",))
            return f"Markdown 块 {int(locator['block_index']) + 1} · {section}"
        return f"TXT 第 {int(locator['line_index']) + 1} 行"
