"""Zero-install lexical RAG baseline.

Documents are untrusted data. This module never executes document/query text and
never calls model, network, filesystem-write, or action providers.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from pathlib import Path
import re
import sqlite3
import unicodedata
from uuid import UUID, NAMESPACE_URL, uuid5

import fitz
from docx import Document

from assistant.contracts.base import sha256_bytes, sha256_text
from assistant.contracts.document_locators import (
    RuntimeDocxLocator, RuntimeMarkdownLocator, RuntimePdfLocator, RuntimeTextLocator,
)
from assistant.contracts.documents import ApplicabilityScope, Authority, LifecycleAtResult
from assistant.contracts.enums import (
    AuthorityClass, CitationRelation, DocumentFormat, FactKind, LifecycleResultStatus,
    LifecycleStatus, ProcessingStatus, SupportLevel, VerificationStatus,
)
from assistant.contracts.evidence import TechnicalCitation
from assistant.contracts.facts import RuntimeKnowledgeFact
from assistant.contracts.retrieval import RetrievalFilters, RetrievalResult, RetrievalRun

PARSER_VERSION = "c-parser-1.0.0"
CANONICALIZER_VERSION = "c-canonical-1.0.0"
CHUNKING_VERSION = "c-block-1.0.0"
INDEX_VERSION = "c-lexical-1.0.0"
_CJK = re.compile(r"[\u3400-\u9fff]")
_LATIN = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*")


def canonicalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in text.split("\n")).strip()


def _id(kind: str, *parts: object) -> UUID:
    return uuid5(NAMESPACE_URL, "module-c:" + kind + ":" + ":".join(map(str, parts)))


def tokenize(text: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", text).lower()
    latin = _LATIN.findall(normalized)
    chars = [c for c in normalized if _CJK.fullmatch(c)]
    bigrams = ["".join(chars[i:i + 2]) for i in range(len(chars) - 1)]
    return tuple(latin + chars + bigrams)


@dataclass(frozen=True)
class ParsedBlock:
    text: str
    kind: str
    ordinal: int
    section_path: tuple[str, ...] = ()
    page_index: int | None = None


def parse_document(path: str | Path) -> tuple[ParsedBlock, ...]:
    path = Path(path)
    suffix = path.suffix.lower()
    blocks: list[ParsedBlock] = []
    if suffix == ".pdf":
        with fitz.open(path) as doc:
            for i, page in enumerate(doc):
                text = canonicalize(page.get_text("text"))
                if text:
                    blocks.append(ParsedBlock(text, "page", i, page_index=i))
    elif suffix == ".docx":
        headings: list[str] = []
        for i, paragraph in enumerate(Document(path).paragraphs):
            text = canonicalize(paragraph.text)
            if not text:
                continue
            if paragraph.style and paragraph.style.name.startswith("Heading"):
                headings = [text]
            blocks.append(ParsedBlock(text, "paragraph", i, tuple(headings)))
    elif suffix in {".md", ".markdown"}:
        headings: list[str] = []
        for i, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines()):
            text = canonicalize(raw)
            if not text:
                continue
            match = re.match(r"^(#{1,6})\s+(.+)$", text)
            if match:
                level = len(match.group(1)); headings = headings[:level-1] + [match.group(2)]
                kind = "heading"
            else:
                kind = "block"
            blocks.append(ParsedBlock(text, kind, i, tuple(headings)))
    elif suffix == ".txt":
        for i, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines()):
            text = canonicalize(raw)
            if text:
                blocks.append(ParsedBlock(text, "line", i))
    else:
        raise ValueError("UNSUPPORTED_MEDIA_TYPE")
    if not blocks:
        raise ValueError("EMPTY_CANONICAL_TEXT")
    return tuple(blocks)


@dataclass(frozen=True)
class CConfig:
    top_k: int = 5
    min_score: float = 0.05
    rrf_k: int = 60


@dataclass(frozen=True)
class CDocument:
    document_id: UUID
    document_version_id: UUID
    runtime_ingestion_revision_id: UUID
    index_snapshot_id: UUID
    document_format: DocumentFormat
    blocks: tuple[ParsedBlock, ...]
    authority: Authority
    applicability_scope: ApplicabilityScope
    precedence: int
    lifecycle: LifecycleAtResult
    processing_status: ProcessingStatus = ProcessingStatus.INDEXED
    is_synthetic: bool = True


@dataclass(frozen=True)
class CResponse:
    outcome: str
    claims: tuple[str, ...]
    citations: tuple[TechnicalCitation, ...]
    retrieval_run: RetrievalRun
    reason_code: str | None = None


@dataclass(frozen=True)
class _Entry:
    fact: RuntimeKnowledgeFact
    locator: object
    tokens: tuple[str, ...]
    eligible: bool
    exclusion: tuple[str, ...]


class DeterministicRetriever:
    """SQLite FTS5 candidate store plus reproducible BM25/RRF lexical scoring."""
    def __init__(self, config: CConfig = CConfig()):
        self.config = config
        self._entries: list[_Entry] = []
        self._db = sqlite3.connect(":memory:")
        self._db.execute("CREATE VIRTUAL TABLE lexical USING fts5(tokens, content='')")

    def add(self, document: CDocument) -> None:
        eligible = (
            document.processing_status is ProcessingStatus.INDEXED
            and document.lifecycle.result_status is LifecycleResultStatus.RESOLVED
            and document.lifecycle.lifecycle_status is LifecycleStatus.ACTIVE
        )
        exclusion = () if eligible else ("NOT_ACTIVE_OR_INDEXED",)
        canonical = "\n".join(b.text for b in document.blocks)
        canonical_hash = sha256_text(canonical)
        for block in document.blocks:
            quote = block.text
            chunk_id = _id("chunk", document.document_version_id, block.ordinal, sha256_text(quote))
            common = dict(
                runtime_locator_id=_id("locator", chunk_id), document_id=document.document_id,
                document_version_id=document.document_version_id, canonical_text_sha256=canonical_hash,
                chunk_id=chunk_id, chunk_sha256=sha256_text(quote), chunk_char_start=0,
                chunk_char_end=len(quote), parser_id="module_c", parser_version=PARSER_VERSION,
                canonicalizer_version=CANONICALIZER_VERSION, quote=quote, quote_sha256=sha256_text(quote),
            )
            if document.document_format is DocumentFormat.PDF:
                locator = RuntimePdfLocator(**common, page_index=block.page_index or 0,
                    page_canonical_text_sha256=sha256_text(quote), page_char_start=0, page_char_end=len(quote))
            elif document.document_format is DocumentFormat.DOCX:
                locator = RuntimeDocxLocator(**common, section_path=block.section_path or ("ROOT",),
                    paragraph_index=block.ordinal, paragraph_text_sha256=sha256_text(quote),
                    paragraph_char_start=0, paragraph_char_end=len(quote))
            elif document.document_format is DocumentFormat.MARKDOWN:
                locator = RuntimeMarkdownLocator(**common, section_path=block.section_path or ("ROOT",),
                    block_index=block.ordinal, block_kind=block.kind, block_text_sha256=sha256_text(quote),
                    block_char_start=0, block_char_end=len(quote))
            else:
                locator = RuntimeTextLocator(**common, line_index=block.ordinal,
                    line_text_sha256=sha256_text(quote), line_char_start=0, line_char_end=len(quote))
            fact = RuntimeKnowledgeFact(
                runtime_fact_id=_id("fact", chunk_id), runtime_ingestion_revision_id=document.runtime_ingestion_revision_id,
                index_snapshot_id=document.index_snapshot_id, document_id=document.document_id,
                document_version_id=document.document_version_id, fact_key=f"chunk:{chunk_id}",
                fact_kind=FactKind.SPECIFICATION, statement=quote, statement_sha256=sha256_text(quote),
                authority=document.authority, precedence=document.precedence,
                precedence_policy_version="A-1.1.0", applicability_scope=document.applicability_scope,
                runtime_document_locators=(locator,), is_synthetic=document.is_synthetic,
                provenance=("module_c:deterministic_extraction",),
            )
            tokens = tokenize(quote)
            self._entries.append(_Entry(fact, locator, tokens, eligible, exclusion))
            self._db.execute("INSERT INTO lexical(rowid,tokens) VALUES (?,?)", (len(self._entries), " ".join(tokens)))
        self._db.commit()

    def retrieve(self, query: str, *, case_id: UUID, as_of: datetime,
                 metadata_snapshot_hash: str, environment_namespace, top_k: int | None = None) -> RetrievalRun:
        top_k = top_k or self.config.top_k
        q = tokenize(query); n = len(self._entries)
        df = {t: sum(t in e.tokens for e in self._entries) for t in set(q)}
        ranked: list[tuple[float, str, int]] = []
        for i, entry in enumerate(self._entries):
            counts = {t: entry.tokens.count(t) for t in set(q)}
            score = sum((1 + math.log(c)) * math.log(1 + (n + 1) / (df[t] + 1)) for t, c in counts.items() if c)
            if score >= self.config.min_score:
                ranked.append((score, str(entry.fact.runtime_fact_id), i))
        ranked.sort(key=lambda x: (-x[0], x[1]))
        run_id = _id("run", case_id, sha256_text(query), as_of.isoformat(), metadata_snapshot_hash)
        results = []
        for rank, (score, _, i) in enumerate(ranked[:top_k], 1):
            entry = self._entries[i]
            results.append(RetrievalResult(result_id=_id("result", run_id, entry.fact.runtime_fact_id), rank=rank,
                document_id=entry.fact.document_id, document_version_id=entry.fact.document_version_id,
                runtime_fact_id=entry.fact.runtime_fact_id, chunk_id=entry.locator.chunk_id,
                runtime_document_locator=entry.locator, score=f"{score:.8f}", score_type="C_LEXICAL_BM25_RRF_V1",
                evidence_eligible=entry.eligible, exclusion_reason_codes=entry.exclusion))
        snapshot_id = self._entries[0].fact.index_snapshot_id if self._entries else _id("empty-index")
        return RetrievalRun(retrieval_run_id=run_id, case_id=case_id, query_sha256=sha256_text(query), as_of=as_of,
            metadata_snapshot_hash=metadata_snapshot_hash, lifecycle_policy_version="A-1.1.0",
            filters=RetrievalFilters(), top_k=top_k, index_snapshot_id=snapshot_id, index_version=INDEX_VERSION,
            retrieval_config_version="c-retrieval-1.0.0", environment_namespace=environment_namespace,
            results=tuple(results))

    def answer(self, query: str, **kwargs) -> CResponse:
        run = self.retrieve(query, **kwargs)
        eligible = [r for r in run.results if r.evidence_eligible]
        if not eligible:
            reason = "NO_ACTIVE_SOURCE_VERSION" if run.results else "INSUFFICIENT_EVIDENCE"
            return CResponse("REFUSED", (), (), run, reason)
        result = eligible[0]
        entry = next(e for e in self._entries if e.fact.runtime_fact_id == result.runtime_fact_id)
        claim_id = _id("claim", run.case_id, entry.fact.runtime_fact_id)
        citation = TechnicalCitation(technical_citation_id=_id("citation", claim_id), claim_id=claim_id,
            case_id=run.case_id, runtime_fact_id=entry.fact.runtime_fact_id, fact_kind=entry.fact.fact_kind,
            retrieval_run_id=run.retrieval_run_id, result_id=result.result_id, document_id=result.document_id,
            document_version_id=result.document_version_id, chunk_id=result.chunk_id,
            runtime_document_locator=result.runtime_document_locator, retrieval_result=result, runtime_fact=entry.fact,
            relation=CitationRelation.SUPPORTS, support_level=SupportLevel.DIRECT,
            verification_status=VerificationStatus.EXACT_MATCH, evidence_eligible=True)
        return CResponse("ANSWERED", (entry.fact.statement,), (citation,), run)
