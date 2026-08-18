"""Gold/runtime document locators and render receipts."""

from __future__ import annotations

import unicodedata
from typing import Annotated, Literal, Union
from uuid import UUID

from pydantic import Field, model_validator

from .base import ContractModel, NonEmptyString, Sha256, UtcDatetime, sha256_text
from .enums import (
    DocumentFormat,
    LocatorRole,
    LocatorStatus,
    MappingStatus,
    SourceRenderMappingStatus,
)


class _QuoteModel(ContractModel):
    quote: NonEmptyString
    quote_sha256: Sha256

    @model_validator(mode="after")
    def verify_quote_hash(self) -> "_QuoteModel":
        if unicodedata.normalize("NFC", self.quote) != self.quote:
            raise ValueError("document quote must use Unicode NFC")
        if sha256_text(self.quote) != self.quote_sha256:
            raise ValueError("quote_sha256 does not match quote")
        return self


class _GoldDocumentBase(ContractModel):
    gold_locator_id: UUID
    locator_role: Literal[LocatorRole.GOLD] = LocatorRole.GOLD
    locator_status: LocatorStatus
    document_id: UUID
    document_version_id: UUID
    canonical_text_sha256: Sha256 | None = None
    quote: NonEmptyString | None = None
    quote_sha256: Sha256 | None = None

    def validate_gold_materialization(
        self, *, start: int | None, end: int | None, identity_values: tuple[object | None, ...]
    ) -> None:
        materialized_values = (
            self.canonical_text_sha256,
            self.quote,
            self.quote_sha256,
            start,
            end,
            *identity_values,
        )
        if self.locator_status == LocatorStatus.PLANNED:
            if any(value is not None for value in materialized_values):
                raise ValueError("PLANNED gold locator cannot claim completed coordinates")
            return
        if any(value is None for value in materialized_values):
            raise ValueError("materialized gold locator requires complete format coordinates")
        assert self.quote is not None and self.quote_sha256 is not None
        assert start is not None and end is not None
        if start > end or end - start != len(self.quote):
            raise ValueError("gold locator range must exactly cover the quote")
        if sha256_text(self.quote) != self.quote_sha256:
            raise ValueError("quote_sha256 does not match quote")


class GoldPdfLocator(_GoldDocumentBase):
    document_format: Literal[DocumentFormat.PDF] = DocumentFormat.PDF
    page_index: int | None = Field(default=None, ge=0)
    page_canonical_text_sha256: Sha256 | None = None
    page_char_start: int | None = Field(default=None, ge=0)
    page_char_end: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def bounds(self) -> "GoldPdfLocator":
        self.validate_gold_materialization(
            start=self.page_char_start,
            end=self.page_char_end,
            identity_values=(self.page_index, self.page_canonical_text_sha256),
        )
        return self


class GoldDocxLocator(_GoldDocumentBase):
    document_format: Literal[DocumentFormat.DOCX] = DocumentFormat.DOCX
    section_path: tuple[NonEmptyString, ...] | None = None
    paragraph_index: int | None = Field(default=None, ge=0)
    paragraph_text_sha256: Sha256 | None = None
    paragraph_char_start: int | None = Field(default=None, ge=0)
    paragraph_char_end: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def bounds(self) -> "GoldDocxLocator":
        self.validate_gold_materialization(
            start=self.paragraph_char_start,
            end=self.paragraph_char_end,
            identity_values=(self.section_path, self.paragraph_index, self.paragraph_text_sha256),
        )
        return self


class GoldMarkdownLocator(_GoldDocumentBase):
    document_format: Literal[DocumentFormat.MARKDOWN] = DocumentFormat.MARKDOWN
    section_path: tuple[NonEmptyString, ...] | None = None
    block_index: int | None = Field(default=None, ge=0)
    block_kind: NonEmptyString | None = None
    block_text_sha256: Sha256 | None = None
    block_char_start: int | None = Field(default=None, ge=0)
    block_char_end: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def bounds(self) -> "GoldMarkdownLocator":
        self.validate_gold_materialization(
            start=self.block_char_start,
            end=self.block_char_end,
            identity_values=(
                self.section_path,
                self.block_index,
                self.block_kind,
                self.block_text_sha256,
            ),
        )
        return self


class GoldTextLocator(_GoldDocumentBase):
    document_format: Literal[DocumentFormat.TEXT] = DocumentFormat.TEXT
    line_index: int | None = Field(default=None, ge=0)
    line_text_sha256: Sha256 | None = None
    line_char_start: int | None = Field(default=None, ge=0)
    line_char_end: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def bounds(self) -> "GoldTextLocator":
        self.validate_gold_materialization(
            start=self.line_char_start,
            end=self.line_char_end,
            identity_values=(self.line_index, self.line_text_sha256),
        )
        return self


GoldDocumentLocator = Annotated[
    Union[GoldPdfLocator, GoldDocxLocator, GoldMarkdownLocator, GoldTextLocator],
    Field(discriminator="document_format"),
]


class _RuntimeDocumentBase(_QuoteModel):
    runtime_locator_id: UUID
    locator_role: Literal[LocatorRole.RUNTIME] = LocatorRole.RUNTIME
    document_id: UUID
    document_version_id: UUID
    canonical_text_sha256: Sha256
    chunk_id: UUID
    chunk_sha256: Sha256
    chunk_char_start: int = Field(ge=0)
    chunk_char_end: int = Field(ge=0)
    parser_id: NonEmptyString
    parser_version: NonEmptyString
    canonicalizer_version: NonEmptyString

    @model_validator(mode="after")
    def chunk_bounds(self) -> "_RuntimeDocumentBase":
        if (
            self.chunk_char_start > self.chunk_char_end
            or self.chunk_char_end - self.chunk_char_start != len(self.quote)
        ):
            raise ValueError("chunk character range must exactly cover the quote")
        return self


class RuntimePdfLocator(_RuntimeDocumentBase):
    document_format: Literal[DocumentFormat.PDF] = DocumentFormat.PDF
    page_index: int = Field(ge=0)
    page_canonical_text_sha256: Sha256
    page_char_start: int = Field(ge=0)
    page_char_end: int = Field(ge=0)

    @model_validator(mode="after")
    def page_bounds(self) -> "RuntimePdfLocator":
        if self.page_char_end - self.page_char_start != len(self.quote):
            raise ValueError("PDF page range must exactly cover the quote")
        return self


class RuntimeDocxLocator(_RuntimeDocumentBase):
    document_format: Literal[DocumentFormat.DOCX] = DocumentFormat.DOCX
    section_path: tuple[NonEmptyString, ...]
    paragraph_index: int = Field(ge=0)
    paragraph_text_sha256: Sha256
    paragraph_char_start: int = Field(ge=0)
    paragraph_char_end: int = Field(ge=0)

    @model_validator(mode="after")
    def paragraph_bounds(self) -> "RuntimeDocxLocator":
        if self.paragraph_char_end - self.paragraph_char_start != len(self.quote):
            raise ValueError("DOCX paragraph range must exactly cover the quote")
        return self


class RuntimeMarkdownLocator(_RuntimeDocumentBase):
    document_format: Literal[DocumentFormat.MARKDOWN] = DocumentFormat.MARKDOWN
    section_path: tuple[NonEmptyString, ...]
    block_index: int = Field(ge=0)
    block_kind: NonEmptyString
    block_text_sha256: Sha256
    block_char_start: int = Field(ge=0)
    block_char_end: int = Field(ge=0)

    @model_validator(mode="after")
    def block_bounds(self) -> "RuntimeMarkdownLocator":
        if self.block_char_end - self.block_char_start != len(self.quote):
            raise ValueError("Markdown block range must exactly cover the quote")
        return self


class RuntimeTextLocator(_RuntimeDocumentBase):
    document_format: Literal[DocumentFormat.TEXT] = DocumentFormat.TEXT
    line_index: int = Field(ge=0)
    line_text_sha256: Sha256
    line_char_start: int = Field(ge=0)
    line_char_end: int = Field(ge=0)

    @model_validator(mode="after")
    def line_bounds(self) -> "RuntimeTextLocator":
        if self.line_char_end - self.line_char_start != len(self.quote):
            raise ValueError("text line range must exactly cover the quote")
        return self


RuntimeDocumentLocator = Annotated[
    Union[RuntimePdfLocator, RuntimeDocxLocator, RuntimeMarkdownLocator, RuntimeTextLocator],
    Field(discriminator="document_format"),
]


class LocatorMapping(ContractModel):
    mapping_id: UUID
    mapping_version: NonEmptyString
    gold_locator_id: UUID
    document_version_id: UUID
    canonical_text_sha256: Sha256
    runtime_locators: tuple[RuntimeDocumentLocator, ...] = ()
    mapping_status: Literal[MappingStatus.EXACT, MappingStatus.SPLIT, MappingStatus.FAILED]
    parser_version: NonEmptyString
    chunking_version: NonEmptyString
    verification_method: NonEmptyString

    @model_validator(mode="after")
    def mapping_cardinality(self) -> "LocatorMapping":
        if self.mapping_status == MappingStatus.EXACT and len(self.runtime_locators) != 1:
            raise ValueError("EXACT mapping requires one runtime locator")
        if self.mapping_status == MappingStatus.SPLIT and len(self.runtime_locators) < 2:
            raise ValueError("SPLIT mapping requires multiple runtime locators")
        if self.mapping_status == MappingStatus.FAILED and self.runtime_locators:
            raise ValueError("FAILED mapping cannot contain runtime locators")
        if any(
            locator.document_version_id != self.document_version_id
            or locator.canonical_text_sha256 != self.canonical_text_sha256
            for locator in self.runtime_locators
        ):
            raise ValueError("runtime locator does not match mapping snapshot")
        return self


class RenderedPageSegment(ContractModel):
    page_index: int = Field(ge=0)
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)

    @model_validator(mode="after")
    def segment_bounds(self) -> "RenderedPageSegment":
        if self.char_start > self.char_end:
            raise ValueError("rendered page segment range is invalid")
        return self


class SourceToRenderMapping(ContractModel):
    source_locator_id: UUID
    rendered_page_segments: tuple[RenderedPageSegment, ...] = ()
    status: SourceRenderMappingStatus
    reason_code: str | None = None

    @model_validator(mode="after")
    def mapping_shape(self) -> "SourceToRenderMapping":
        if self.status is SourceRenderMappingStatus.MAPPED and not self.rendered_page_segments:
            raise ValueError("mapped source requires page segments")
        if self.status is not SourceRenderMappingStatus.MAPPED and self.rendered_page_segments:
            raise ValueError("failed source cannot carry page segments")
        if self.status is not SourceRenderMappingStatus.MAPPED and not self.reason_code:
            raise ValueError("failed source mapping requires reason_code")
        if self.status is SourceRenderMappingStatus.MAPPED and self.reason_code is not None:
            raise ValueError("mapped source cannot carry failure reason")
        return self


class RenderReceipt(ContractModel):
    render_receipt_id: UUID
    document_version_id: UUID
    source_sha256: Sha256
    canonical_text_sha256: Sha256
    parser_id: NonEmptyString
    parser_version: NonEmptyString
    canonicalizer_version: NonEmptyString
    renderer_id: NonEmptyString
    renderer_version: NonEmptyString
    render_config_hash: Sha256
    rendered_asset_sha256: Sha256
    page_count: int = Field(ge=1)
    page_text_hashes: tuple[Sha256, ...] = Field(min_length=1)
    source_to_render_mappings: tuple[SourceToRenderMapping, ...] = Field(min_length=1)
    mapping_status: Literal[MappingStatus.COMPLETE, MappingStatus.PARTIAL, MappingStatus.FAILED]
    created_at: UtcDatetime

    @model_validator(mode="after")
    def mapping_consistency(self) -> "RenderReceipt":
        mapped = sum(
            item.status is SourceRenderMappingStatus.MAPPED
            for item in self.source_to_render_mappings
        )
        total = len(self.source_to_render_mappings)
        if len(self.page_text_hashes) != self.page_count:
            raise ValueError("page_text_hashes must match page_count")
        if self.mapping_status == MappingStatus.COMPLETE and mapped != total:
            raise ValueError("COMPLETE receipt requires every mapping")
        if self.mapping_status == MappingStatus.FAILED and mapped:
            raise ValueError("FAILED receipt cannot contain mapped sources")
        if self.mapping_status == MappingStatus.PARTIAL and not (0 < mapped < total):
            raise ValueError("PARTIAL receipt requires mixed mapping results")
        return self
