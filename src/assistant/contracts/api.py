"""Request/response models for the nine frozen API endpoints."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import Field

from .base import API_VERSION, CONTRACT_VERSION, NonEmptyString, Sha256, UtcDatetime, VersionedContract
from .cases import CaseRecord
from .customers import CustomerMessage
from .documents import DocumentMetadata, SourceDocumentMetadata
from .drafts import ReplyDraft, ReviewDecision
from .enums import ReviewDecisionType
from .qa import QAResponse
from .requirements import CustomerRequirement
from .retrieval import RetrievalFilters, RetrievalRun


class ResponseMeta(VersionedContract):
    api_version: Literal["v1"] = API_VERSION
    trace_id: UUID


class IdempotentCommand(VersionedContract):
    idempotency_key: NonEmptyString


class CreateDocumentRequest(IdempotentCommand):
    source_metadata: SourceDocumentMetadata


class CreateDocumentResponse(ResponseMeta):
    document_metadata: DocumentMetadata


class GetDocumentVersionRequest(VersionedContract):
    document_id: UUID
    document_version_id: UUID
    as_of: UtcDatetime | None = None


class GetDocumentVersionResponse(ResponseMeta):
    document_metadata: DocumentMetadata


class CreateCaseRequest(IdempotentCommand):
    title: NonEmptyString
    customer_ref: NonEmptyString
    messages: tuple[CustomerMessage, ...] = Field(min_length=1)


class CreateCaseResponse(ResponseMeta):
    case_record: CaseRecord


class GetCaseRequest(VersionedContract):
    case_id: UUID


class GetCaseResponse(ResponseMeta):
    case_record: CaseRecord


class ExtractRequirementsRequest(IdempotentCommand):
    case_id: UUID
    message_ids: tuple[UUID, ...] = Field(min_length=1)
    input_snapshot_hash: Sha256


class ExtractRequirementsResponse(ResponseMeta):
    requirement: CustomerRequirement


class SearchRetrievalRequest(IdempotentCommand):
    case_id: UUID
    query: NonEmptyString
    as_of: UtcDatetime
    filters: RetrievalFilters
    top_k: int = Field(ge=1, le=50)


class SearchRetrievalResponse(ResponseMeta):
    retrieval_run: RetrievalRun


class GenerateAnswerRequest(IdempotentCommand):
    case_id: UUID
    question: NonEmptyString
    retrieval_run_id: UUID
    as_of: UtcDatetime


class GenerateAnswerResponse(ResponseMeta):
    qa_response: QAResponse


class GenerateReplyDraftRequest(IdempotentCommand):
    case_id: UUID
    requirement_id: UUID
    requirement_revision: int = Field(ge=1)
    retrieval_run_ids: tuple[UUID, ...]
    input_snapshot_hash: Sha256


class GenerateReplyDraftResponse(ResponseMeta):
    reply_draft: ReplyDraft


class ReviewReplyDraftRequest(IdempotentCommand):
    case_id: UUID
    draft_id: UUID
    expected_revision: int = Field(ge=1)
    draft_content_sha256: Sha256
    decision: ReviewDecisionType
    reviewer_id: UUID
    reviewer_role: NonEmptyString
    comment: str | None = None


class ReviewReplyDraftResponse(ResponseMeta):
    review_decision: ReviewDecision


class EndpointSpec(VersionedContract):
    operation_id: NonEmptyString
    method: Literal["GET", "POST"]
    path: NonEmptyString
    request_model: NonEmptyString
    response_model: NonEmptyString


API_ENDPOINTS: tuple[EndpointSpec, ...] = (
    EndpointSpec(operation_id="create_document", method="POST", path="/api/v1/documents", request_model="CreateDocumentRequest", response_model="CreateDocumentResponse"),
    EndpointSpec(operation_id="get_document_version", method="GET", path="/api/v1/documents/{document_id}/versions/{document_version_id}", request_model="GetDocumentVersionRequest", response_model="GetDocumentVersionResponse"),
    EndpointSpec(operation_id="create_case", method="POST", path="/api/v1/cases", request_model="CreateCaseRequest", response_model="CreateCaseResponse"),
    EndpointSpec(operation_id="get_case", method="GET", path="/api/v1/cases/{case_id}", request_model="GetCaseRequest", response_model="GetCaseResponse"),
    EndpointSpec(operation_id="extract_requirements", method="POST", path="/api/v1/cases/{case_id}/requirements:extract", request_model="ExtractRequirementsRequest", response_model="ExtractRequirementsResponse"),
    EndpointSpec(operation_id="search_retrieval", method="POST", path="/api/v1/cases/{case_id}/retrieval:search", request_model="SearchRetrievalRequest", response_model="SearchRetrievalResponse"),
    EndpointSpec(operation_id="generate_answer", method="POST", path="/api/v1/cases/{case_id}/answers:generate", request_model="GenerateAnswerRequest", response_model="GenerateAnswerResponse"),
    EndpointSpec(operation_id="generate_reply_draft", method="POST", path="/api/v1/cases/{case_id}/reply-drafts:generate", request_model="GenerateReplyDraftRequest", response_model="GenerateReplyDraftResponse"),
    EndpointSpec(operation_id="review_reply_draft", method="POST", path="/api/v1/cases/{case_id}/reply-drafts/{draft_id}/reviews", request_model="ReviewReplyDraftRequest", response_model="ReviewReplyDraftResponse"),
)

assert len(API_ENDPOINTS) == 9
