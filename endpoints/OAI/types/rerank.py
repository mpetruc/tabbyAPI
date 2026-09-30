"""Types for the rerank endpoint. Follows the Cohere rerank protocol
(the same protocol Infinity's own /rerank route implements)."""

import time
from typing import List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from endpoints.OAI.types.embedding import UsageInfo


class RerankRequest(BaseModel):
    query: str = Field(..., description="Search query to rerank documents against.")
    documents: List[str] = Field(
        ..., description="List of documents to rerank against the query."
    )
    return_documents: bool = Field(
        False, description="Include the document text in the response."
    )
    raw_scores: bool = Field(
        False,
        description="Return raw scores instead of sigmoid-normalized relevance scores.",
    )
    model: Optional[str] = Field(
        None,
        description="Name of the reranking model to use. "
        "If not provided, the default model will be used.",
    )
    top_n: Optional[int] = Field(
        None, ge=1, description="Number of top results to return. Returns all when omitted."
    )


class RerankObject(BaseModel):
    relevance_score: float = Field(..., description="Relevance score of the document.")
    index: int = Field(..., description="Index of the document in the original input list.")
    document: Optional[str] = Field(
        None, description="Document text, included when return_documents is true."
    )


class RerankResponse(BaseModel):
    object: str = Field("rerank", description="Type of the object.")
    results: List[RerankObject] = Field(..., description="Reranked documents, best first.")
    model: str = Field(..., description="Name of the model used.")
    usage: UsageInfo = Field(..., description="Information about token usage.")
    id: str = Field(default_factory=lambda: f"rerank-{uuid4()}", description="Response ID.")
    created: int = Field(default_factory=lambda: int(time.time()), description="Creation time.")
