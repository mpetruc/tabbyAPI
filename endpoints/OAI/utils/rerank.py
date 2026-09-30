"""Handler for the rerank endpoint."""

from fastapi import HTTPException, Request

from common import model
from common.logger import xlogger
from common.networking import handle_request_error, request_tag
from common.optional_dependencies import dependencies
from endpoints.OAI.types.embedding import UsageInfo
from endpoints.OAI.types.rerank import RerankObject, RerankRequest, RerankResponse

# Only available when the embeddings stack is installed; the router's
# check_embeddings_container dependency already rejects requests otherwise.
if dependencies.extras:
    from infinity_emb.primitives import ModelNotDeployedError
else:  # pragma: no cover
    ModelNotDeployedError = Exception


async def get_rerank(data: RerankRequest, request: Request) -> RerankResponse:
    model_path = model.embeddings_container.model_dir

    xlogger.debug(f"Received rerank request {request.state.id}")

    try:
        rerank_data = await model.embeddings_container.rerank(
            query=data.query,
            documents=data.documents,
            raw_scores=data.raw_scores,
            top_n=data.top_n,
        )
    except ModelNotDeployedError as exc:
        # The loaded model is likely an embedding model, not a reranker
        error_message = handle_request_error(str(exc), exc_info=False).error.message
        raise HTTPException(400, error_message) from exc

    results = rerank_data.get("results")
    usage = rerank_data.get("usage")

    rerank_objects = [
        RerankObject(
            relevance_score=entry.relevance_score,
            index=entry.index,
            document=entry.document if data.return_documents else None,
        )
        for entry in results
    ]

    response = RerankResponse(
        results=rerank_objects,
        model=model_path.name,
        usage=UsageInfo(prompt_tokens=usage, total_tokens=usage),
    )

    xlogger.info(
        f"{request_tag(request)} rerank: {len(data.documents)} documents, {usage} tokens",
        {"request_id": request.state.id, "documents": len(data.documents), "tokens": usage},
    )

    return response
