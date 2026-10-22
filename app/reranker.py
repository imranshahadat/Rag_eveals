import logging
import os
from typing import Any, List, Optional

import requests
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from langsmith import traceable
from pydantic import model_validator

from app.config import settings

logger = logging.getLogger(__name__)


class JinaReranker(BaseDocumentCompressor):
    """Custom Document Compressor using the Jina AI Reranker API.

    Usage:
        reranker = JinaReranker(top_n=5)
        docs = reranker.compress_documents(docs, query)
    """

    # BaseDocumentCompressor is a Pydantic model -> all attributes must be
    # declared as fields; you cannot assign arbitrary attrs in __init__.
    api_key: Optional[str] = None
    api_url: str = "https://api.jina.ai/v1/rerank"
    model: str = "jina-reranker-v2-base-multilingual"
    top_n: int = 3
    request_timeout: float = 30.0

    @model_validator(mode="after")
    def _resolve_api_key(self) -> "JinaReranker":
        if not self.api_key:
            self.api_key = settings.JINA_API_KEY or os.getenv("JINA_API_KEY")
        if not self.api_key:
            raise ValueError(
                "JINA_API_KEY is missing. Set it in your .env file "
                "or pass JinaReranker(api_key='...')."
            )
        return self

    @traceable(name="Jina Reranker", run_type="tool", tags=["reranker", "jina"])
    def compress_documents(
        self,
        documents: List[Document],
        query: str,
        **kwargs: Any,
    ) -> List[Document]:
        if not documents:
            return []

        payload = {
            "model": self.model,
            "query": query,
            "documents": [doc.page_content for doc in documents],
            "top_n": self.top_n,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        try:
            response = requests.post(
                self.api_url,
                json=payload,
                headers=headers,
                timeout=self.request_timeout,
            )
            response.raise_for_status()
            results = response.json().get("results", [])
        except (requests.RequestException, ValueError) as e:
            logger.error("Jina Reranker API failed: %s", e)
            # Safe fallback: keep original order, truncated to top_n
            return list(documents)[: self.top_n]

        if not results:
            logger.warning("Jina Reranker returned no results; using original order.")
            return list(documents)[: self.top_n]

        # Jina sorts by relevance already, but sort defensively anyway.
        results = sorted(results, key=lambda r: r.get("relevance_score", 0.0), reverse=True)

        reranked: List[Document] = []
        for item in results:
            idx = item.get("index")
            if idx is None or idx >= len(documents):
                continue
            src = documents[idx]
            reranked.append(
                Document(
                    page_content=src.page_content,
                    metadata={
                        **src.metadata,
                        "reranker_score": round(item.get("relevance_score", 0.0), 4),
                    },
                )
            )
        return reranked