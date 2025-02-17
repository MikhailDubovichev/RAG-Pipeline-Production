import logging
from pathlib import Path
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)
router = APIRouter()

class RAGService:
    def __init__(self, search_service, llm_service):
        self.search_service = search_service
        self.llm_service = llm_service

    async def process_query(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        try:
            # Perform search
            search_results = self.search_service.search(
                query=query,
                top_k=top_k
            )

            # Prepare context from search results
            context = "\n\n".join([text for text, score, metadata in search_results])

            # Generate LLM response
            response, _ = self.llm_service.generate_response(query, context)

            # Format response with references
            formatted_response = self.llm_service.format_response_with_references(
                response, search_results
            )

            return {
                "response": formatted_response,
                "status": "success"
            }

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Error processing query: {str(e)}"
            )

# Initialize services (to be done in main.py)
rag_service = None

def initialize_routes(search_svc, llm_svc):
    """Initialize the RAG service with required dependencies."""
    global rag_service
    rag_service = RAGService(search_svc, llm_svc)

@router.post("/query")
async def process_query(query: str):
    """Process a query and return the response."""
    if not rag_service:
        raise HTTPException(
            status_code=500,
            detail="RAG service not initialized"
        )
    return await rag_service.process_query(query)

@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"} 