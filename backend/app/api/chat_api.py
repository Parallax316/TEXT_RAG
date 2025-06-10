# backend/app/api/chat_api.py

from fastapi import APIRouter, HTTPException, Depends, Body

# Import schemas
from backend.app.models.schemas import QueryRequest, RAGResponse

# Import services and settings
from backend.app.core.config import settings
from backend.app.services.rag_svc import RAGService
from backend.app.services.vstore_svc import VectorStoreService # For RAGService dependency
from backend.app.services.cluster_analysis_svc import ClusterAnalysisService # Import ClusterAnalysisService

# --- Dependency to get service instances ---
# Re-using the singleton pattern established in docs_api.py for consistency
# We need VectorStoreService for RAGService. DocParserService isn't directly
# used by this endpoint but its initialization might be linked if VectorStoreService
# was designed to be initialized alongside it. For now, RAGService just needs VectorStoreService.

_vector_store_service_instance = None
_rag_service_instance = None
_cluster_analysis_service_instance = None # Add instance for ClusterAnalysisService

def get_vector_store_service():
    global _vector_store_service_instance
    if _vector_store_service_instance is None:
        _vector_store_service_instance = VectorStoreService()
    return _vector_store_service_instance

def get_rag_service(vstore_svc: VectorStoreService = Depends(get_vector_store_service)):
    global _rag_service_instance
    if _rag_service_instance is None:
        _rag_service_instance = RAGService(vector_store_service=vstore_svc)
    return _rag_service_instance

def get_cluster_analysis_service(): # Add dependency for ClusterAnalysisService
    global _cluster_analysis_service_instance
    if _cluster_analysis_service_instance is None:
        _cluster_analysis_service_instance = ClusterAnalysisService()
    return _cluster_analysis_service_instance
# --- End of Dependency Setup ---

router = APIRouter(prefix="/chat", tags=["Chat"])

@router.post("/query", response_model=RAGResponse)
async def handle_query(
    query_request: QueryRequest, # Request body will be parsed into this Pydantic model
    rag_svc: RAGService = Depends(get_rag_service)
):
    """
    Receives a user query, processes it using the RAG service,
    and returns a synthesized answer, identified themes, and citations.
    """
    if not query_request.query or not query_request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        print(f"Received query in API: '{query_request.query}'")
        # The RAGService's get_answer_and_themes method will handle
        # retrieving context, calling the LLM, and formatting the response.
        # You can pass other parameters like n_retrieved_docs if you add them to QueryRequest
        # and handle them in RAGService.
        # For now, using default n_retrieved_docs in RAGService.
        print(f"Collection name from request: '{query_request.collection}'")
        
        result = rag_svc.get_answer_and_themes(
            query=query_request.query,
            collection_name=query_request.collection
        )

        if result is None:
            # This might happen if RAGService has an internal issue before returning a structured error.
            raise HTTPException(status_code=500, detail="Failed to get a response from the RAG service.")

        # The result from RAGService should already match the RAGResponse schema.
        # Pydantic will validate it on return.
        return result

    except HTTPException as http_exc:
        # Re-raise HTTPException if it's already one (e.g., from input validation)
        raise http_exc
    except Exception as e:
        # Catch any other unexpected errors during processing
        print(f"Error processing query in chat_api: {e}")
        # import traceback
        # traceback.print_exc() # For more detailed server-side logs during development
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

# New endpoint for analyzing a collection
@router.post("/collections/{collection_name}/analyze")
async def analyze_collection(
    collection_name: str,
    cluster_analysis_svc: ClusterAnalysisService = Depends(get_cluster_analysis_service)
):
    """
    Analyzes a collection using clustering and summarization.
    """
    try:
        result = cluster_analysis_svc.analyze_collection(collection_name)
        return result
    except Exception as e:
        print(f"Error analyzing collection in chat_api: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

# New endpoint for clearing analysis cache
@router.post("/collections/{collection_name}/clear-cache")
async def clear_analysis_cache(
    collection_name: str,
    cluster_analysis_svc: ClusterAnalysisService = Depends(get_cluster_analysis_service)
):
    """
    Clears the analysis cache for a collection.
    This should be called when documents are added or removed from a collection.
    """
    try:
        cluster_analysis_svc._clear_cache(collection_name)
        return {"message": f"Cache cleared for collection: {collection_name}"}
    except Exception as e:
        print(f"Error clearing cache in chat_api: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

# Example for testing (not part of the API itself)
if __name__ == "__main__":
    print("chat_api.py can be tested by running the main FastAPI application and sending requests to its /query endpoint.")
    # Example of how you might test this with `requests` if the server were running:
    # import requests
    # query_payload = {"query": "What are the main themes in the SEBI Act documents?"}
    # try:
    #     response = requests.post("http://localhost:8000/api/v1/chat/query", json=query_payload)
    #     response.raise_for_status()
    #     print("Test query response:")
    #     print(response.json())
    # except requests.exceptions.RequestException as e:
    #     print(f"Test query failed: {e}")
    # except Exception as e_gen:
    #     print(f"An error occurred during test: {e_gen}")

