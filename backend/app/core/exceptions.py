# backend/app/core/exceptions.py

class DocumentProcessingError(Exception):
    """Raised when document processing fails."""
    pass

class EmbeddingError(Exception):
    """Raised when embedding generation or storage fails."""
    pass

class RetrievalError(Exception):
    """Raised when retrieval from the vector store fails."""
    pass

class LLMError(Exception):
    """Raised when LLM call or response parsing fails."""
    pass
