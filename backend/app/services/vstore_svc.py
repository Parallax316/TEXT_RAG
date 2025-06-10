import uuid
from typing import List, Dict, Any, Optional, Tuple
import logging
from bson import ObjectId
from datetime import datetime

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document as LangchainDocument
from backend.app.core.mongodb import create_collection as mongo_create_collection, drop_collection as mongo_drop_collection, list_collections as mongo_list_collections, insert_embedding, find_embeddings
from backend.app.models.schemas import DocumentStatus, DocumentStatusResponse

from backend.app.core.config import settings
from backend.app.core.exceptions import DocumentProcessingError, EmbeddingError, RetrievalError, LLMError

# Set up logging
logger = logging.getLogger(__name__)

class VectorStoreService:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(VectorStoreService, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return

        logger.info("Initializing VectorStoreService (Latest LangChain)...")

        try:
            logger.info(f"Loading HuggingFaceEmbeddings with model: {settings.EMBEDDING_MODEL_NAME}")
            self.embedding_function = HuggingFaceEmbeddings(
                model_name=settings.EMBEDDING_MODEL_NAME,
                model_kwargs={'device': 'cpu'}
            )
            logger.info("HuggingFaceEmbeddings loaded successfully.")
        except Exception as e:
            raise EmbeddingError(f"Error initializing HuggingFaceEmbeddings: {e}")

        # self.chroma_instance = self._langchain_chroma_instance  # Removed: not needed for MongoDB

        self._initialized = True
        logger.info("VectorStoreService (Latest LangChain) initialization complete.")

    def create_collection(self, name: str) -> bool:
        logger.info(f"Creating MongoDB collection: {name}")
        try:
            mongo_create_collection(name)
            logger.info(f"Successfully created collection: {name}")
            return True
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise ValueError(f"Failed to create collection '{name}': {e}")

    def list_collections(self) -> List[Tuple[str, int]]:
        logger.info("Listing MongoDB collections")
        collection_names = mongo_list_collections()
        # Exclude system collections if needed
        filtered = [name for name in collection_names if not name.startswith('system.') and name != 'system_config']
        from backend.app.core.mongodb import db
        result = []
        for name in filtered:
            count = db[name].count_documents({})
            result.append((name, count))
        return result

    def delete_collection(self, name: str) -> bool:
        logger.info(f"Deleting MongoDB collection: {name}")
        try:
            mongo_drop_collection(name)
            logger.info(f"Successfully deleted collection: {name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            raise ValueError(f"Failed to delete collection '{name}': {e}")

    def add_documents(self, chunks: List[str], metadatas: List[Dict[str, Any]], doc_ids: Optional[List[str]] = None, collection_name: Optional[str] = None) -> bool:
        if not chunks:
            logger.warning("No chunks provided to add.")
            return False
        if len(chunks) != len(metadatas):
            logger.error("The number of chunks and metadatas must be the same.")
            return False
        for i, chunk_text in enumerate(chunks):
            doc = {
                "collection_name": collection_name,
                "type": "text",
                "embedding": metadatas[i].get("embedding"),
                "data": {"chunk_text": chunk_text},
                "metadata": metadatas[i]  # Store the metadata directly, not nested
            }
            insert_embedding(doc)
        logger.info(f"Added {len(chunks)} documents to collection '{collection_name}' in MongoDB.")
        return True

    def query_documents_with_scores(self, query_text: str, n_results: int = 5, filter_dict: Optional[Dict[str, Any]] = None, collection_name: Optional[str] = None) -> Optional[List[Tuple[LangchainDocument, float]]]:
        logger.info(f"Querying MongoDB collection '{collection_name}' for text documents.")
        query = {"collection_name": collection_name, "type": "text"}
        docs = find_embeddings(query)
        # Convert MongoDB docs to LangChain Document objects
        from langchain_core.documents import Document as LangchainDocument
        langchain_docs = []
        for doc in docs[:n_results]:
            # Extract content from the correct location
            data = doc.get('data', {})
            content = data.get('chunk_text', '') if isinstance(data, dict) else ''
            if not content:
                content = doc.get('content', doc.get('text', ''))
            
            # Extract metadata from both doc level and nested metadata
            doc_metadata = doc.get('metadata', {})
            metadata = {
                'source_doc_id': doc_metadata.get('source_doc_id', doc.get('source_doc_id', 'N/A')),
                'file_name': doc_metadata.get('file_name', doc.get('file_name', 'N/A')),
                'page_number': doc_metadata.get('page_number', doc.get('page_number')),
                'paragraph_number_in_page': doc_metadata.get('paragraph_number_in_page', doc.get('paragraph_number_in_page')),
                'chunk_index': doc_metadata.get('chunk_index', doc.get('chunk_index'))
            }
            
            if content:  # Only add documents with content
                langchain_docs.append(LangchainDocument(page_content=content, metadata=metadata))
        
        # Return LangChain Document objects with dummy score 0.0 (add embedding similarity if needed)
        results = [(doc, 0.0) for doc in langchain_docs]
        return results

    def _serialize_doc(self, doc):
        doc = dict(doc)
        if '_id' in doc and isinstance(doc['_id'], ObjectId):
            doc['_id'] = str(doc['_id'])
        return doc

    def get_collection(self, name: str = "default"):
        from backend.app.core.mongodb import db
        return db[name]
    
    # Document Status Management Methods
    def store_document_metadata(self, doc_id: str, filename: str, collection_name: str = "default") -> bool:
        """Store document metadata without processing"""
        try:
            from backend.app.core.mongodb import db
            doc_metadata = {
                "_id": doc_id,
                "filename": filename,
                "collection_name": collection_name,
                "status": DocumentStatus.UPLOADED,
                "upload_time": datetime.utcnow(),
                "processing_time": None,
                "error_message": None,
                "chunk_count": 0
            }
            
            result = db["document_metadata"].insert_one(doc_metadata)
            logger.info(f"Stored metadata for document {doc_id} in collection {collection_name}")
            return bool(result.inserted_id)
            
        except Exception as e:
            logger.error(f"Error storing document metadata for {doc_id}: {str(e)}")
            return False
    
    def get_document_status(self, doc_id: str) -> Optional[DocumentStatusResponse]:
        """Get document processing status"""
        try:
            from backend.app.core.mongodb import db
            doc = db["document_metadata"].find_one({"_id": doc_id})
            
            if not doc:
                return None
                
            return DocumentStatusResponse(
                doc_id=doc["_id"],
                filename=doc["filename"],
                status=doc["status"],
                upload_time=doc.get("upload_time"),
                processing_time=doc.get("processing_time"),
                error_message=doc.get("error_message"),
                chunk_count=doc.get("chunk_count", 0),
                collection_name=doc["collection_name"]
            )
            
        except Exception as e:
            logger.error(f"Error getting document status for {doc_id}: {str(e)}")
            return None
    
    def update_document_status(self, doc_id: str, status: DocumentStatus, 
                             error_message: str = None, chunk_count: int = None) -> bool:
        """Update document processing status"""
        try:
            from backend.app.core.mongodb import db
            update_data = {"status": status}
            
            if status == DocumentStatus.PROCESSING:
                update_data["processing_time"] = datetime.utcnow()
            elif status in [DocumentStatus.PROCESSED, DocumentStatus.ERROR]:
                if error_message:
                    update_data["error_message"] = error_message
                if chunk_count is not None:
                    update_data["chunk_count"] = chunk_count
            
            result = db["document_metadata"].update_one(
                {"_id": doc_id},
                {"$set": update_data}
            )
            
            logger.info(f"Updated status for document {doc_id} to {status}")
            return result.modified_count > 0
            
        except Exception as e:
            logger.error(f"Error updating document status for {doc_id}: {str(e)}")
            return False
    
    def get_unprocessed_documents(self, collection_name: str = "default") -> List[DocumentStatusResponse]:
        """Get all unprocessed documents in a collection"""
        try:
            from backend.app.core.mongodb import db
            docs = db["document_metadata"].find({
                "collection_name": collection_name,
                "status": {"$in": [DocumentStatus.UPLOADED, DocumentStatus.ERROR]}
            })
            
            result = []
            for doc in docs:
                result.append(DocumentStatusResponse(
                    doc_id=doc["_id"],
                    filename=doc["filename"],
                    status=doc["status"],
                    upload_time=doc.get("upload_time"),
                    processing_time=doc.get("processing_time"),
                    error_message=doc.get("error_message"),
                    chunk_count=doc.get("chunk_count", 0),
                    collection_name=doc["collection_name"]
                ))
            
            logger.info(f"Found {len(result)} unprocessed documents in collection {collection_name}")
            return result
            
        except Exception as e:
            logger.error(f"Error getting unprocessed documents: {str(e)}")
            return []
    
    def get_all_document_statuses(self, collection_name: str = "default") -> List[DocumentStatusResponse]:
        """Get status of all documents in a collection"""
        try:
            from backend.app.core.mongodb import db
            docs = db["document_metadata"].find({"collection_name": collection_name})
            
            result = []
            for doc in docs:
                result.append(DocumentStatusResponse(
                    doc_id=doc["_id"],
                    filename=doc["filename"],
                    status=doc["status"],
                    upload_time=doc.get("upload_time"),
                    processing_time=doc.get("processing_time"),
                    error_message=doc.get("error_message"),
                    chunk_count=doc.get("chunk_count", 0),
                    collection_name=doc["collection_name"]
                ))
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting all document statuses: {str(e)}")
            return []
