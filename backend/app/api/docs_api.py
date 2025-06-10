# backend/app/api/docs_api.py

import os
import shutil
from pathlib import Path
from typing import List, Union, Optional 

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, BackgroundTasks # Removed Form

# Import schemas
from backend.app.models.schemas import DocumentProcessResponse, DocumentStatus, DocumentStatusResponse, ProcessDocumentRequest, ProcessBatchRequest

# Import services and settings
from backend.app.core.config import settings
# REMOVED: from backend.app.services.doc_parser import DocParserService 
from backend.app.services.doc_parser_fast import DocParserFastService # ONLY Fast rule-based parser
from backend.app.services.vstore_svc import VectorStoreService 
from backend.app.core.utils import get_next_document_id

# --- Dependency to get service instances ---
_vector_store_service_instance = None
# REMOVED: _doc_parser_llm_instance = None
_doc_parser_fast_instance = None # This will be the only parser used

def get_vector_store_service():
    global _vector_store_service_instance
    if _vector_store_service_instance is None:
        _vector_store_service_instance = VectorStoreService()
    return _vector_store_service_instance

# REMOVED: get_doc_parser_llm_service

def get_doc_parser_fast_service(vstore_svc: VectorStoreService = Depends(get_vector_store_service)):
    global _doc_parser_fast_instance
    if _doc_parser_fast_instance is None:
        _doc_parser_fast_instance = DocParserFastService(vector_store_service=vstore_svc)
    return _doc_parser_fast_instance

router = APIRouter(prefix="/docs", tags=["Documents"])
Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

# REMOVED: AnyDocParserService type alias as only one parser is used

def process_document_background(
    file_path: str, 
    source_doc_id: str, 
    doc_parser_svc_instance: DocParserFastService, # MODIFIED: Specific to DocParserFastService
    serial_no: Optional[int] = None, 
    total_count: Optional[int] = None,
    collection_name: Optional[str] = None
):
    parser_name = doc_parser_svc_instance.__class__.__name__ # Will be DocParserFastService
    progress_log = f"Document {serial_no}/{total_count}" if serial_no and total_count else "Single document"
    
    print(f"Background task started for: {Path(file_path).name}, Source ID: {source_doc_id}, Parser: {parser_name}, ({progress_log})")
    try:
        success = doc_parser_svc_instance.process_document(
            file_path=file_path, 
            source_doc_id=source_doc_id,
            collection_name=collection_name
        )
        if success:
            print(f"Background processing completed successfully for {source_doc_id} ({Path(file_path).name}) using {parser_name}")
        else:
            print(f"Background processing (using {parser_name}) had issues or no chunks generated for {source_doc_id} ({Path(file_path).name})")
    except Exception as e:
        print(f"Error during background document processing for {source_doc_id} ({Path(file_path).name}) (using {parser_name}): {e}")

@router.post("/upload", response_model=DocumentProcessResponse, status_code=202)
async def upload_document(
    file: UploadFile = File(..., description="The document file to upload (PDF or image)."),
    collection: str = None,  # Add collection parameter
    vstore_svc: VectorStoreService = Depends(get_vector_store_service)
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file name provided.")

    # Validate collection if provided
    if collection:
        try:
            vstore_svc.get_collection(collection)
        except Exception as e:
            if "does not exists" in str(e).lower():
                raise HTTPException(status_code=404, detail=f"Collection '{collection}' not found")
            raise HTTPException(status_code=500, detail=f"Error validating collection: {str(e)}")

    source_doc_id = get_next_document_id()
    safe_original_filename = "".join(c if c.isalnum() or c in ['.', '_'] else '_' for c in file.filename)
    temp_file_path = Path(settings.UPLOAD_DIR) / f"{source_doc_id}_{safe_original_filename}"

    try:
        # Save file to disk
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"File '{file.filename}' saved to '{temp_file_path}'. Assigned Source ID: {source_doc_id}")
        
        # Store document metadata without processing
        collection_name = collection or "default"
        metadata_stored = vstore_svc.store_document_metadata(source_doc_id, file.filename, collection_name)
        
        if not metadata_stored:
            # Clean up file if metadata storage failed
            if temp_file_path.exists():
                os.remove(temp_file_path)
            raise HTTPException(status_code=500, detail="Failed to store document metadata")
            
    except Exception as e:
        if temp_file_path.exists():
            os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=f"Could not save uploaded file: {e}")
    finally:
        file.file.close()

    return DocumentProcessResponse(
        message="File uploaded successfully. Use /process endpoint to start processing.",
        file_name=file.filename,
        source_doc_id=source_doc_id,
        status="uploaded",
        processing_mode_used="fast_rule_based"
    )

@router.post("/upload-multiple", response_model=List[DocumentProcessResponse], status_code=202)
async def upload_multiple_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="A list of document files to upload."),
    collection: str = None,  # Main collection parameter
    collection_name: str = None,  # Backward compatibility
    fast_parser: DocParserFastService = Depends(get_doc_parser_fast_service),
    vstore_svc: VectorStoreService = Depends(get_vector_store_service)
):
    # Use 'collection' if provided, else fallback to 'collection_name'
    collection = collection or collection_name
    responses = []
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    # Validate collection if provided
    if collection:
        try:
            vstore_svc.get_collection(collection)
        except Exception as e:
            if "does not exists" in str(e).lower():
                raise HTTPException(status_code=404, detail=f"Collection '{collection}' not found")
            raise HTTPException(status_code=500, detail=f"Error validating collection: {str(e)}")

    # No need to choose parser, always use fast_parser
    chosen_parser = fast_parser

    total_files_in_batch = len(files)
    for idx, file_upload_item in enumerate(files, start=1):
        if not file_upload_item.filename:
            responses.append(DocumentProcessResponse(
                message="File skipped: No filename.",
                file_name="Unknown",
                status="failed_upload",
                processing_mode_used="fast_rule_based"
            ))
            continue

        source_doc_id = get_next_document_id()
        safe_original_filename = "".join(c if c.isalnum() or c in ['.', '_'] else '_' for c in file_upload_item.filename)
        temp_file_path = Path(settings.UPLOAD_DIR) / f"{source_doc_id}_{safe_original_filename}"

        try:
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(file_upload_item.file, buffer)
            print(f"File '{file_upload_item.filename}' saved to '{temp_file_path}' for fast rule-based processing. Assigned Source ID: {source_doc_id} ({idx}/{total_files_in_batch})")
            
            # Add collection to the background task
            background_tasks.add_task(
                process_document_background,
                str(temp_file_path),
                source_doc_id,
                chosen_parser,
                serial_no=idx,
                total_count=total_files_in_batch,
                collection_name=collection  # Pass collection name to background task
            )
            responses.append(DocumentProcessResponse(
                message="File uploaded successfully and queued for fast rule-based background processing.",
                file_name=file_upload_item.filename,
                source_doc_id=source_doc_id,
                status="queued_for_processing",
                processing_mode_used="fast_rule_based"
            ))
        except Exception as e:
            if temp_file_path.exists():
                os.remove(temp_file_path)
            responses.append(DocumentProcessResponse(
                message=f"Failed to save or queue file '{file_upload_item.filename}': {e}",
                file_name=file_upload_item.filename,
                source_doc_id=source_doc_id,
                status="failed_to_queue",
                processing_mode_used="fast_rule_based"
            ))
        finally:
            if file_upload_item and hasattr(file_upload_item, 'file') and file_upload_item.file and not file_upload_item.file.closed:
                 try:
                    file_upload_item.file.close()
                 except Exception as e_close:
                    print(f"Error closing file {file_upload_item.filename if file_upload_item else 'N/A'}: {e_close}")
            
    return responses

# --- Document Processing Endpoints ---

def _process_document_background(doc_id: str, filename: str, collection_name: str):
    """Background function to process a document"""
    try:
        # Get the vector store service
        vstore_svc = get_vector_store_service()
        
        # Get fast parser
        fast_parser = get_doc_parser_fast_service(vstore_svc)
        
        # Update status to processing
        vstore_svc.update_document_status(doc_id, DocumentStatus.PROCESSING)
        
        # Find the file path
        safe_filename = "".join(c if c.isalnum() or c in ['.', '_'] else '_' for c in filename)
        file_path = Path(settings.UPLOAD_DIR) / f"{doc_id}_{safe_filename}"
        
        if not file_path.exists():
            error_msg = f"File not found: {file_path}"
            vstore_svc.update_document_status(doc_id, DocumentStatus.ERROR, error_message=error_msg)
            return
        
        # Process the document
        success = fast_parser.process_document(
            file_path=str(file_path),
            source_doc_id=doc_id,
            collection_name=collection_name
        )
        
        if success:
            # Count chunks (approximate - you might want to implement a better method)
            chunk_count = 1  # This is a placeholder - implement actual chunk counting
            vstore_svc.update_document_status(doc_id, DocumentStatus.PROCESSED, chunk_count=chunk_count)
            print(f"Successfully processed document {doc_id}")
        else:
            vstore_svc.update_document_status(doc_id, DocumentStatus.ERROR, error_message="Processing failed")
            print(f"Failed to process document {doc_id}")
            
    except Exception as e:
        error_msg = f"Error processing document: {str(e)}"
        vstore_svc.update_document_status(doc_id, DocumentStatus.ERROR, error_message=error_msg)
        print(f"Error processing document {doc_id}: {error_msg}")

@router.post("/process/{doc_id}")
async def process_document(
    doc_id: str,
    collection_name: str = "default",
    background_tasks: BackgroundTasks = BackgroundTasks(),
    vstore_svc: VectorStoreService = Depends(get_vector_store_service)
):
    """Process a specific uploaded document (chunking + embedding)"""
    try:
        # Check if document exists and is unprocessed
        doc_status = vstore_svc.get_document_status(doc_id)
        if not doc_status:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
        
        if doc_status.status == DocumentStatus.PROCESSED:
            return {"message": f"Document {doc_id} already processed", "status": "already_processed"}
        
        if doc_status.status == DocumentStatus.PROCESSING:
            return {"message": f"Document {doc_id} is currently being processed", "status": "processing"}
        
        # Start background processing
        background_tasks.add_task(
            _process_document_background,
            doc_id,
            doc_status.filename,
            collection_name
        )
        
        print(f"Started processing document {doc_id} in collection {collection_name}")
        return {
            "message": f"Document {doc_id} processing started", 
            "doc_id": doc_id,
            "status": "processing_started"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error starting document processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error starting processing: {str(e)}")

@router.post("/process-batch")
async def process_batch_documents(
    request: ProcessBatchRequest,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    vstore_svc: VectorStoreService = Depends(get_vector_store_service)
):
    """Process multiple documents in batch"""
    try:
        collection_name = request.collection_name or "default"
        
        if request.doc_ids:
            # Process specific documents
            docs_to_process = []
            for doc_id in request.doc_ids:
                doc_status = vstore_svc.get_document_status(doc_id)
                if doc_status and doc_status.status in [DocumentStatus.UPLOADED, DocumentStatus.ERROR]:
                    docs_to_process.append(doc_status)
        else:
            # Process all unprocessed documents in collection
            docs_to_process = vstore_svc.get_unprocessed_documents(collection_name)
        
        if not docs_to_process:
            return {"message": "No documents to process", "processed_count": 0}
        
        # Start background processing for each document
        for doc in docs_to_process:
            background_tasks.add_task(
                _process_document_background,
                doc.doc_id,
                doc.filename,
                collection_name
            )
        
        print(f"Started batch processing {len(docs_to_process)} documents in collection {collection_name}")
        return {
            "message": f"Batch processing started for {len(docs_to_process)} documents",
            "doc_ids": [doc.doc_id for doc in docs_to_process],
            "processed_count": len(docs_to_process)
        }
        
    except Exception as e:
        print(f"Error starting batch processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error starting batch processing: {str(e)}")

@router.get("/status/{doc_id}", response_model=DocumentStatusResponse)
async def get_document_status(
    doc_id: str,
    vstore_svc: VectorStoreService = Depends(get_vector_store_service)
):
    """Get document processing status"""
    try:
        doc_status = vstore_svc.get_document_status(doc_id)
        if not doc_status:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
        
        return doc_status
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting document status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting status: {str(e)}")

@router.get("/status", response_model=List[DocumentStatusResponse])
async def get_all_document_statuses(
    collection_name: str = "default",
    vstore_svc: VectorStoreService = Depends(get_vector_store_service)
):
    """Get status of all documents in a collection"""
    try:
        return vstore_svc.get_all_document_statuses(collection_name)
        
    except Exception as e:
        print(f"Error getting all document statuses: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting statuses: {str(e)}")

if __name__ == "__main__":
    print("docs_api.py can be tested by running the main FastAPI application.")
