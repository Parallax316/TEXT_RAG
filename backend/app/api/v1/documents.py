from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from typing import List, Optional
from pathlib import Path
import os
import uuid
from datetime import datetime

from backend.app.services.doc_parser_fast import DocParserFastService, process_document_background
from backend.app.services.vlm_parser_service import VlmParserService
from backend.app.services.vstore_svc import VectorStoreService

router = APIRouter()

@router.post("/upload-multiple-vlm")
async def upload_multiple_documents_vlm(
    files: List[UploadFile] = File(...),
    collection_name: Optional[str] = None,
    background_tasks: BackgroundTasks = None
):
    """
    Upload multiple documents and process them using VLM-based parsing.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    # Create upload directory if it doesn't exist
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)

    # Initialize services
    vector_store_service = VectorStoreService()
    vlm_parser_service = VlmParserService(vector_store_service)

    # Process each file
    results = []
    for i, file in enumerate(files):
        try:
            # Generate unique ID for the document
            source_doc_id = str(uuid.uuid4())
            
            # Save file
            file_path = upload_dir / f"{source_doc_id}_{file.filename}"
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)

            # Add to background tasks if available
            if background_tasks:
                background_tasks.add_task(
                    process_document_background,
                    file_path=str(file_path),
                    source_doc_id=source_doc_id,
                    doc_parser_svc_instance=vlm_parser_service,
                    serial_no=i + 1,
                    total_count=len(files),
                    collection_name=collection_name
                )
                results.append({
                    "filename": file.filename,
                    "status": "processing",
                    "source_doc_id": source_doc_id
                })
            else:
                # Process immediately
                success = vlm_parser_service.process_document(
                    file_path=str(file_path),
                    source_doc_id=source_doc_id,
                    collection_name=collection_name
                )
                results.append({
                    "filename": file.filename,
                    "status": "success" if success else "failed",
                    "source_doc_id": source_doc_id
                })

        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "failed",
                "error": str(e)
            })

    return {
        "message": "Documents uploaded and processing started" if background_tasks else "Documents processed",
        "results": results
    } 