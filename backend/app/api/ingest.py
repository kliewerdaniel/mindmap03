from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from pydantic import BaseModel, validator
from typing import List
from pathlib import Path
from ..db.db import insert_note
from ..services.extractor import process_note
from ..config import settings
import zipfile
import io

router = APIRouter()

class IngestTextRequest(BaseModel):
    filename: str
    content: str
    source_path: str = None

    @validator('filename')
    def validate_filename(cls, v):
        """Validate filename has allowed extension."""
        if not v:
            raise ValueError('Filename cannot be empty')
        ext = Path(v).suffix.lower()
        if ext not in settings.allowed_extensions:
            raise ValueError(f'File extension {ext} not allowed. Allowed: {settings.allowed_extensions}')
        return v

    @validator('content')
    def validate_content_size(cls, v):
        """Validate content size does not exceed max upload size."""
        content_size = len(v.encode('utf-8'))
        if content_size > settings.max_upload_size:
            raise ValueError(f'Content size {content_size} bytes exceeds maximum allowed size of {settings.max_upload_size} bytes')
        return v

class IngestResponse(BaseModel):
    note_id: int
    status: str
    message: str

@router.post("/text", response_model=IngestResponse)
async def ingest_text(payload: IngestTextRequest, background_tasks: BackgroundTasks):
    """
    Ingest text content for processing.

    Saves note to database and triggers asynchronous extraction.
    """
    try:
        # Insert note
        note_id = insert_note(
            payload.filename,
            payload.content,
            payload.source_path
        )

        # Process in background
        background_tasks.add_task(process_note, note_id)

        return IngestResponse(
            note_id=note_id,
            status="accepted",
            message="Note saved and queued for processing"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/file")
async def ingest_file(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Ingest markdown file(s) for processing.

    Supports single .md files or .zip archives containing multiple .md files.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    # Check file extension against allowed extensions
    ext = Path(file.filename).suffix.lower()
    if ext not in settings.allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"File extension {ext} not allowed. Allowed extensions: {settings.allowed_extensions}"
        )

    content = await file.read()

    # Check file size against max upload size
    if len(content) > settings.max_upload_size:
        raise HTTPException(
            status_code=413,
            detail=f"File size {len(content)} bytes exceeds maximum allowed size of {settings.max_upload_size} bytes"
        )
    note_ids = []

    try:
        if file.filename.endswith('.zip'):
            # Handle zip archive
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                for filename in zf.namelist():
                    if filename.endswith(('.md', '.txt')):
                        file_content = zf.read(filename).decode('utf-8')
                        note_id = insert_note(filename, file_content, file.filename)
                        note_ids.append(note_id)

                        # Process in background
                        if background_tasks:
                            background_tasks.add_task(process_note, note_id)
        else:
            # Single file
            file_content = content.decode('utf-8')
            note_id = insert_note(file.filename, file_content, file.filename)
            note_ids.append(note_id)

            # Process in background
            if background_tasks:
                background_tasks.add_task(process_note, note_id)

        return {
            "status": "accepted",
            "note_ids": note_ids,
            "message": f"Ingested {len(note_ids)} file(s), processing started"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{note_id}")
async def get_ingestion_status(note_id: int):
    """Check processing status of a note."""
    from ..db.db import get_note, get_extracts_for_note

    note = get_note(note_id)
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")

    extracts = get_extracts_for_note(note_id)

    return {
        "note_id": note_id,
        "processed": bool(note['processed']),
        "num_extracts": len(extracts),
        "created_at": note['created_at']
    }
