import uuid
from pathlib import Path
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.core.config import settings
from app.models.database import db_service

files_router = APIRouter(prefix="/files", tags=["files"])


def get_file_type(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    type_map = {
        ".ply": "ply",
        ".las": "las",
        ".laz": "laz",
        ".tif": "tif",
        ".tiff": "tiff",
        ".shp": "shp",
        ".geojson": "geojson",
    }
    return type_map.get(ext, "unknown")


@files_router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()

    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"不支持的文件格式: {ext}")

    content = await file.read()

    if len(content) > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"文件大小超过限制: {settings.MAX_UPLOAD_SIZE / 1024 / 1024}MB",
        )

    file_type = get_file_type(file.filename or "unknown")
    file_id = str(uuid.uuid4())
    file_path = settings.UPLOADS_DIR / f"{file_id}_{file.filename}"

    with open(file_path, "wb") as f:
        f.write(content)

    file_db = await db_service.add_file(
        file_id=file_id,
        filename=file.filename or "unknown",
        file_type=file_type,
        file_size=len(content),
        file_path=str(file_path),
    )

    return {
        "file_id": file_db.id,
        "filename": file_db.filename,
        "file_type": file_db.file_type,
        "file_size": file_db.file_size,
    }


@files_router.get("/list")
async def list_files():
    files = await db_service.list_files()
    return {
        "files": [
            {
                "id": f.id,
                "filename": f.filename,
                "file_type": f.file_type,
                "file_size": f.file_size,
                "uploaded_at": f.uploaded_at.isoformat(),
            }
            for f in files
        ]
    }


@files_router.get("/{file_id}")
async def get_file_info(file_id: str):
    file_info = await db_service.get_file(file_id)
    if not file_info:
        raise HTTPException(status_code=404, detail="文件不存在")
    return {
        "id": file_info.id,
        "filename": file_info.filename,
        "file_type": file_info.file_type,
        "file_size": file_info.file_size,
        "file_path": file_info.file_path,
        "uploaded_at": file_info.uploaded_at.isoformat(),
    }


@files_router.delete("/{file_id}")
async def delete_file(file_id: str):
    file_info = await db_service.get_file(file_id)
    if not file_info:
        raise HTTPException(status_code=404, detail="文件不存在")

    file_path = Path(file_info.file_path)
    if file_path.exists():
        file_path.unlink()

    await db_service.delete_file(file_id)
    return {"success": True}
