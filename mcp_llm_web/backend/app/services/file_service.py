import uuid
import os
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
from app.core.config import settings
from app.models.schemas import UploadedFile, FileStatus


class FileService:
    def __init__(self):
        self.files: Dict[str, UploadedFile] = {}

    async def save_file(
        self, filename: str, content: bytes, original_name: str, file_type: str
    ) -> UploadedFile:
        file_id = str(uuid.uuid4())

        ext = Path(filename).suffix.lower()
        safe_filename = f"{file_id}{ext}"
        file_path = settings.UPLOADS_DIR / safe_filename

        with open(file_path, "wb") as f:
            f.write(content)

        file_size = len(content)

        metadata = await self._extract_metadata(file_path, file_type)

        uploaded_file = UploadedFile(
            id=file_id,
            filename=safe_filename,
            original_name=original_name,
            file_path=str(file_path),
            file_type=file_type,
            file_size=file_size,
            status=FileStatus.UPLOADED,
            metadata=metadata,
        )

        self.files[file_id] = uploaded_file
        return uploaded_file

    async def _extract_metadata(
        self, file_path: Path, file_type: str
    ) -> Dict[str, Any]:
        metadata = {"file_size": file_path.stat().st_size if file_path.exists() else 0}

        if file_type in ["ply", "las", "laz"]:
            try:
                import laspy

                with laspy.open(file_path) as las:
                    header = las.header
                    metadata.update(
                        {
                            "point_count": header.point_count
                            if hasattr(header, "point_count")
                            else 0,
                            "x_range": [header.x_min, header.x_max]
                            if hasattr(header, "x_min")
                            else None,
                            "y_range": [header.y_min, header.y_max]
                            if hasattr(header, "y_min")
                            else None,
                            "z_range": [header.z_min, header.z_max]
                            if hasattr(header, "z_min")
                            else None,
                        }
                    )
            except Exception:
                metadata["point_count"] = "unknown"

        return metadata

    async def get_file(self, file_id: str) -> Optional[UploadedFile]:
        return self.files.get(file_id)

    async def list_files(self) -> list:
        return list(self.files.values())

    async def delete_file(self, file_id: str) -> bool:
        if file_id in self.files:
            file_obj = self.files[file_id]
            try:
                if os.path.exists(file_obj.file_path):
                    os.remove(file_obj.file_path)
            except Exception:
                pass
            del self.files[file_id]
            return True
        return False

    async def get_file_path(self, file_id: str) -> Optional[str]:
        file_obj = await self.get_file(file_id)
        if file_obj:
            return file_obj.file_path
        return None

    async def get_file_info(self, file_id: str) -> Optional[Dict[str, Any]]:
        file_obj = await self.get_file(file_id)
        if file_obj:
            return {
                "id": file_obj.id,
                "original_name": file_obj.original_name,
                "file_type": file_obj.file_type,
                "file_size": file_obj.file_size,
                "status": file_obj.status,
                "metadata": file_obj.metadata,
                "uploaded_at": file_obj.uploaded_at.isoformat(),
            }
        return None


file_service = FileService()
