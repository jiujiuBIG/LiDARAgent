import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
RESULTS_DIR = DATA_DIR / "results"
CACHE_DIR = DATA_DIR / "cache"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://127.0.0.1:8000/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "forestmind")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")

MCP_TOOLS_DIR = BASE_DIR / "tools"

DATABASE_URL = f"sqlite+aiosqlite:///{DATA_DIR}/forestmind.db"


class Settings:
    PROJECT_NAME: str = "LiDAR Agent - 点云智能分析系统"
    VERSION: str = "1.0.0"
    API_V1_PREFIX: str = "/api/v1"

    BASE_DIR: Path = BASE_DIR
    DATA_DIR: Path = DATA_DIR
    UPLOADS_DIR: Path = UPLOADS_DIR
    RESULTS_DIR: Path = RESULTS_DIR
    CACHE_DIR: Path = CACHE_DIR
    MCP_TOOLS_DIR: Path = MCP_TOOLS_DIR
    DATABASE_URL: str = DATABASE_URL

    LLM_BASE_URL: str = LLM_BASE_URL
    LLM_MODEL: str = LLM_MODEL
    LLM_API_KEY: str = LLM_API_KEY

    MAX_CONCURRENT_TASKS: int = 5
    TASK_TIMEOUT: int = 300

    MAX_UPLOAD_SIZE: int = 1024 * 1024 * 1024
    ALLOWED_EXTENSIONS = {".ply", ".las", ".laz", ".tif", ".tiff", ".shp", ".geojson"}


settings = Settings()
