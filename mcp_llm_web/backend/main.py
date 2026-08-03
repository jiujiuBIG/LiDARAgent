from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from app.core.config import settings
from app.api.chat import chat_router
from app.api.files import files_router
from app.models.database import init_db
from app.services.mcp_service import mcp_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"🚀 {settings.PROJECT_NAME} 启动中...")
    print(f"📡 API 服务: http://localhost:8001")
    print(f"🤖 模型: {settings.LLM_MODEL}")

    print("📦 初始化数据库...")
    await init_db()
    print("✅ 数据库已初始化")

    print("🔌 连接 MCP 工具服务器...")
    try:
        await mcp_service.connect_all_tools()
        print("✅ MCP 工具已加载")
    except Exception as e:
        print(f"⚠️ MCP 工具连接失败: {e}")

    yield

    print("🔌 关闭 MCP 连接...")
    await mcp_service.cleanup()
    print("👋 服务已关闭")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="🌲 点云智能分析系统 - 基于 DeepSeek 的林业数据分析助手",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router, prefix=settings.API_V1_PREFIX)
app.include_router(files_router, prefix=settings.API_V1_PREFIX)


@app.get("/")
async def root():
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "api": f"{settings.API_V1_PREFIX}",
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    return {"status": "ok", "model": settings.LLM_MODEL}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
