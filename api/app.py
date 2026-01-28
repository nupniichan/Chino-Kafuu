import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from api.routes import stt, base, dialog, memory, system

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Chino Kafuu AI System API",
        description="API for STT, TTS, Dialog, and Memory modules",
        version="0.1.0"
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(base.router)
    app.include_router(stt.router)
    app.include_router(dialog.router)
    app.include_router(memory.router)
    app.include_router(system.router)

    @app.get("/")
    async def root():
        return JSONResponse(
            status_code=200,
            content={
                "message": "Chino Kafuu AI System API",
                "version": "0.1.0",
                "endpoints": {
                    "health": "/base/health",
                    "stt": {
                        "transcribe": "/stt/transcribe"
                    },
                    "dialog": {
                        "chat": "/dialog/chat",
                        "history": "/dialog/history",
                        "clear": "/dialog/clear",
                        "status": "/dialog/status",
                        "memory_stats": "/dialog/memory_stats"
                    },
                    "memory": {
                        "short_term": "/memory/short-term/*",
                        "long_term": "/memory/long-term/*",
                        "stats": "/memory/stats"
                    },
                    "system": {
                        "info": "/system/info",
                        "config": "/system/config",
                        "health": "/system/health-detailed"
                    },
                    "docs": "/docs",
                    "redoc": "/redoc"
                }
            }
        )

    return app
