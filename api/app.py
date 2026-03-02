import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.core.bootstrap import registry
from api.routes import stt, base, dialog, memory, system, tts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Chino Kafuu AI System...")
    await registry.startup()
    yield
    logger.info("Shutting down Chino Kafuu AI System...")
    await registry.shutdown()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Chino Kafuu AI System API",
        description="API for STT, TTS, Dialog, and Memory modules",
        version="0.2.0",
        lifespan=lifespan,
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
    app.include_router(tts.router)
    app.include_router(dialog.router)
    app.include_router(memory.router)
    app.include_router(system.router)

    @app.get("/")
    async def root():
        return JSONResponse(
            status_code=200,
            content={
                "message": "Chino Kafuu AI System API",
                "version": "0.2.0",
                "endpoints": {
                    "health": "/base/ping",
                    "stt": "/stt/transcribe",
                    "tts": "/tts/synthesize",
                    "dialog": "/dialog/chat",
                    "memory": "/memory/stats",
                    "system": "/system/info",
                    "event_bus": "/system/event-bus-stats",
                    "docs": "/docs",
                },
            },
        )

    return app
