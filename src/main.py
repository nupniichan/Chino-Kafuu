"""
FastAPI application for testing AI modules (STT, TTS, RVC, etc).
Entry point for the API server.
"""
import logging
import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
sys.path.insert(0, str(Path(__file__).parent))
from api.routes import stt, base, dialog
from setting import API_HOST, API_PORT, API_RELOAD

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Module Tester API",
    description="API for testing STT, TTS, RVC and other AI modules",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(stt.router)
app.include_router(base.router)
app.include_router(dialog.router)

@app.get("/")
async def root():
    """Root endpoint with API information and available endpoints."""
    return JSONResponse(
        status_code=200,
        content={
            "message": "AI Module API",
            "version": "0.1.0",
            "endpoints": {
                "stt": "/stt/transcribe",
                "health": "/base/health",
                "docs": "/docs",
                "redoc": "/redoc"
            }
        }
    )

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting API server on {API_HOST}:{API_PORT}")
    uvicorn.run(
        "main:app",
        host=API_HOST,
        port=API_PORT,
        reload=API_RELOAD
    )