import logging
from fastapi import APIRouter
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/base", tags=["Base"])


@router.get("/ping")
async def ping():
    return {"status": "ok", "message": "pong"}
