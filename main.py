import logging
import uvicorn
from src.setting import API_HOST, API_PORT, API_RELOAD

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logger.info(f"Starting Chino Kafuu AI System API on {API_HOST}:{API_PORT}")
    uvicorn.run(
        "api.app:create_app",
        host=API_HOST,
        port=API_PORT,
        reload=API_RELOAD,
        factory=True
    )
