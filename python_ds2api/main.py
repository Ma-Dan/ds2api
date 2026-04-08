"""
DS2API Python Version - Main Application Entry Point
Converts DeepSeek Web API to OpenAI-compatible API.
"""
import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import config_store
from app.account_pool import account_pool
from app.deepseek_client import deepseek_client
from app.openai_api import router as openai_router


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting DS2API Python version...")

    # Load configuration
    config_store.load_config()
    logger.info(f"Loaded {len(config_store.config.accounts)} accounts")
    logger.info(f"Loaded {len(config_store.config.keys)} API keys")

    # Initialize account pool and login
    if config_store.config.accounts:
        logger.info("Logging in accounts...")
        await account_pool.login_all()

    logger.info("DS2API started successfully")

    yield

    # Shutdown
    logger.info("Shutting down DS2API...")
    await deepseek_client.close()
    logger.info("DS2API shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="DS2API Python",
    description="Convert DeepSeek Web API to OpenAI-compatible API",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoints
@app.get("/healthz")
async def healthz():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/readyz")
async def readyz():
    """Readiness check endpoint."""
    # Check if accounts are available
    if not config_store.config.accounts:
        return {"status": "not_ready", "reason": "No accounts configured"}

    # Check if at least one account has a token
    account_status = account_pool.get_status()
    ready_accounts = [a for a in account_status if a.get("has_token")]

    if not ready_accounts:
        return {"status": "not_ready", "reason": "No accounts with valid tokens"}

    return {"status": "ready", "accounts": len(ready_accounts)}


# Include routers
app.include_router(openai_router)


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "DS2API Python",
        "version": "1.0.0",
        "description": "DeepSeek Web API to OpenAI-compatible API converter",
        "endpoints": {
            "openai": "/v1",
            "models": "/v1/models",
            "chat": "/v1/chat/completions",
            "embeddings": "/v1/embeddings",
            "health": "/healthz",
            "ready": "/readyz"
        }
    }


def run():
    """Run the application."""
    import uvicorn

    port = config_store.settings.port
    log_level = config_store.settings.log_level.lower()

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level=log_level
    )


if __name__ == "__main__":
    run()
