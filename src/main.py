"""FastAPI application wiring together all agent routes."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
import logging
from pathlib import Path

from agent import register_endpoints

# Configure logging to write both to stdout and to a log file
log_dir = Path(__file__).resolve().parent.parent / "logs"
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "app.log"),
        logging.StreamHandler(),
    ],
)

app = FastAPI(
    title="Agent Boilerplate",
    description="Minimal FastAPI app using OpenAI Agents SDK",
)

app.include_router(register_endpoints(), prefix="/api")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def redirect_to_docs():
    """Redirect the root path to the interactive documentation."""

    return RedirectResponse(url="/docs")
