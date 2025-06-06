"""FastAPI application wiring together all agent routes."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from logging_config import setup_logging

# configure logging before creating the FastAPI application
setup_logging()

from agent import register_endpoints

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
