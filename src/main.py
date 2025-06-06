# Initialize FastAPI
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
import logging

logger_to_configure = logging.getLogger("new.agents.main")
logger_to_configure.setLevel(logging.DEBUG)

if logger_to_configure.hasHandlers():
    logger_to_configure.handlers.clear()

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

log_file_name = "fastapi_logs.log"
fh = logging.FileHandler(log_file_name, mode='w')
fh.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)

logger_to_configure.addHandler(ch)
logger_to_configure.addHandler(fh)

logger_to_configure.propagate = False

from brainstorming import register_endpoints as register_brainstorming_endpoints  # noqa: E402
from analyzer import register_endpoints as register_analyzer_endpoints  # noqa: E402
from criteria import register_endpoints as register_criteria_endpoints  # noqa: E402


app_kwargs = {
        "title": "IM agents API",
        "description": "Providing API to interact with IM agents",
        "openapi_url": "/openapi.json",  # Explicitly set OpenAPI schema URL
        "docs_url": "/docs",  # Explicitly set docs URL
        "redoc_url": "/redoc",  # Explicitly set redoc URL
    }

app = FastAPI(**app_kwargs)

api_prefix = "/api"
app.include_router(register_brainstorming_endpoints(), prefix=api_prefix)
app.include_router(register_analyzer_endpoints(), prefix=api_prefix)
app.include_router(register_criteria_endpoints(), prefix=api_prefix)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

@app.get("/")
async def redirect_to_swagger():
    return RedirectResponse(url="/docs")