"""FastAPI router exposing endpoints for the example agents."""

from fastapi import APIRouter
from pydantic import BaseModel

from .runner import run, MultiAgentOutput

router = APIRouter(prefix="/agent-one", tags=["agent_one"])


class InputRequest(BaseModel):
    """Payload accepted by the multi-agent runner."""

    text: str


@router.post("/run", response_model=MultiAgentOutput)
async def orchestrate(request: InputRequest) -> MultiAgentOutput:
    """Trigger the summary agent followed by the follow-up agent."""

    return await run(request.text)


def register_endpoints() -> APIRouter:
    """Return the router so it can be included from the main app."""

    return router

