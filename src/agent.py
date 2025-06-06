"""Helper for registering routers from each individual agent package."""

from fastapi import APIRouter

from agent_one import register_endpoints as register_agent_one_endpoints


def register_endpoints() -> APIRouter:
    """Collect and return all API routers exposed by individual agents."""

    router = APIRouter()
    router.include_router(register_agent_one_endpoints())
    return router
