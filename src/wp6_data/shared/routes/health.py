"""Health and version endpoints (no auth: k8s probes and open pages poll them)."""

from fastapi import APIRouter, Response

from wp6_data.shared.templates import config

router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/version")
async def version(response: Response) -> dict[str, str | None]:
    """The running build, compared by open pages against the one they loaded.

    No-store: a cached answer would hide exactly the change being polled for.
    """
    response.headers["Cache-Control"] = "no-store"
    return {"version": config._app_version or None}
