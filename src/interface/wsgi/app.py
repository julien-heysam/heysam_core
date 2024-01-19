import logging

from fastapi import FastAPI
from fastapi.concurrency import asynccontextmanager

from src.interface.wsgi.routes.probes import probe_route

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.cache = {}


def build_app() -> FastAPI:
    app = FastAPI(
        title="",
        description="",
        summary="",
        version="0.0.1",
        contact={
            "name": "",
            "url": "",
        },
        license_info={
            "name": "Apache 2.0",
            "identifier": "MIT",
        },
        lifespan=lifespan,
    )
    app.include_router(probe_route)

    return app


app = build_app()
