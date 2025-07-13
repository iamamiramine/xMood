from fastapi import FastAPI

from api.controllers import (
    symbolic_controller,
    health_controller,
    latent_controller,
    generator_controller,
    pipeline_controller,
)
from handlers.exception_handler import add_exception_handlers

tags_metadata = [
    {
        "name": "health",
        "description": "checks the health of the API services",
    },
    {
        "name": "pipeline",
        "description": "Pipeline Orchestration",
    },
    {
        "name": "symbolic",
        "description": "Encoder",
    },
    {
        "name": "latent",
        "description": "Feature Extraction Module",
    },
    {
        "name": "generator",
        "description": "Generator",
    },
]


app = FastAPI(
    version="1.0",
    title="PA-AI API",
    description="API for PA-AI",
    openapi_tags=tags_metadata,
)

app.include_router(
    health_controller.router,
    prefix="/health",
    tags=["health"],
    responses={404: {"description": "Not found"}},
)
app.include_router(
    pipeline_controller.router,
    prefix="/pipeline",
    tags=["pipeline"],
    responses={404: {"description": "Not found"}},
)
app.include_router(
    symbolic_controller.router,
    prefix="/symbolic",
    tags=["symbolic"],
    responses={404: {"description": "Not found"}},
)
app.include_router(
    latent_controller.router,
    prefix="/latent",
    tags=["latent"],
    responses={404: {"description": "Not found"}},
)
app.include_router(
    generator_controller.router,
    prefix="/generator",
    tags=["generator"],
    responses={404: {"description": "Not found"}},
)


add_exception_handlers(app=app)


# if __name__ == "__main__":
#     import uvicorn
#
#     uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
