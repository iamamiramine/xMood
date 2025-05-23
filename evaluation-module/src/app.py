from fastapi import FastAPI

from api.controllers import (
    health_controller,
    evaluation_controller,
)
from handlers.exception_handler import add_exception_handlers

tags_metadata = [
    {
        "name": "health",
        "description": "checks the health of the API services",
    },
    {
        "name": "evaluation",
        "description": "Metrics evaluation for MIDI files",
    },
]


app = FastAPI(
    version="1.0",
    title="Evaluation Module API",
    description="API for MIDI Evaluation and Analysis",
    openapi_tags=tags_metadata,
)

app.include_router(
    health_controller.router,
    prefix="/health",
    tags=["health"],
    responses={404: {"description": "Not found"}},
)

app.include_router(
    evaluation_controller.router,
    prefix="/evaluation",
    tags=["evaluation"],
    responses={404: {"description": "Not found"}},
)

add_exception_handlers(app=app)


# if __name__ == "__main__":
#     import uvicorn

#     uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True) 