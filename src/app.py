from fastapi import FastAPI

from api.controllers import (
    encoder_controller,
    music_base_controller,
    health_controller,
    feature_extraction_controller,
    generator_controller,
    emotion_mapper_controller,
    captioning_controller,
)
from handlers.exception_handler import add_exception_handlers

tags_metadata = [
    {
        "name": "health",
        "description": "checks the health of the API services",
    },
    {
        "name": "music_base",
        "description": "Music Base",
    },
    {
        "name": "encoder",
        "description": "Encoder",
    },
    {
        "name": "feature_extraction",
        "description": "Feature Extraction Module",
    },
    {
        "name": "emotion_mapper",
        "description": "Emotion Mapper",
    },
    {
        "name": "captioning",
        "description": "Captioning",
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
    music_base_controller.router,
    prefix="/music_base",
    tags=["music_base"],
    responses={404: {"description": "Not found"}},
)
app.include_router(
    encoder_controller.router,
    prefix="/encoder",
    tags=["encoder"],
    responses={404: {"description": "Not found"}},
)
app.include_router(
    feature_extraction_controller.router,
    prefix="/feature_extraction",
    tags=["feature_extraction"],
    responses={404: {"description": "Not found"}},
)
app.include_router(
    emotion_mapper_controller.router,
    prefix="/emotion_mapper",
    tags=["emotion_mapper"],
    responses={404: {"description": "Not found"}},
)
app.include_router(
    captioning_controller.router,
    prefix="/captioning",
    tags=["captioning"],
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
