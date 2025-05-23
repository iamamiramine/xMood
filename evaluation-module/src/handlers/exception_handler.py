from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import traceback


class MoodClassificationError(Exception):
    """Base exception for mood classification errors."""
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class FileProcessingError(MoodClassificationError):
    """Exception raised when there's an error processing input files."""
    pass


class ModelLoadingError(MoodClassificationError):
    """Exception raised when there's an error loading the model."""
    pass


class InferenceError(MoodClassificationError):
    """Exception raised when there's an error during inference."""
    pass


def add_exception_handlers(app: FastAPI):
    """Add exception handlers to the FastAPI app."""
    
    @app.exception_handler(MoodClassificationError)
    async def handle_mood_classification_error(
        request: Request, exc: MoodClassificationError
    ):
        return JSONResponse(
            status_code=400,
            content={"error": exc.__class__.__name__, "message": exc.message},
        )
    
    @app.exception_handler(FileProcessingError)
    async def handle_file_processing_error(
        request: Request, exc: FileProcessingError
    ):
        return JSONResponse(
            status_code=400,
            content={"error": "FileProcessingError", "message": exc.message},
        )
    
    @app.exception_handler(ModelLoadingError)
    async def handle_model_loading_error(
        request: Request, exc: ModelLoadingError
    ):
        return JSONResponse(
            status_code=503,  # Service Unavailable
            content={"error": "ModelLoadingError", "message": exc.message},
        )
    
    @app.exception_handler(InferenceError)
    async def handle_inference_error(
        request: Request, exc: InferenceError
    ):
        return JSONResponse(
            status_code=500,
            content={"error": "InferenceError", "message": exc.message},
        )
    
    @app.exception_handler(Exception)
    async def handle_general_exception(request: Request, exc: Exception):
        error_detail = str(exc)
        stack_trace = traceback.format_exc()
        
        print(f"Unhandled exception: {error_detail}")
        print(f"Stack trace: {stack_trace}")
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "InternalServerError",
                "message": "An unexpected error occurred. Please try again later.",
            },
        ) 