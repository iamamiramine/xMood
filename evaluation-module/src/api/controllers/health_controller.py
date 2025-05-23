from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def get_health():
    """
    Returns the health status of the service.
    """
    return {"status": "healthy"}


@router.get("/ping")
async def ping():
    """
    Returns a simple ping response.
    """
    return {"ping": "pong"} 