"""
Legacy app.py - Updated to use the new restructured code.

This file is kept for backward compatibility.
The new structured code is in the src/ directory.
"""

from src.main import app
import uvicorn
from src.core.config import settings

# Re-export the app for backward compatibility
__all__ = ["app"]

if __name__ == "__main__":

    uvicorn.run(
        "app:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )
