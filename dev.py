#!/usr/bin/env python3
"""Development script for running the application."""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def main():
    """Main entry point for development."""
    import uvicorn
    from src.core.config import settings

    print("Starting E-commerce Facecare Recommender...")
    print(f"Environment: {settings.environment}")
    print(f"Host: {settings.host}:{settings.port}")
    print(f"Log Level: {settings.log_level}")
    print("=" * 50)

    try:
        uvicorn.run(
            "src.main:app",
            host=settings.host,
            port=settings.port,
            reload=True,
            log_level=settings.log_level.lower(),
            access_log=True,
        )
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        print(f"Error starting application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
