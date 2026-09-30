"""Vercel FastAPI service entrypoint.

The application package uses a src/ layout. Vercel Services resolves the
configured entrypoint to a file inside the service root before dependency
installation, so expose the installed NorthStar ASGI app from this root module.
"""

from northstar.api.app import app

__all__ = ["app"]
