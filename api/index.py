import sys
import os
from pathlib import Path

# Ensure project root is on sys.path for Vercel Serverless Functions
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Set VERCEL flag
os.environ.setdefault("VERCEL", "1")

from backend.main import app as fastapi_app

class VercelPathASGIWrapper:
    def __init__(self, asgi_app):
        self.asgi_app = asgi_app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            headers = dict(scope.get("headers", []))
            matched = (
                headers.get(b"x-matched-path") or
                headers.get(b"x-rewrite-path") or
                headers.get(b"x-invoke-path") or
                headers.get(b"x-original-uri")
            )
            if matched:
                clean = matched.decode("utf-8").split("?")[0]
                if not clean.endswith(".py"):
                    scope["path"] = clean
                    scope["raw_path"] = clean.encode("utf-8")
            elif scope.get("path", "").endswith("/index.py"):
                clean = scope["path"].replace("/index.py", "")
                if clean:
                    scope["path"] = clean
                    scope["raw_path"] = clean.encode("utf-8")
        await self.asgi_app(scope, receive, send)

app = VercelPathASGIWrapper(fastapi_app)

