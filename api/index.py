import sys
import os
from pathlib import Path

# Ensure project root is on sys.path for Vercel Serverless Functions
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Set VERCEL flag
os.environ.setdefault("VERCEL", "1")

from backend.main import app
