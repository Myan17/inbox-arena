"""Test configuration.

The server modules use flat imports (`from models import ...`) because the
container runs with the repo root on sys.path, so tests do the same.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "server"))
