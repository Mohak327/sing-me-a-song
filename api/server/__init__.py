"""
Resound's API. It runs the ear model on short blocks of audio for the web page.

Locally the model is imported from the repository root. When hosted, only the
api/ folder is deployed, so the model comes from the copy in api/vendor
(see tools/sync_model.py).
"""
import sys
from pathlib import Path

API = Path(__file__).resolve().parents[1]
ROOT = API.parent
MODEL = ROOT if (ROOT / 'auditory_periphery.py').is_file() else API / 'vendor'
if str(MODEL) not in sys.path:
    sys.path.insert(0, str(MODEL))
