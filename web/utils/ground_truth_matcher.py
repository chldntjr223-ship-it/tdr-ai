import os
import json
import hashlib

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
INDEX_PATH = os.path.join(ROOT_DIR, "reference_index", "ground_truth_index.json")
if not os.path.exists(INDEX_PATH):
    INDEX_PATH = os.path.join(ROOT_DIR, "data", "reference_index", "ground_truth_index.json")

_GT_INDEX = None

def _load_index():
    global _GT_INDEX
    if os.path.exists(INDEX_PATH):
        with open(INDEX_PATH, "r", encoding="utf-8") as f:
            _GT_INDEX = json.load(f)
    else:
        _GT_INDEX = {}

def match_ground_truth(file_bytes):
    """
    Matches uploaded waveform bytes against the canonical reference index via SHA-256.
    Returns: ground truth dict if known reference waveform, otherwise None (public inference mode).
    """
    global _GT_INDEX
    if _GT_INDEX is None:
        _load_index()
    h = hashlib.sha256(file_bytes).hexdigest()
    return _GT_INDEX.get(h, None)

