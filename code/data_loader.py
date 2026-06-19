"""
data_loader.py
--------------
Loads all dataset files. Returns plain Python dicts/lists.
No model calls. No business logic.
"""
import csv
import os
import base64
from typing import Optional
from config import (
    CLAIMS_CSV, SAMPLE_CLAIMS_CSV, USER_HISTORY_CSV,
    EVIDENCE_REQ_CSV, DATASET_DIR,
)


# ---------------------------------------------------------------------------
# CSV loaders
# ---------------------------------------------------------------------------

def load_claims(path: str = CLAIMS_CSV) -> list[dict]:
    """Load claims.csv or sample_claims.csv as a list of row dicts."""
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_user_history(path: str = USER_HISTORY_CSV) -> dict[str, dict]:
    """
    Load user_history.csv and index it by user_id.
    Returns: { "user_001": {all columns as strings}, ... }
    """
    index = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            index[row["user_id"]] = row
    return index


def load_evidence_requirements(path: str = EVIDENCE_REQ_CSV) -> list[dict]:
    """
    Load evidence_requirements.csv as a list of dicts.
    Each dict has: requirement_id, claim_object, applies_to,
                   minimum_image_evidence
    """
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def get_evidence_requirement(
    requirements: list[dict],
    claim_object: str,
    issue_family: str,
) -> Optional[str]:
    """
    Look up the minimum_image_evidence text for a given
    claim_object + issue_family combination.

    Checks:
      1. Exact match on both claim_object and applies_to
      2. claim_object == "all" catch-all rows
    Returns the minimum_image_evidence string, or None if no rule found.
    """
    # Normalize
    obj = claim_object.strip().lower()
    family = issue_family.strip().lower()

    exact = None
    catch_all = None

    for req in requirements:
        req_obj    = req.get("claim_object", "").strip().lower()
        req_family = req.get("applies_to",   "").strip().lower()

        if req_obj == obj and req_family == family:
            exact = req["minimum_image_evidence"]
        elif req_obj == "all" and req_family == family:
            catch_all = req["minimum_image_evidence"]

    return exact or catch_all


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def resolve_image_path(relative_path: str) -> str:
    """
    Convert the CSV-relative path  (e.g. images/test/case_001/img_1.jpg)
    to an absolute filesystem path anchored at DATASET_DIR.
    """
    return os.path.join(DATASET_DIR, relative_path.strip())


def image_to_base64(abs_path: str) -> Optional[tuple[str, str]]:
    """
    Read an image file and return (media_type, base64_data).
    Returns None if the file does not exist or cannot be read.
    """
    if not os.path.isfile(abs_path):
        return None
    ext = os.path.splitext(abs_path)[1].lower()
    media_map = {
        ".jpg":  "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png":  "image/png",
        ".gif":  "image/gif",
        ".webp": "image/webp",
    }
    media_type = media_map.get(ext, "image/jpeg")
    with open(abs_path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")
    return media_type, data


def load_images_for_claim(image_paths_str: str) -> list[dict]:
    """
    Parse the semicolon-separated image_paths field and load each image.

    Returns a list of dicts:
      {
        "image_id":   "img_1",
        "rel_path":   "images/test/case_001/img_1.jpg",
        "abs_path":   "/absolute/path/...",
        "media_type": "image/jpeg",
        "b64_data":   "<base64 string>",
        "loaded":     True/False,
      }
    """
    results = []
    paths = [p.strip() for p in image_paths_str.split(";") if p.strip()]
    for rel_path in paths:
        image_id = os.path.splitext(os.path.basename(rel_path))[0]
        abs_path = resolve_image_path(rel_path)
        encoded  = image_to_base64(abs_path)
        if encoded:
            media_type, b64_data = encoded
            results.append({
                "image_id":   image_id,
                "rel_path":   rel_path,
                "abs_path":   abs_path,
                "media_type": media_type,
                "b64_data":   b64_data,
                "loaded":     True,
            })
        else:
            results.append({
                "image_id":   image_id,
                "rel_path":   rel_path,
                "abs_path":   abs_path,
                "media_type": None,
                "b64_data":   None,
                "loaded":     False,
            })
    return results


# ---------------------------------------------------------------------------
# User history helpers
# ---------------------------------------------------------------------------

def get_user_risk_summary(
    user_history: dict[str, dict],
    user_id: str,
) -> dict:
    """
    Extract risk-relevant fields for a user.
    Returns a dict with numeric fields parsed, or a safe default if
    the user is not found in history.
    """
    default = {
        "found": False,
        "past_claim_count": 0,
        "rejected_claim": 0,
        "manual_review_claim": 0,
        "last_90_days_claim_count": 0,
        "history_flags": "",
        "history_summary": "",
        "accept_claim": 0,
    }
    if user_id not in user_history:
        return default

    row = user_history[user_id]

    def safe_int(val):
        try:
            return int(val)
        except (ValueError, TypeError):
            return 0

    return {
        "found":                      True,
        "past_claim_count":           safe_int(row.get("past_claim_count")),
        "rejected_claim":             safe_int(row.get("rejected_claim")),
        "manual_review_claim":        safe_int(row.get("manual_review_claim")),
        "last_90_days_claim_count":   safe_int(row.get("last_90_days_claim_count")),
        "history_flags":              row.get("history_flags", ""),
        "history_summary":            row.get("history_summary", ""),
        "accept_claim":               safe_int(row.get("accept_claim")),
    }
