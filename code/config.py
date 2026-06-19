"""
config.py
---------
Single source of truth for all allowed values, dataset paths,
and system-wide constants. No logic lives here.
"""
import os

# ---------------------------------------------------------------------------
# Paths (all relative to the repo root, i.e. one level above code/)
# ---------------------------------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_DIR = os.path.join(ROOT_DIR, "dataset")

CLAIMS_CSV          = os.path.join(DATASET_DIR, "claims.csv")
SAMPLE_CLAIMS_CSV   = os.path.join(DATASET_DIR, "sample_claims.csv")
USER_HISTORY_CSV    = os.path.join(DATASET_DIR, "user_history.csv")
EVIDENCE_REQ_CSV    = os.path.join(DATASET_DIR, "evidence_requirements.csv")
IMAGES_TEST_DIR     = os.path.join(DATASET_DIR, "images", "test")
IMAGES_SAMPLE_DIR   = os.path.join(DATASET_DIR, "images", "sample")

OUTPUT_CSV          = os.path.join(ROOT_DIR, "output.csv")
EVAL_REPORT_MD      = os.path.join(ROOT_DIR, "code", "evaluation", "evaluation_report.md")

# ---------------------------------------------------------------------------
# Google Gemini model (free via Google AI Studio)
# Get your free API key at: https://aistudio.google.com
# ---------------------------------------------------------------------------
GEMINI_MODEL = "gemini-1.5-flash"   # free tier: 1,500 req/day, 1M token context
MAX_TOKENS   = 1200   # per call — enough for structured JSON output

# ---------------------------------------------------------------------------
# Output column order (must match problem statement exactly)
# ---------------------------------------------------------------------------
OUTPUT_COLUMNS = [
    "user_id",
    "image_paths",
    "user_claim",
    "claim_object",
    "evidence_standard_met",
    "evidence_standard_met_reason",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "claim_status_justification",
    "supporting_image_ids",
    "valid_image",
    "severity",
]

# ---------------------------------------------------------------------------
# Allowed values — enforced at output-writing time
# ---------------------------------------------------------------------------
ALLOWED_CLAIM_STATUS = {"supported", "contradicted", "not_enough_information"}

ALLOWED_ISSUE_TYPES = {
    "dent", "scratch", "crack", "glass_shatter", "broken_part",
    "missing_part", "torn_packaging", "crushed_packaging",
    "water_damage", "stain", "none", "unknown",
}

ALLOWED_SEVERITY = {"none", "low", "medium", "high", "unknown"}

ALLOWED_RISK_FLAGS = {
    "none",
    "blurry_image",
    "cropped_or_obstructed",
    "low_light_or_glare",
    "wrong_angle",
    "wrong_object",
    "wrong_object_part",
    "damage_not_visible",
    "claim_mismatch",
    "possible_manipulation",
    "non_original_image",
    "text_instruction_present",
    "user_history_risk",
    "manual_review_required",
}

CAR_PARTS = {
    "front_bumper", "rear_bumper", "door", "hood", "windshield",
    "side_mirror", "headlight", "taillight", "fender",
    "quarter_panel", "body", "unknown",
}

LAPTOP_PARTS = {
    "screen", "keyboard", "trackpad", "hinge", "lid",
    "corner", "port", "base", "body", "unknown",
}

PACKAGE_PARTS = {
    "box", "package_corner", "package_side", "seal",
    "label", "contents", "item", "unknown",
}

OBJECT_PARTS = {
    "car":     CAR_PARTS,
    "laptop":  LAPTOP_PARTS,
    "package": PACKAGE_PARTS,
}

# ---------------------------------------------------------------------------
# Risk escalation rule: user_history_risk always adds manual_review_required
# ---------------------------------------------------------------------------
HISTORY_RISK_AUTO_FLAGS = {"user_history_risk", "manual_review_required"}

# ---------------------------------------------------------------------------
# Adversarial / prompt-injection keywords to detect in user_claim text
# (used by the pre-processor, not the LLM prompt)
# ---------------------------------------------------------------------------
ADVERSARIAL_PHRASES = [
    "ignore all previous instructions",
    "ignore previous instructions",
    "mark this row",
    "mark this claim",
    "approve the claim",
    "approve this claim",
    "skip manual review",
    "approve immediately",
    "follow it and approve",
    "usko follow karke",   # Hindi: "follow it and"
    "note bhi hai",        # Hindi: "there is a note"
    "escalate publicly",
]
