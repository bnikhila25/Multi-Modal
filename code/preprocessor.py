"""
preprocessor.py
---------------
All logic that runs BEFORE the LLM call.
- Detects adversarial / prompt-injection content in the claim text
- Computes a user risk tier from user history data
- Extracts the actual damage claim from the conversation (heuristic)
- Identifies the issue family for evidence requirement lookup
No model calls here.
"""
from config import ADVERSARIAL_PHRASES, HISTORY_RISK_AUTO_FLAGS


# ---------------------------------------------------------------------------
# Adversarial detection
# ---------------------------------------------------------------------------

def detect_adversarial_text(user_claim: str) -> bool:
    """
    Return True if the claim conversation contains any known
    prompt-injection or manipulation phrase.
    Case-insensitive match.
    """
    lower = user_claim.lower()
    return any(phrase in lower for phrase in ADVERSARIAL_PHRASES)


# ---------------------------------------------------------------------------
# User risk scoring
# ---------------------------------------------------------------------------

def compute_user_risk_flags(user_summary: dict) -> set[str]:
    """
    Given a user_summary dict (from data_loader.get_user_risk_summary),
    return a set of risk flags to pre-add before the LLM call.

    Rules (aligned with sample_claims.csv patterns):
    - If rejected_claim >= 2                      → user_history_risk
    - If manual_review_claim >= 2                 → user_history_risk
    - If last_90_days_claim_count >= 3            → user_history_risk
    - If history_flags contains "fraud" or
      "manipulation" or "exaggeration"            → user_history_risk
    - Any user_history_risk                       → also manual_review_required
    - If user not found in history at all         → no flags (unknown is not risk)
    """
    flags = set()

    if not user_summary.get("found"):
        return flags

    rejected  = user_summary.get("rejected_claim", 0)
    manual    = user_summary.get("manual_review_claim", 0)
    recent    = user_summary.get("last_90_days_claim_count", 0)
    hflags    = user_summary.get("history_flags", "").lower()
    hsummary  = user_summary.get("history_summary", "").lower()

    risk_keywords = {"fraud", "manipulation", "exaggeration",
                     "suspicious", "escalat", "repeat", "rejected"}

    if rejected >= 2:
        flags.add("user_history_risk")
    if manual >= 2:
        flags.add("user_history_risk")
    if recent >= 3:
        flags.add("user_history_risk")
    if any(kw in hflags for kw in risk_keywords):
        flags.add("user_history_risk")
    if any(kw in hsummary for kw in risk_keywords):
        flags.add("user_history_risk")

    if "user_history_risk" in flags:
        flags.add("manual_review_required")

    return flags


def build_user_history_context_text(user_summary: dict) -> str:
    """
    Build a concise plain-text summary of the user's history
    to inject into the LLM prompt.
    """
    if not user_summary.get("found"):
        return "No user history found."

    parts = [
        f"Past claims: {user_summary['past_claim_count']}",
        f"Accepted: {user_summary['accept_claim']}",
        f"Rejected: {user_summary['rejected_claim']}",
        f"Manual review: {user_summary['manual_review_claim']}",
        f"Claims in last 90 days: {user_summary['last_90_days_claim_count']}",
    ]
    if user_summary.get("history_flags"):
        parts.append(f"Flags: {user_summary['history_flags']}")
    if user_summary.get("history_summary"):
        parts.append(f"Summary: {user_summary['history_summary']}")

    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Issue family mapping
# (used to look up the right evidence requirement row)
# ---------------------------------------------------------------------------

# Maps issue_type values → issue family used in evidence_requirements.applies_to
ISSUE_TYPE_TO_FAMILY = {
    "dent":             "dent",
    "scratch":          "scratch",
    "crack":            "crack",
    "glass_shatter":    "glass_shatter",
    "broken_part":      "broken_part",
    "missing_part":     "missing_part",
    "torn_packaging":   "torn_packaging",
    "crushed_packaging":"crushed_packaging",
    "water_damage":     "water_damage",
    "stain":            "stain",
    "none":             "none",
    "unknown":          "unknown",
}

# Keyword → probable issue family (used for pre-lookup before LLM call)
CLAIM_TEXT_TO_FAMILY = {
    "dent":           "dent",
    "dented":         "dent",
    "bumped":         "dent",
    "scratch":        "scratch",
    "scratched":      "scratch",
    "scrape":         "scratch",
    "crack":          "crack",
    "cracked":        "crack",
    "shatter":        "glass_shatter",
    "shattered":      "glass_shatter",
    "broken":         "broken_part",
    "broke":          "broken_part",
    "missing":        "missing_part",
    "missing_part":   "missing_part",
    "torn":           "torn_packaging",
    "tear":           "torn_packaging",
    "crushed":        "crushed_packaging",
    "crush":          "crushed_packaging",
    "water":          "water_damage",
    "wet":            "water_damage",
    "liquid":         "water_damage",
    "stain":          "stain",
    "stained":        "stain",
    "oil":            "stain",
}


def infer_issue_family_from_text(user_claim: str) -> str:
    """
    Heuristic pre-scan of the claim text to guess the issue family.
    Used only to look up the evidence requirement before the LLM call.
    The LLM will make the definitive issue_type determination.
    Returns the best-guess family string, or "unknown".
    """
    lower = user_claim.lower()
    for keyword, family in CLAIM_TEXT_TO_FAMILY.items():
        if keyword in lower:
            return family
    return "unknown"
