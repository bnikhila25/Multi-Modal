"""
postprocessor.py
----------------
Takes the raw dict returned by llm_agent.analyze_claim() and:
1. Validates every field against allowed value lists
2. Enforces business logic rules (e.g. user_history_risk → manual_review_required)
3. Merges pre-computed risk flags from preprocessor
4. Coerces types (bool strings → bool, lists → semicolon strings)
5. Returns a clean dict ready to write as a CSV row

No API calls. No I/O. Pure data transformation.
"""
from config import (
    ALLOWED_CLAIM_STATUS, ALLOWED_ISSUE_TYPES, ALLOWED_SEVERITY,
    ALLOWED_RISK_FLAGS, OBJECT_PARTS,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _coerce_bool(val) -> bool:
    """Accept True/False/\"true\"/\"false\"/1/0."""
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() == "true"
    return bool(val)


def _clean_flags(raw_flags, pre_flags: set[str]) -> list[str]:
    """
    Normalize and merge risk flags.
    - raw_flags: list or semicolon string from LLM
    - pre_flags: set from preprocessor (user_history_risk etc.)
    Returns a sorted list of valid flags, or ["none"].
    """
    # Normalise LLM output to a list of strings
    if isinstance(raw_flags, list):
        items = [str(f).strip().lower() for f in raw_flags]
    elif isinstance(raw_flags, str):
        items = [f.strip().lower() for f in raw_flags.split(";")]
    else:
        items = []

    # Merge pre-computed flags
    all_items = set(items) | {f.lower() for f in pre_flags}

    # Enforce business rule: user_history_risk → manual_review_required
    if "user_history_risk" in all_items:
        all_items.add("manual_review_required")

    # Keep only valid flags
    valid = {f for f in all_items if f in ALLOWED_RISK_FLAGS}

    # Remove "none" if other flags are present
    if len(valid) > 1:
        valid.discard("none")

    if not valid:
        valid = {"none"}

    return sorted(valid)


def _clean_supporting_ids(raw) -> list[str]:
    """
    Normalise supporting_image_ids to a list of clean ID strings.
    """
    if isinstance(raw, list):
        items = [str(i).strip() for i in raw if str(i).strip()]
    elif isinstance(raw, str):
        items = [i.strip() for i in raw.split(";") if i.strip()]
    else:
        items = []

    if not items or items == ["none"]:
        return ["none"]
    return items


def _safe_str(val, allowed: set, default: str) -> str:
    """Return val if it is in allowed, else default."""
    v = str(val).strip().lower() if val else ""
    return v if v in allowed else default


def _validate_object_part(part: str, claim_object: str) -> str:
    """Ensure part is valid for this claim_object."""
    allowed = OBJECT_PARTS.get(claim_object.lower(), set())
    p = str(part).strip().lower() if part else ""
    return p if p in allowed else "unknown"


# ---------------------------------------------------------------------------
# Main post-processing function
# ---------------------------------------------------------------------------

def postprocess(
    raw: dict,
    claim_object: str,
    pre_risk_flags: set[str],
    adversarial_detected: bool,
) -> dict:
    """
    Validate and clean a raw LLM output dict.

    Parameters
    ----------
    raw              : dict returned by llm_agent.analyze_claim()
    claim_object     : "car", "laptop", or "package"
    pre_risk_flags   : flags already computed by preprocessor
    adversarial_detected : True if adversarial text was found in claim

    Returns
    -------
    A clean dict with all output fields as final values.
    """

    # --- Boolean fields ---
    evidence_standard_met = _coerce_bool(raw.get("evidence_standard_met", False))
    valid_image           = _coerce_bool(raw.get("valid_image", True))

    # --- Categorical fields ---
    claim_status = _safe_str(
        raw.get("claim_status"),
        ALLOWED_CLAIM_STATUS,
        "not_enough_information",
    )
    issue_type = _safe_str(
        raw.get("issue_type"),
        ALLOWED_ISSUE_TYPES,
        "unknown",
    )
    object_part = _validate_object_part(
        raw.get("object_part", "unknown"),
        claim_object,
    )
    severity = _safe_str(
        raw.get("severity"),
        ALLOWED_SEVERITY,
        "unknown",
    )

    # --- Risk flags (merge LLM + pre-computed) ---
    extra_flags = set(pre_risk_flags)
    if adversarial_detected:
        extra_flags.add("text_instruction_present")
        extra_flags.add("manual_review_required")

    risk_flags_list = _clean_flags(raw.get("risk_flags", []), extra_flags)

    # --- Business logic enforcement ---

    # Rule 1: evidence_standard_met=False → claim_status must be not_enough_information
    if not evidence_standard_met:
        claim_status = "not_enough_information"
        severity     = "unknown"

    # Rule 2: not_enough_information → severity must be unknown
    if claim_status == "not_enough_information":
        severity = "unknown"

    # Rule 3: contradicted with issue_type=none → severity=none
    if claim_status == "contradicted" and issue_type == "none":
        severity = "none"

    # Rule 4: not_enough_information → supporting_image_ids must be none
    supporting_ids = _clean_supporting_ids(raw.get("supporting_image_ids", ["none"]))
    if claim_status == "not_enough_information":
        supporting_ids = ["none"]

    # --- Text fields (pass through, strip whitespace) ---
    evidence_standard_met_reason = str(
        raw.get("evidence_standard_met_reason", "")
    ).strip()
    claim_status_justification = str(
        raw.get("claim_status_justification", "")
    ).strip()

    # --- Serialise list fields to semicolon strings ---
    risk_flags_str       = ";".join(risk_flags_list)
    supporting_ids_str   = ";".join(supporting_ids)

    return {
        "evidence_standard_met":        str(evidence_standard_met).lower(),
        "evidence_standard_met_reason": evidence_standard_met_reason,
        "risk_flags":                   risk_flags_str,
        "issue_type":                   issue_type,
        "object_part":                  object_part,
        "claim_status":                 claim_status,
        "claim_status_justification":   claim_status_justification,
        "supporting_image_ids":         supporting_ids_str,
        "valid_image":                  str(valid_image).lower(),
        "severity":                     severity,
    }
