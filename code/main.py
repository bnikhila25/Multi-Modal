"""
main.py
-------
Entry point for the prediction pipeline.
Reads dataset/claims.csv and writes output.csv.

Usage:
    python code/main.py                  # uses Strategy B (default, best performer)
    python code/main.py --strategy A     # uses Strategy A (direct classification)
    python code/main.py --strategy B     # uses Strategy B (chain-of-thought)
    python code/main.py --resume 10      # resume from row 10

Environment:
    GEMINI_API_KEY must be set in shell or .env file at repo root.
    Get your FREE key at: https://aistudio.google.com
"""
import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

from config import CLAIMS_CSV, OUTPUT_CSV, GEMINI_MODEL
from data_loader import (
    load_claims, load_user_history, load_evidence_requirements,
    load_images_for_claim, get_user_risk_summary, get_evidence_requirement,
)
from preprocessor import (
    detect_adversarial_text, compute_user_risk_flags,
    build_user_history_context_text, infer_issue_family_from_text,
)
from llm_agent import analyze_claim, STRATEGIES
from postprocessor import postprocess
from output_writer import append_output_row

# Seconds between API calls — keeps effective rate well under 50 RPM limit
INTER_CALL_DELAY = 2.0


def process_claim(
    row: dict,
    user_history: dict,
    evidence_requirements: list,
    strategy: str = "B",
) -> dict:
    """
    Full processing pipeline for a single claim row.
    Returns a complete output dict ready for CSV writing.
    """
    user_id      = row["user_id"]
    image_paths  = row["image_paths"]
    user_claim   = row["user_claim"]
    claim_object = row["claim_object"]

    print(f"  user={user_id} | object={claim_object} | strategy={strategy}")

    # 1. Adversarial detection (no API call)
    adversarial = detect_adversarial_text(user_claim)
    if adversarial:
        print(f"  ⚠️  Adversarial text detected")

    # 2. User history risk scoring (no API call)
    user_summary   = get_user_risk_summary(user_history, user_id)
    pre_risk_flags = compute_user_risk_flags(user_summary)
    history_text   = build_user_history_context_text(user_summary)

    # 3. Evidence requirement lookup (no API call)
    issue_family      = infer_issue_family_from_text(user_claim)
    evidence_req_text = get_evidence_requirement(
        evidence_requirements, claim_object, issue_family
    ) or ""

    # 4. Load images (no API call)
    images = load_images_for_claim(image_paths)
    loaded = [img for img in images if img["loaded"]]
    failed = [img for img in images if not img["loaded"]]
    if failed:
        print(f"  ⚠️  Missing images: {[img['image_id'] for img in failed]}")

    # 5. LLM analysis — ONE API call
    raw_result = analyze_claim(
        claim_object=claim_object,
        user_claim=user_claim,
        user_history_text=history_text,
        evidence_requirement=evidence_req_text,
        images=loaded,
        pre_risk_flags=pre_risk_flags,
        adversarial_detected=adversarial,
        strategy=strategy,
    )

    # 6. Post-process and validate
    clean = postprocess(
        raw=raw_result,
        claim_object=claim_object,
        pre_risk_flags=pre_risk_flags,
        adversarial_detected=adversarial,
    )

    # 7. Assemble full output row
    output_row = {
        "user_id":      user_id,
        "image_paths":  image_paths,
        "user_claim":   user_claim,
        "claim_object": claim_object,
        **clean,
    }

    icon = {"supported": "✅", "contradicted": "❌",
            "not_enough_information": "❓"}.get(clean["claim_status"], "?")
    print(f"  {icon} {clean['claim_status']} | "
          f"part={clean['object_part']} | severity={clean['severity']}")

    return output_row


def run_pipeline(
    claims_path: str = CLAIMS_CSV,
    output_path: str = OUTPUT_CSV,
    strategy:    str = "B",
    delay:       float = INTER_CALL_DELAY,
    start_from:  int = 0,
    quiet:       bool = False,
) -> list[dict]:
    """
    Main pipeline: loads all data, processes each claim, writes output.csv.
    Rows are written immediately after processing (append mode) so
    an interrupted run is never fully lost.
    """
    if not quiet:
        print("=" * 60)
        print(f"  PREDICTION PIPELINE  [{STRATEGIES[strategy]['name']}]")
        print("=" * 60)

    print(f"\nLoading dataset files...")
    claims       = load_claims(claims_path)
    user_history = load_user_history()
    evidence_req = load_evidence_requirements()
    print(f"  Claims: {len(claims)} | Users: {len(user_history)} | Rules: {len(evidence_req)}")

    # Fresh output file when starting from row 0
    if start_from == 0 and os.path.exists(output_path):
        os.remove(output_path)

    results = []
    total   = len(claims)

    for i, row in enumerate(claims):
        if i < start_from:
            continue

        print(f"\n[{i+1}/{total}]")
        t0 = time.time()

        try:
            output_row = process_claim(row, user_history, evidence_req, strategy)
        except Exception as e:
            print(f"  [ERROR] Row {i+1} failed: {e}")
            output_row = _error_row(row)

        results.append(output_row)
        append_output_row(output_row, output_path)

        print(f"  ⏱  {time.time() - t0:.1f}s")

        if i < total - 1:
            time.sleep(delay)

    print(f"\n{'='*60}")
    print(f"  Done. {len(results)}/{total} rows → {output_path}")
    print(f"{'='*60}\n")
    return results


def _error_row(row: dict) -> dict:
    return {
        "user_id":      row.get("user_id", ""),
        "image_paths":  row.get("image_paths", ""),
        "user_claim":   row.get("user_claim", ""),
        "claim_object": row.get("claim_object", ""),
        "evidence_standard_met":        "false",
        "evidence_standard_met_reason": "Processing error.",
        "risk_flags":                   "none",
        "issue_type":                   "unknown",
        "object_part":                  "unknown",
        "claim_status":                 "not_enough_information",
        "claim_status_justification":   "Automated analysis failed.",
        "supporting_image_ids":         "none",
        "valid_image":                  "false",
        "severity":                     "unknown",
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Modal Evidence Review — Prediction")
    parser.add_argument("--strategy", choices=["A", "B"], default="B",
                        help="Prompting strategy: A=Direct, B=Chain-of-Thought (default: B)")
    parser.add_argument("--resume", type=int, default=0,
                        help="Resume from row N (0-indexed)")
    parser.add_argument("--output", default=OUTPUT_CSV,
                        help="Output CSV path")
    args = parser.parse_args()

    run_pipeline(
        strategy=args.strategy,
        start_from=args.resume,
        output_path=args.output,
    )
