"""
output_writer.py
----------------
Writes the final output.csv in the exact column order required.
"""
import csv
import os
from config import OUTPUT_COLUMNS, OUTPUT_CSV


def write_output(rows: list[dict], output_path: str = OUTPUT_CSV) -> None:
    """
    Write a list of fully-processed row dicts to output.csv.
    Each dict must contain all keys from OUTPUT_COLUMNS plus the
    passthrough input fields (user_id, image_paths, user_claim, claim_object).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=OUTPUT_COLUMNS,
            quoting=csv.QUOTE_ALL,
            extrasaction="ignore",   # silently drop any extra keys
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in OUTPUT_COLUMNS})

    print(f"\n✅  Output written → {output_path}  ({len(rows)} rows)")


def append_output_row(row: dict, output_path: str = OUTPUT_CSV) -> None:
    """
    Append a single row to output.csv.
    Used for streaming writes so progress is not lost if the run is interrupted.
    """
    file_exists = os.path.isfile(output_path)
    with open(output_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=OUTPUT_COLUMNS,
            quoting=csv.QUOTE_ALL,
            extrasaction="ignore",
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow({col: row.get(col, "") for col in OUTPUT_COLUMNS})
