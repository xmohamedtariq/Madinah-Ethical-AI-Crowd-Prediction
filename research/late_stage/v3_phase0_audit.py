#!/usr/bin/env python3
"""V3 phase-zero read-only audit of the Madinah-AI-Fresh project.

Reads existing reports and source files without changing them, and writes only
reports/v3_phase0/schema_audit.json. No model, predictions, or submission created.

Usage (from project root):
    python v3_phase0_audit.py --root .
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

EXPECTED = {
    "decision": "reports/breakthrough_v2/decision.md",
    "missing_targets": "reports/breakthrough_v2/missing_target_diagnostics.csv",
    "missing_groups": "reports/breakthrough_v2/missing_error_groups.csv",
    "task_comparisons": "reports/breakthrough_v2/per_task_score_comparisons.csv",
    "recording_results": "reports/breakthrough_v2/per_recording_results.csv",
    "validation_integrity": "reports/breakthrough_v2/validation_integrity.json",
    "missing_state_space_source": "src/final_missing_hidden_research/missing_state_space.py",
    "hidden_profile_transfer_source": "src/final_missing_hidden_research/hidden_profile_transfer.py",
}

REC_KEYS = ("recording", "validation_recording", "source_recording", "recording_id", "rec")
CAM_KEYS = ("unit", "camera", "cam", "camera_id", "target_camera")
TIME_KEYS = ("minute_of_day", "minute", "timestamp", "time", "id")
GROUP_HINTS = ("regime", "pattern", "trend", "error_group", "falling", "gap_type", "shape")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def existing_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    lookup = {c.lower(): c for c in columns}
    return next((lookup[c] for c in candidates if c in lookup), None)


def csv_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        columns = list(reader.fieldnames or [])
        if not columns:
            raise ValueError("CSV has no header")
        n = 0
        rec = Counter()
        cam = Counter()
        groups = {}
        selected = {k: existing_column(columns, v) for k, v in {
            "recording": REC_KEYS, "camera": CAM_KEYS, "time": TIME_KEYS,
        }.items()}
        group_cols = [c for c in columns if any(h in c.lower() for h in GROUP_HINTS)][:6]
        for col in group_cols:
            groups[col] = Counter()
        nulls = Counter()
        key_seen = set()
        duplicate_keys = 0
        for row in reader:
            n += 1
            for col, val in row.items():
                if val is None or val == "" or str(val).lower() in ("nan", "null", "none"):
                    nulls[col] += 1
            if selected["recording"]:
                rec[row[selected["recording"]]] += 1
            if selected["camera"]:
                cam[row[selected["camera"]]] += 1
            for col in group_cols:
                groups[col][str(row.get(col, ""))[:80]] += 1
            if all(selected.values()):
                key = tuple(row[k] for k in selected.values())
                if key in key_seen:
                    duplicate_keys += 1
                key_seen.add(key)
        return {
            "rows": n,
            "columns": columns,
            "key_columns": selected,
            "recording_counts": dict(rec.most_common(20)),
            "camera_counts": dict(cam.most_common(25)),
            "group_counts": {k: dict(v.most_common(20)) for k, v in groups.items()},
            "null_counts_nonzero": {k: v for k, v in nulls.items() if v},
            "duplicate_key_count": duplicate_keys if all(selected.values()) else None,
        }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path.cwd(), help="Path to Madinah-AI-Fresh")
    args = p.parse_args()
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"ERROR: root does not exist: {root}")

    audit = {
        "tool": "V3_PHASE0_AUDIT_ONLY",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "reported_champion_local_score_not_replayed": 0.6057000521183546,
        "target_score_strictly_below": 0.6,
        "expected_missing_frozen_rows": 358,
        "expected_fast_falling_rows_from_report": 147,
        "writes_submission": False,
        "files": {},
        "warnings": [],
    }
    for label, rel in EXPECTED.items():
        path = root / rel
        d = {"relative_path": rel, "exists": path.is_file()}
        if path.is_file():
            d["size_bytes"] = path.stat().st_size
            d["sha256"] = digest(path)
            try:
                if path.suffix.lower() == ".csv":
                    d["schema"] = csv_summary(path)
                elif path.suffix.lower() == ".json":
                    with path.open("r", encoding="utf-8-sig") as f:
                        j = json.load(f)
                    d["top_level_keys"] = sorted(j.keys()) if isinstance(j, dict) else ["non_object_json"]
            except Exception as exc:
                d["read_error"] = f"{type(exc).__name__}: {exc}"
                audit["warnings"].append(f"Unable to inspect {rel}: {exc}")
        else:
            audit["warnings"].append(f"Missing expected file: {rel}")
        audit["files"][label] = d

    target = audit["files"]["missing_targets"].get("schema", {})
    if target.get("rows") not in (358, None):
        audit["warnings"].append(f"Target count unexpected: {target.get('rows')} instead of 358")
    if target.get("duplicate_key_count", 0) not in (0, None):
        audit["warnings"].append("Duplicate target keys detected; inspect before experiments")

    script_candidates = []
    src = root / "src"
    if src.exists():
        for path in src.rglob("*.py"):
            name = path.name.lower()
            if any(s in name for s in ("eval", "scor", "frozen", "champion", "missing")):
                script_candidates.append(str(path.relative_to(root)))
    audit["possible_evaluator_or_missing_scripts"] = sorted(script_candidates)[:70]

    output = root / "reports" / "v3_phase0" / "schema_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print("V3 PHASE-0: READ-ONLY SOURCE AUDIT")
    for name, result in audit["files"].items():
        count = result.get("schema", {}).get("rows")
        print(f"  {name:30s}: {'OK' if result['exists'] else 'MISSING'}" + (f"; {count} rows" if count is not None else ""))
    print(f"Missing diagnostic rows: {target.get('rows', 'UNKNOWN')}; expected 358")
    print(f"Validation/frozen score: NOT recomputed by this audit")
    print(f"Warnings: {len(audit['warnings'])}")
    for warning in audit["warnings"]:
        print("  WARNING:", warning)
    print("Schema audit:", output)
    print("NO predictions, submission, or model files created or changed.")


if __name__ == "__main__":
    main()
