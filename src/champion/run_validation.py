"""Run the saved development composite with the unchanged evaluator."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kit", type=Path, default=Path("research_kit"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    kit = args.kit.resolve()
    out = args.output.resolve()
    if (out / "validation_report.json").exists():
        parser.error("Choose a fresh output directory.")
    os.environ["MADINAH_AIDE_AUDIT_DIR"] = str(kit / "audit")
    sys.path.insert(0, str(kit / "input"))
    from benchmark import evaluate
    from candidate import predict_fold

    start = time.monotonic()
    report = evaluate(predict_fold, check_causality=True, output_dir=out)
    root = Path(__file__).resolve().parent
    metadata = {
        "elapsed_seconds": time.monotonic() - start,
        "python_version": platform.python_version(),
        "dependencies": {name: importlib.metadata.version(name) for name in
                         ("numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl")},
        "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(root.rglob("*.py"))},
        "target": .400,
        "target_met_on_local_protocol": report["local_surrogate_score"] <= .400,
        "official_kaggle_score": None,
        "submission_created": False,
        "submission_uploaded": False,
    }
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"ELAPSED_SECONDS={metadata['elapsed_seconds']:.3f}", flush=True)


if __name__ == "__main__":
    main()
