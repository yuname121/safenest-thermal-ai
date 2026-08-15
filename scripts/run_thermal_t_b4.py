#!/usr/bin/env python3
"""CLI for the SafeNest Thermal T-B4 conversion/equivalence runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.thermal.t_b4_runner import CORRECTION_MODE, FULL_MODE, READINESS_MODE, run  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Thermal T-B4 frozen Float/TFLite/INT8 equivalence audit")
    parser.add_argument("--mode", choices=(READINESS_MODE, FULL_MODE, CORRECTION_MODE), default=READINESS_MODE)
    parser.add_argument("--canonical-root", required=True)
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--repo-root", default=str(ROOT))
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--owner-authorized", action="store_true")
    args = parser.parse_args()
    try:
        result = run(mode=args.mode, canonical_root=args.canonical_root, work_root=args.work_root, output_root=args.output_root, repo_root=args.repo_root, execute=args.execute, owner_authorized=args.owner_authorized, checkpoint_path=args.checkpoint_path)
    except Exception as exc:
        print(json.dumps({"phase": "T-B4", "status": "BLOCKED", "error": str(exc)}, ensure_ascii=False, indent=2, sort_keys=True))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
