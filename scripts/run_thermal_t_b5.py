#!/usr/bin/env python3
"""Run the Thermal T-B5 readiness or frozen offline experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.thermal.t_b5_runner import FULL_MODE, READINESS_MODE, run


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SafeNest Thermal T-B5")
    parser.add_argument("--mode", choices=(READINESS_MODE, FULL_MODE), default=READINESS_MODE)
    parser.add_argument("--repo-root", default=str(ROOT))
    parser.add_argument("--canonical-root", required=True, help="External SSD canonical root (for example .../thermal/canonical)")
    parser.add_argument("--work-root", required=True, help="External SSD scratch root")
    parser.add_argument("--output-root", required=True, help="External SSD T-B5 evidence output root")
    parser.add_argument("--execute", action="store_true", help="Execute FULL_EXPERIMENT; without it only report readiness")
    args = parser.parse_args()
    result = run(mode=args.mode, repo_root=Path(args.repo_root), canonical_root=Path(args.canonical_root), work_root=Path(args.work_root), output_root=Path(args.output_root), execute=args.execute)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
