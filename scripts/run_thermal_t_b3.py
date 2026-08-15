#!/usr/bin/env python3
"""Run the SafeNest Thermal T-B3 frame-only multi-seed confirmation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.thermal.t_b3_runner import FULL_MODE, READINESS_MODE, RunnerContractError, run  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="SafeNest Thermal T-B3 frame-only multi-seed stability confirmation")
    parser.add_argument("--canonical-root", required=True, help="External SSD T-A6 canonical root")
    parser.add_argument("--work-root", required=True, help="Mac-local scratch directory")
    parser.add_argument("--output-root", required=True, help="External SSD T-B3 output parent")
    parser.add_argument("--repo-root", default=str(ROOT))
    parser.add_argument("--mode", choices=(READINESS_MODE, FULL_MODE), default=READINESS_MODE)
    parser.add_argument("--execute", action="store_true", help="Write readiness or execute the owner-authorized experiment")
    parser.add_argument("--authorize-full-experiment", action="store_true", help="Explicit owner authorization for new seed training")
    args = parser.parse_args()
    try:
        result = run(
            mode=args.mode,
            canonical_root=args.canonical_root,
            work_root=args.work_root,
            output_root=args.output_root,
            repo_root=args.repo_root,
            execute=bool(args.execute),
            owner_authorized=bool(args.authorize_full_experiment),
        )
    except (RunnerContractError, OSError, ValueError) as exc:
        print(json.dumps({"phase": "T-B3", "status": "BLOCKED", "error": str(exc)}, ensure_ascii=False, indent=2, sort_keys=True))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
