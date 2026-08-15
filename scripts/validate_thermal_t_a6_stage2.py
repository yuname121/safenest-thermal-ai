#!/usr/bin/env python3
"""Standalone compact-evidence validator for Thermal T-A6 Stage 2.

The validator never opens raw ZIP payloads or canonical bulk tensors.  It
checks the deterministic JSON/checksum bundle emitted by the Colab runner and
fails closed when a required audit, role, path, or downstream-gate contract is
missing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.thermal.t_a6_stage2 import validate_stage2_bundle  # noqa: E402


def validate_evidence(bundle_dir: Path) -> dict[str, object]:
    return validate_stage2_bundle(Path(bundle_dir), require_validation_result=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a compact Thermal T-A6 Stage 2 result bundle")
    parser.add_argument("--bundle-dir", type=Path, required=True)
    args = parser.parse_args()
    result = validate_evidence(args.bundle_dir)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    raise SystemExit(0 if result.get("evidence_validation") == "PASS" else 1)


if __name__ == "__main__":
    main()
