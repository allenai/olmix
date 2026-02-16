#!/usr/bin/env python3
"""Normalize mix weights in migrated configs so they sum to 1.0.

Usage:
    python scripts/normalize_mix_weights.py
    python scripts/normalize_mix_weights.py --dry-run
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def normalize_file(path: Path, dry_run: bool = False) -> None:
    """Normalize mix weights in a single config file."""
    with open(path) as f:
        raw = f.read()

    data = yaml.safe_load(raw)
    mix = data.get("mix")
    if not mix:
        print(f"  SKIP (no mix): {path}")
        return

    total = sum(entry["weight"] for entry in mix.values())
    if total == 0:
        print(f"  SKIP (all weights zero): {path}")
        return

    # Check if already normalized
    if abs(total - 1.0) < 1e-9:
        print(f"  SKIP (already normalized): {path}")
        return

    for entry in mix.values():
        entry["weight"] = round(entry["weight"] / total, 6)

    if dry_run:
        print(f"  DRY-RUN: {path} (total was {total})")
        for k, v in mix.items():
            print(f"    {k}: weight={v['weight']}")
        return

    # Preserve comment header
    header_lines = []
    for line in raw.splitlines():
        if line.startswith("#"):
            header_lines.append(line)
        else:
            break
    header = "\n".join(header_lines) + "\n\n" if header_lines else ""

    body = yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True)
    with open(path, "w") as f:
        f.write(header + body)

    print(f"  NORMALIZED: {path} (total was {total})")


def main():
    parser = argparse.ArgumentParser(description="Normalize mix weights to sum to 1.0")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    args = parser.parse_args()

    configs_dir = Path("configs/experiments")
    yaml_files = sorted(configs_dir.rglob("*.yaml"))

    print(f"Found {len(yaml_files)} config files in {configs_dir}/")
    for f in yaml_files:
        normalize_file(f, dry_run=args.dry_run)

    print("\nDone!")


if __name__ == "__main__":
    main()
