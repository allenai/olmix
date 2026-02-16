#!/usr/bin/env python3
"""One-time migration script: extract nested weights into explicit top-level `mix` fields.

Walks each LaunchConfig YAML in configs/experiments/, computes flat mix entries
from the product of source/topic/quality weights, removes nested weight fields,
and adds a top-level `mix` section.

Usage:
    python scripts/migrate_configs.py
    python scripts/migrate_configs.py --dry-run   # preview without writing
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def _build_mix(sources: list[dict]) -> dict[str, dict[str, float]]:
    """Walk source tree and build flat mix dict from weight products."""
    mix: dict[str, dict[str, float]] = {}

    for source in sources:
        source_name = source["name"]
        source_weight = source.get("weight", 1.0)
        rep_factor = source.get("max_repetition_factor", 1.0)

        topics = source.get("topics")
        quality = source.get("quality")

        if topics:
            for topic in topics:
                topic_name = topic["name"]
                topic_weight = topic.get("weight", 1.0)
                topic_quality = topic.get("quality")

                if topic_quality:
                    # source:topic:quality — 3 levels
                    for q in topic_quality:
                        q_name = q["name"]
                        q_weight = q.get("weight", 1.0)
                        q_rep = q.get("max_repetition_factor", rep_factor)
                        key = f"{source_name}:{topic_name}:{q_name}"
                        mix[key] = {
                            "weight": source_weight * topic_weight * q_weight,
                            "repetition_factor": q_rep,
                        }
                else:
                    # source:topic — 2 levels
                    t_rep = topic.get("max_repetition_factor", rep_factor)
                    key = f"{source_name}:{topic_name}"
                    mix[key] = {
                        "weight": source_weight * topic_weight,
                        "repetition_factor": t_rep,
                    }
        elif quality:
            # source:quality — 2 levels (no topics)
            for q in quality:
                q_name = q["name"]
                q_weight = q.get("weight", 1.0)
                q_rep = q.get("max_repetition_factor", rep_factor)
                key = f"{source_name}:{q_name}"
                mix[key] = {
                    "weight": source_weight * q_weight,
                    "repetition_factor": q_rep,
                }
        else:
            # source-only (leaf paths)
            key = source_name
            mix[key] = {
                "weight": source_weight,
                "repetition_factor": rep_factor,
            }

    return mix


def _strip_weights(sources: list[dict]) -> None:
    """Remove weight fields from sources/topics/quality in-place."""
    for source in sources:
        source.pop("weight", None)
        for topic in source.get("topics", []):
            topic.pop("weight", None)
            for q in topic.get("quality", []):
                q.pop("weight", None)
        for q in source.get("quality", []):
            q.pop("weight", None)


def migrate_file(path: Path, dry_run: bool = False) -> None:
    """Migrate a single config file."""
    with open(path) as f:
        raw = f.read()

    data = yaml.safe_load(raw)

    # Skip if already has a mix field
    if "mix" in data:
        print(f"  SKIP (already has mix): {path}")
        return

    sources = data.get("data", {}).get("sources", [])
    if not sources:
        print(f"  SKIP (no sources): {path}")
        return

    mix = _build_mix(sources)
    _strip_weights(sources)
    data["mix"] = mix

    if dry_run:
        print(f"  DRY-RUN: {path}")
        for k, v in mix.items():
            print(f"    {k}: weight={v['weight']}, repetition_factor={v['repetition_factor']}")
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

    print(f"  MIGRATED: {path} ({len(mix)} mix entries)")


def main():
    parser = argparse.ArgumentParser(description="Migrate nested weights to explicit mix fields")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    args = parser.parse_args()

    configs_dir = Path("configs/experiments")
    yaml_files = sorted(configs_dir.rglob("*.yaml"))

    print(f"Found {len(yaml_files)} config files in {configs_dir}/")
    for f in yaml_files:
        migrate_file(f, dry_run=args.dry_run)

    print("\nDone!")


if __name__ == "__main__":
    main()
