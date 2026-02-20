#!/usr/bin/env python3
"""Convert flat mix entries to nested format in all experiment configs.

Usage:
    python scripts/nest_mix_weights.py
    python scripts/nest_mix_weights.py --dry-run
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def _is_leaf(value: object) -> bool:
    """Check if a dict is a leaf MixEntry (only weight/repetition_factor keys)."""
    return isinstance(value, dict) and set(value.keys()) <= {"weight", "repetition_factor"}


def _sum_leaf_weights(node: dict) -> float:
    """Sum all leaf weights under a subtree."""
    if _is_leaf(node):
        return node.get("weight", 0.0)
    return sum(_sum_leaf_weights(v) for v in node.values() if isinstance(v, dict))


def _collect_reps(node: dict) -> set[float]:
    """Collect all repetition_factor values from leaves."""
    if _is_leaf(node):
        return {node.get("repetition_factor", 1.0)}
    reps: set[float] = set()
    for v in node.values():
        if isinstance(v, dict):
            reps |= _collect_reps(v)
    return reps


def _convert_node(node: dict, parent_total: float | None = None) -> dict:
    """Convert a tree node to nested format with relative weights."""
    total = _sum_leaf_weights(node) if parent_total is None else parent_total
    result: dict = {}

    for key, value in node.items():
        if _is_leaf(value):
            rel = value["weight"] / total if total > 0 else 0.0
            entry: dict = {"weight": round(rel, 6)}
            rep = value.get("repetition_factor", 1.0)
            if rep != 1.0:
                entry["repetition_factor"] = rep
            result[key] = entry
        elif isinstance(value, dict):
            subtotal = _sum_leaf_weights(value)
            rel = subtotal / total if total > 0 else 0.0
            child = _convert_node(value, subtotal)

            nested: dict = {"weight": round(rel, 6)}

            # Hoist common repetition_factor to this level
            reps = _collect_reps(value)
            if len(reps) == 1:
                common_rep = reps.pop()
                if common_rep != 1.0:
                    nested["repetition_factor"] = common_rep
                    for cv in child.values():
                        if isinstance(cv, dict):
                            cv.pop("repetition_factor", None)

            nested.update(child)
            result[key] = nested

    return result


def flat_to_nested(flat_mix: dict) -> dict:
    """Convert flat colon-separated mix keys to a nested dict."""
    # Build tree from flat keys
    tree: dict = {}
    for key, entry in flat_mix.items():
        parts = key.split(":")
        node = tree
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = dict(entry)

    # Check if nesting actually helps (all keys are single-component → no nesting needed)
    has_nesting = any(":" in k for k in flat_mix)
    if not has_nesting:
        # No colons → all top-level leaves, return as-is
        return flat_mix

    return _convert_node(tree)


def convert_file(path: Path, dry_run: bool = False) -> None:
    """Convert a single config's mix from flat to nested format."""
    with open(path) as f:
        raw = f.read()

    data = yaml.safe_load(raw)
    mix = data.get("mix")
    if not mix:
        print(f"  SKIP (no mix): {path}")
        return

    # Check if already nested (any value has non-property keys)
    already_nested = False
    for v in mix.values():
        if isinstance(v, dict):
            non_props = set(v.keys()) - {"weight", "repetition_factor"}
            if non_props:
                already_nested = True
                break

    if already_nested:
        print(f"  SKIP (already nested): {path}")
        return

    nested = flat_to_nested(mix)
    data["mix"] = nested

    if dry_run:
        print(f"  DRY-RUN: {path}")
        print(yaml.dump({"mix": nested}, default_flow_style=False, sort_keys=False))
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

    print(f"  NESTED: {path}")


def main():
    parser = argparse.ArgumentParser(description="Convert flat mix to nested format")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    args = parser.parse_args()

    configs_dir = Path("config/examples/launch")
    yaml_files = sorted(configs_dir.rglob("*.yaml"))

    print(f"Found {len(yaml_files)} config files in {configs_dir}/")
    for f in yaml_files:
        convert_file(f, dry_run=args.dry_run)

    print("\nDone!")


if __name__ == "__main__":
    main()
