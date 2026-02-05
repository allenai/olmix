#!/bin/bash
# Batch launch all data proportion experiments
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for config in "$SCRIPT_DIR"/*.yaml; do
    echo "Launching: $config"
    olmix launch run --config "$config"
done
