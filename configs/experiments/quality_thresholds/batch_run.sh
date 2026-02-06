#!/bin/bash
# Batch launch all quality threshold experiments (nested directories)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for config in "$SCRIPT_DIR"/**/*.yaml; do
    echo "Launching: $config"
    olmix launch run --config "$config"
done
