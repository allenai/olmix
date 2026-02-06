#!/bin/bash
# Batch launch all quality upsampling experiments (nested directories)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for config in "$SCRIPT_DIR"/**/*.yaml; do
    echo "Launching: $config"
    yes | olmix launch run --config "$config" --no-cache
done
