#!/usr/bin/env bash
set -euo pipefail

# WFD-2020 Wheat Disease Classification — Pipeline Launcher
# Usage:
#   ./run_pipeline.sh                          # run all experiments
#   ./run_pipeline.sh efficientnet_b4          # run a specific experiment
#   ./run_pipeline.sh efficientnet_b4 convnext_small  # run multiple
#   ./run_pipeline.sh --seed 42                # override seed

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Default behaviour: run all if no args given
if [ $# -eq 0 ]; then
    echo "Running all experiments..."
    python pipeline.py --all
else
    python pipeline.py --experiments "$@"
fi

echo "Pipeline completed successfully."
