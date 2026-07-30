#!/usr/bin/env bash
set -euo pipefail

# WFD-2020 Wheat Disease Classification — Pipeline Launcher
# Usage:
#   ./run_pipeline.sh                          # download + run all
#   ./run_pipeline.sh efficientnet_b4          # run a specific experiment
#   ./run_pipeline.sh --download               # download data only
#   ./run_pipeline.sh --no-download --all       # run all without downloading
#   ./run_pipeline.sh --seed 42                # override seed

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

DOWNLOAD=true

POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-download)
            DOWNLOAD=false
            shift
            ;;
        --download)
            DOWNLOAD=true
            shift
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done

set -- "${POSITIONAL[@]}"

if [ $# -eq 0 ]; then
    if [ "$DOWNLOAD" = true ]; then
        echo "Downloading dataset and running all experiments..."
        python pipeline.py --download --all
    else
        echo "Running all experiments..."
        python pipeline.py --all
    fi
else
    if [ "$DOWNLOAD" = true ]; then
        python pipeline.py --download --experiments "$@"
    else
        python pipeline.py --experiments "$@"
    fi
fi

echo "Pipeline completed successfully."
