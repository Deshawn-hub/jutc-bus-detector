#!/bin/bash
set -e

if [ -z "${1:-}" ]; then
  echo "Usage: $0 \"STREAM_URL\"" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/camera_2.env"
VENV_ACTIVATE="$SCRIPT_DIR/venv/bin/activate"

cd "$SCRIPT_DIR"

if [ ! -f "$ENV_FILE" ]; then
  echo "Missing env file: $ENV_FILE" >&2
  exit 1
fi

if [ ! -f "$VENV_ACTIVATE" ]; then
  echo "Missing virtual environment activation script: $VENV_ACTIVATE" >&2
  exit 1
fi

source "$VENV_ACTIVATE"

set -a
source "$ENV_FILE"
set +a

export JUTC_STREAM_URL="$1"

python -m jutc_detector.detector_service --mode all
