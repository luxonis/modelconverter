#!/usr/bin/env bash

set -e  # Exit immediately if a command fails

# Check if required arguments were provided
if [ -z "${1:-}" ] || [ -z "${2:-}" ] || [ -z "${3:-}" ]; then
  echo "Usage: $0 <HUBAI_API_KEY> <PAT_TOKEN> <DAI_VERSION>"
  exit 1
fi

# Export variables from input arguments
export HUBAI_API_KEY="$1"
export PAT_TOKEN="$2"
export DEPTHAI_VERSION="$3"

# Navigate to project directory
cd /tmp/modelconverter

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install pytest

pip install hil-framework --upgrade \
  --index-url "https://__token__:$PAT_TOKEN@gitlab.luxonis.com/api/v4/projects/213/packages/pypi/simple" \
  > /dev/null

pip install --upgrade \
  --extra-index-url "https://artifacts.luxonis.com/artifactory/luxonis-python-snapshot-local/" \
  --extra-index-url https://artifacts.luxonis.com/artifactory/luxonis-python-release-local \
  "depthai==${DEPTHAI_VERSION}"

# Cache device metadata once for the whole run using the HIL camera CLI. If the
# lookup fails, keep the benchmark runnable and record explicit placeholder
# values for the missing camera-derived metadata.
camera_output=$(
  camera -t "${HIL_TESTBED}" -n test all info -j 2>/dev/null || printf ''
)

if [ -z "$camera_output" ]; then
  echo "Error: failed to obtain camera metadata via camera info." >&2
  exit 1
fi

device_hostname=$(
  printf '%s' "$camera_output" \
    | jq -r '.[0].hostname // empty' 2>/dev/null \
    | head -n1
)
camera_mxid=$(
  printf '%s' "$camera_output" \
    | jq -r '.[0].mxid // empty' 2>/dev/null \
    | head -n1
)
camera_model=$(
  printf '%s' "$camera_output" \
    | jq -r '.[0].model // empty' 2>/dev/null \
    | head -n1
)
camera_revision=$(
  printf '%s' "$camera_output" \
    | jq -r '.[0].revision // empty' 2>/dev/null \
    | head -n1
)
camera_os=$(
  printf '%s' "$camera_output" \
    | jq -r '.[0].os_version // empty' 2>/dev/null \
    | head -n1
)
detected_testbed_name=$(
  printf '%s' "$camera_output" \
    | jq -r '.[0].name // empty' 2>/dev/null \
    | head -n1
)

missing_metadata=()
if [ -z "$device_hostname" ]; then
  missing_metadata+=("hostname")
fi
if [ -z "$camera_mxid" ]; then
  missing_metadata+=("mxid")
fi
if [ -z "$camera_model" ]; then
  missing_metadata+=("model")
fi
if [ -z "$camera_revision" ]; then
  missing_metadata+=("revision")
fi
if [ -z "$camera_os" ]; then
  missing_metadata+=("os_version")
fi

if [ "${#missing_metadata[@]}" -ne 0 ]; then
  echo "Error: camera metadata is incomplete; missing fields: ${missing_metadata[*]}" >&2
  exit 1
fi

runner_hostname=$(hostname 2>/dev/null || printf 'unknown')
server_os=$(uname -s 2>/dev/null | tr '[:upper:]' '[:lower:]' || printf 'unknown')
if [ -z "$runner_hostname" ]; then
  runner_hostname="unknown"
fi
if [ -z "$server_os" ]; then
  server_os="unknown"
fi
if [ -z "$HIL_TESTBED" ]; then
  HIL_TESTBED="${detected_testbed_name:-}"
fi
if [ -z "$HIL_TESTBED" ]; then
  HIL_TESTBED="$(hostname 2>/dev/null || printf '')"
fi

export HIL_TESTBED
export HIL_CAMERA_MXID="$camera_mxid"
export HIL_CAMERA_OS_VERSION="$camera_os"
export HIL_CAMERA_MODEL="$camera_model"
export HIL_CAMERA_REVISION="$camera_revision"
export HIL_SERVER_OS="$server_os"

# Run tests
pytest_args=(
  -s
  -v
  tests/test_benchmark/
)

pytest_args+=(--device-ip "$device_hostname")

echo "Influx metadata debug:"
echo "  INFLUX_HOST=${INFLUX_HOST:-<unset>}"
echo "  INFLUX_ORG=${INFLUX_ORG:-<unset>}"
echo "  INFLUX_BUCKET=${INFLUX_BUCKET:-<unset>}"
echo "  INFLUX_TOKEN=$(if [ -n "${INFLUX_TOKEN:-}" ]; then printf '<set>'; else printf '<unset>'; fi)"
echo "  DEPTHAI_VERSION=${DEPTHAI_VERSION:-<empty>}"
echo "  HIL_TESTBED=${HIL_TESTBED:-<empty>}"
echo "  HIL_CAMERA_MXID=${HIL_CAMERA_MXID:-<empty>}"
echo "  HIL_CAMERA_OS_VERSION=${HIL_CAMERA_OS_VERSION:-<empty>}"
echo "  HIL_CAMERA_MODEL=${HIL_CAMERA_MODEL:-<empty>}"
echo "  HIL_CAMERA_REVISION=${HIL_CAMERA_REVISION:-<empty>}"
echo "  HIL_SERVER_OS=${HIL_SERVER_OS:-<empty>}"
echo "  device_ip=${device_hostname:-<empty>}"
echo "  camera_mxid=${camera_mxid:-<empty>}"
echo "  camera_os_version=${camera_os:-<empty>}"
echo "  camera_model=${camera_model:-<empty>}"
echo "  camera_revision=${camera_revision:-<empty>}"
echo "  runner=${runner_hostname:-<empty>}"
echo "  server_os=${server_os:-<empty>}"
printf '  pytest_args:'
printf ' %q' "${pytest_args[@]}"
printf '\n'

pytest "${pytest_args[@]}"
