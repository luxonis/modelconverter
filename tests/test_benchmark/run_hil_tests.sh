#!/usr/bin/env bash

set -e  # Exit immediately if a command fails

# Check if required arguments were provided
if [ -z "${1:-}" ] || [ -z "${2:-}" ] || [ -z "${3:-}" ] || [ -z "${4:-}" ]; then
  echo "Usage: $0 <HUBAI_API_KEY> <PAT_TOKEN> <DAI_VERSION> <INFLUX_TOKEN> [BENCHMARK_RUN_ID]"
  exit 1
fi

# Export variables from input arguments
export HUBAI_API_KEY="$1"
export PAT_TOKEN="$2"
export DEPTHAI_VERSION="$3"
export INFLUX_TOKEN="$4"
BENCHMARK_RUN_ID="${5:-}"

# Navigate to project directory
cd /tmp/modelconverter

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install pytest

git clone --recurse-submodules -b tjb_influx_pusher https://oauth2:$PAT_TOKEN@gitlab.luxonis.com/luxonis/hil_lab/hil_framework.git
pip install ./hil_framework/

rm -rf hil_framework

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

rvc4_camera=$(
  printf '%s' "$camera_output" \
    | jq -r '.[] | select(.platform == "rvc4") | @json' 2>/dev/null \
    | head -n1
)

if [ -z "$rvc4_camera" ]; then
  echo "Error: no rvc4 camera found in camera metadata." >&2
  exit 1
fi

device_hostname=$(
  printf '%s' "$rvc4_camera" \
    | jq -r '.hostname // empty' 2>/dev/null \
    | head -n1
)
camera_mxid=$(
  printf '%s' "$rvc4_camera" \
    | jq -r '.mxid // empty' 2>/dev/null \
    | head -n1
)
camera_model=$(
  printf '%s' "$rvc4_camera" \
    | jq -r '.model // empty' 2>/dev/null \
    | head -n1
)
camera_revision=$(
  printf '%s' "$rvc4_camera" \
    | jq -r '.revision // empty' 2>/dev/null \
    | head -n1
)
camera_os=$(
  printf '%s' "$rvc4_camera" \
    | jq -r '.os_version // empty' 2>/dev/null \
    | head -n1
)
detected_testbed_name=$(
  printf '%s' "$rvc4_camera" \
    | jq -r '.name // empty' 2>/dev/null \
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
testbed_name="${HIL_TESTBED:-}"
if [ -z "$testbed_name" ]; then
  testbed_name="${detected_testbed_name:-}"
fi
if [ -z "$testbed_name" ]; then
  testbed_name="$(hostname 2>/dev/null || printf '')"
fi

# Run tests
pytest_args=(
  -s
  -v
  tests/test_benchmark/
  --depthai-version "$DEPTHAI_VERSION"
  --testbed-name "$testbed_name"
  --camera-mxid "$camera_mxid"
  --camera-os-version "$camera_os"
  --camera-model "$camera_model"
  --camera-revision "$camera_revision"
  --server-os "$server_os"
)

pytest_args+=(--device-ip "$device_hostname")
if [ -n "$BENCHMARK_RUN_ID" ]; then
  pytest_args+=(--benchmark-run-id "$BENCHMARK_RUN_ID")
fi

echo "Influx metadata debug:"
echo "  INFLUX_BUCKET=fps_metrics"
echo "  INFLUX_TOKEN=$(if [ -n "${INFLUX_TOKEN:-}" ]; then printf '<set>'; else printf '<empty>'; fi)"
echo "  DEPTHAI_VERSION=${DEPTHAI_VERSION:-<empty>}"
echo "  benchmark_run_id=${BENCHMARK_RUN_ID:-<generated>}"
echo "  HIL_TESTBED=${testbed_name:-<empty>}"
echo "  HIL_CAMERA_MXID=${camera_mxid:-<empty>}"
echo "  HIL_CAMERA_OS_VERSION=${camera_os:-<empty>}"
echo "  HIL_CAMERA_MODEL=${camera_model:-<empty>}"
echo "  HIL_CAMERA_REVISION=${camera_revision:-<empty>}"
echo "  HIL_SERVER_OS=${server_os:-<empty>}"
echo "  device_ip=${device_hostname:-<empty>}"
echo "  runner=${runner_hostname:-<empty>}"
echo "  server_os=${server_os:-<empty>}"
printf '  pytest_args:'
printf ' %q' "${pytest_args[@]}"
printf '\n'

pytest "${pytest_args[@]}"
