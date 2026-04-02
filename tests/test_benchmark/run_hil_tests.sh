#!/usr/bin/env bash

set -e  # Exit immediately if a command fails

# Check if required arguments were provided
if [ -z "${1:-}" ] || [ -z "${2:-}" ] || [ -z "${3:-}" ]; then
  echo "Usage: $0 <HUBAI_API_KEY> <PAT_TOKEN> <DAI_VERSION> [TESTBED_NAME]"
  exit 1
fi

# Export variables from input arguments
export HUBAI_API_KEY="$1"
export PAT_TOKEN="$2"
export DEPTHAI_VERSION="$3"
export HIL_TESTBED_NAME="${4:-}"

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

# Cache device metadata once for the whole run using oakctl. If oakctl is not
# available, keep the benchmark runnable but disable Influx push for this run.
oakctl_output=$(oakctl list --format json 2>/dev/null || printf '')

if [ -z "$oakctl_output" ]; then
  echo "Warning: best-effort metadata lookup via oakctl failed; disabling Influx push for this run." >&2
  unset INFLUX_HOST INFLUX_ORG INFLUX_BUCKET INFLUX_TOKEN
fi

device_hostname=$(
  printf '%s' "$oakctl_output" \
    | jq -r '.items[0].ip_addresses[0] // empty' 2>/dev/null \
    | head -n1
)
camera_mxid=$(
  printf '%s' "$oakctl_output" \
    | jq -r '.items[0].serial_number // empty' 2>/dev/null \
    | head -n1
)
camera_model=$(
  printf '%s' "$oakctl_output" \
    | jq -r '.items[0].model // empty' 2>/dev/null \
    | head -n1
)
camera_agent_version=$(
  printf '%s' "$oakctl_output" \
    | jq -r '.items[0].agent_version // empty' 2>/dev/null \
    | head -n1
)
camera_os=$(
  printf '%s' "$oakctl_output" \
    | jq -r '.items[0].os // empty' 2>/dev/null \
    | head -n1
)
detected_testbed_name=$(
  printf '%s' "$oakctl_output" \
    | jq -r '.items[0].name // .items[0].hostname // .items[0].id // empty' 2>/dev/null \
    | head -n1
)
runner_hostname=$(hostname 2>/dev/null || printf 'unknown')
server_os=$(uname -s 2>/dev/null | tr '[:upper:]' '[:lower:]' || printf 'unknown')

if [ -z "$device_hostname" ]; then
  device_hostname="unknown"
fi
if [ -z "$camera_mxid" ]; then
  camera_mxid="unknown"
fi
if [ -z "$camera_model" ]; then
  camera_model="unknown"
fi
if [ -z "$camera_agent_version" ]; then
  camera_agent_version="unknown"
fi
if [ -z "$camera_os" ]; then
  camera_os="unknown"
fi
if [ -z "$runner_hostname" ]; then
  runner_hostname="unknown"
fi
if [ -z "$server_os" ]; then
  server_os="unknown"
fi
if [ -z "$HIL_TESTBED_NAME" ]; then
  HIL_TESTBED_NAME="${detected_testbed_name:-}"
fi
if [ -z "$HIL_TESTBED_NAME" ]; then
  HIL_TESTBED_NAME="$(hostname 2>/dev/null || printf '')"
fi
if [ -z "$HIL_TESTBED_NAME" ]; then
  HIL_TESTBED_NAME="unknown"
fi

# Run tests
pytest_args=(
  -s
  -v
  tests/test_benchmark/
  --testbed-name "$HIL_TESTBED_NAME"
  --camera-mxid "$camera_mxid"
  --camera-os-version "$camera_os"
  --camera-model "$camera_model"
  --camera-agent-version "$camera_agent_version"
  --runner "$runner_hostname"
  --server-os "$server_os"
)

if [ "$device_hostname" != "unknown" ]; then
  pytest_args+=(--device-ip "$device_hostname")
fi

pytest "${pytest_args[@]}"
