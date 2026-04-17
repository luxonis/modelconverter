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

rm -rf hil_framework

git clone --recurse-submodules -b tjb_influx_pusher https://oauth2:$PAT_TOKEN@gitlab.luxonis.com/luxonis/hil_lab/hil_framework.git
pip install ./hil_framework/

rm -rf hil_framework

pip install --upgrade \
  --extra-index-url "https://artifacts.luxonis.com/artifactory/luxonis-python-snapshot-local/" \
  --extra-index-url https://artifacts.luxonis.com/artifactory/luxonis-python-release-local \
  "depthai==${DEPTHAI_VERSION}"

runner_hostname=$(hostname 2>/dev/null || printf 'unknown')
if [ -z "$runner_hostname" ]; then
  runner_hostname="unknown"
fi

# Run tests
pytest_args=(
  -s
  -v
  tests/test_benchmark/
  --depthai-version "$DEPTHAI_VERSION"
)

if [ -n "${HIL_TESTBED:-}" ]; then
  pytest_args+=(--testbed-name "$HIL_TESTBED")
fi
if [ -n "$BENCHMARK_RUN_ID" ]; then
  pytest_args+=(--benchmark-run-id "$BENCHMARK_RUN_ID")
fi

echo "Influx metadata debug:"
echo "  INFLUX_BUCKET=fps_metrics"
echo "  INFLUX_TOKEN=$(if [ -n "${INFLUX_TOKEN:-}" ]; then printf '<set>'; else printf '<empty>'; fi)"
echo "  DEPTHAI_VERSION=${DEPTHAI_VERSION:-<empty>}"
echo "  benchmark_run_id=${BENCHMARK_RUN_ID:-<generated>}"
echo "  HIL_TESTBED=${HIL_TESTBED:-<empty>}"
echo "  runner=${runner_hostname:-<empty>}"
printf '  pytest_args:'
printf ' %q' "${pytest_args[@]}"
printf '\n'

pytest "${pytest_args[@]}"
