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

pip install hil-framework --upgrade \
  --index-url "https://__token__:$PAT_TOKEN@gitlab.luxonis.com/api/v4/projects/213/packages/pypi/simple" \
  > /dev/null

pip install --upgrade \
  --extra-index-url "https://artifacts.luxonis.com/artifactory/luxonis-python-snapshot-local/" \
  --extra-index-url https://artifacts.luxonis.com/artifactory/luxonis-python-release-local \
  "depthai==${DEPTHAI_VERSION}"

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
printf '  pytest_args:'
printf ' %q' "${pytest_args[@]}"
printf '\n'

pytest "${pytest_args[@]}"
