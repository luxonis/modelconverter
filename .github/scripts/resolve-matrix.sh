#!/usr/bin/env bash
# Filter the build matrix in `.github/matrix.json` by platform and tool version.
#
# Usage: resolve-matrix.sh [PLATFORMS] [TOOL_VERSIONS]
#
# Both arguments are comma-separated lists; an empty argument matches everything.
# The filtered matrix is written to stdout as a single line of JSON, ready to be
# fed to `fromJSON` in a workflow's `strategy.matrix.include`.
set -euo pipefail

MATRIX_FILE="$(dirname "$0")/../matrix.json"
PLATFORMS="${1:-}"
TOOL_VERSIONS="${2:-}"

FILTERED=$(
  jq -c --arg platforms "${PLATFORMS}" --arg versions "${TOOL_VERSIONS}" '
    def to_list: split(",") | map(gsub("^\\s+|\\s+$"; "")) | map(select(. != ""));
    ($platforms | ascii_downcase | to_list) as $platforms
    | ($versions | to_list) as $versions
    | map(select(
        (($platforms | length) == 0 or (.package | IN($platforms[])))
        and (($versions | length) == 0 or (.version | IN($versions[])))
      ))
  ' "${MATRIX_FILE}"
)

if [ "$(jq length <<<"${FILTERED}")" -eq 0 ]; then
  {
    echo "No matrix entry matches platforms='${PLATFORMS}' tool_versions='${TOOL_VERSIONS}'."
    echo "Available combinations:"
    jq -r '.[] | "  \(.package) \(.version)"' "${MATRIX_FILE}"
  } >&2
  exit 1
fi

echo "${FILTERED}"
