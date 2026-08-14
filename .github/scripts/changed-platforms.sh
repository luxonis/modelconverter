#!/usr/bin/env bash
# Map changed files to the platforms whose integration tests they can break.
#
# Reads one file path per line on stdin and prints the affected platforms as a
# comma-separated list. Empty output means no integration test needs to run.
#
# Paths that belong to a single platform select only that platform, a handful of
# paths that cannot affect any image (docs, unrelated workflows) select nothing,
# and anything else counts as shared code and selects every platform. Defaulting
# to "everything" keeps a new shared file from silently skipping the tests.
set -euo pipefail

ALL="rvc2,rvc3,rvc4,hailo"
SELECTED=""

select_platforms() {
  for platform in "$@"; do
    case ",${SELECTED}," in
      *",${platform},"*) ;;
      *) SELECTED="${SELECTED:+${SELECTED},}${platform}" ;;
    esac
  done
}

while IFS= read -r file; do
  [ -n "${file}" ] || continue
  case "${file}" in
    # Documentation and repository chores never reach an image. `.dockerignore`
    # is not in this group: it defines the build context of every image.
    *.md | LICENSE | .gitignore | .pre-commit-config.yaml) ;;

    # The conversion tests are unaffected by the other test suites, and the
    # unit tests run on every pull request anyway.
    tests/unit/* | tests/test_benchmark/*) ;;

    # Everything driving the conversion matrix re-runs the whole matrix; the
    # remaining workflows (publishing, HIL, Semgrep) do not.
    .github/workflows/ci.yaml | .github/matrix.json | .github/scripts/*)
      echo "${ALL}"
      exit 0
      ;;
    .github/*) ;;

    # `RVC3Exporter` subclasses `RVC2Exporter`, and `RVC3Inferer` is an alias of
    # `RVC2Inferer`, so a change to the RVC2 code can break RVC3.
    modelconverter/platforms/rvc2/*) select_platforms rvc2 rvc3 ;;
    modelconverter/platforms/rvc3/*) select_platforms rvc3 ;;
    modelconverter/platforms/rvc4/*) select_platforms rvc4 ;;
    modelconverter/platforms/hailo/*) select_platforms hailo ;;

    # RVC3's image is built on OpenVINO like RVC2's and reuses its entrypoint,
    # so the shared OpenVINO tooling and RVC2's docker directory affect both.
    docker/rvc2/* | docker/patches/* | docker/scripts/*)
      select_platforms rvc2 rvc3
      ;;
    docker/rvc3/*) select_platforms rvc3 ;;
    docker/rvc4/*) select_platforms rvc4 ;;
    docker/hailo/*) select_platforms hailo ;;

    # Shared code — test every platform.
    *)
      echo "${ALL}"
      exit 0
      ;;
  esac
done

echo "${SELECTED}"
