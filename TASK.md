# FPS Influx Split Plan

Goal: move reusable Influx-writing behavior out of `modelconverter` and into `hil_framework`, while keeping all FPS benchmark logic and repository-specific CI glue in `modelconverter`.

## Keep in `modelconverter`

- `tests/test_benchmark/test_benchmark_regression.py`
  - `test_benchmark_fps(...)`
  - `_model_slugs(...)`
  - `_model_id(...)`
  - benchmark math and assertion logic
  - reading `benchmark_targets.json`
- `tests/test_benchmark/conftest.py`
  - `pytest_addoption(...)`
  - `pytest_configure(...)`
  - `device_ip(...)`
  - `benchmark_target(...)`
  - `benchmark_run_id(...)` unless it becomes shared HIL metadata
- `tests/test_benchmark/run_hil_tests.sh`
- `.github/workflows/fps_benchmark_influx.yaml`
- `tjb_run_fps_influx.sh`

## Move to `hil_framework`

### 1. Generic Influx write helper

Replace the manual HTTP / line-protocol write path in `test_benchmark_regression.py` with the existing `hil_framework` client API.

Preferred shape:

- build a `Point("fps_benchmark")`
- attach tags and fields
- call `InfluxClient("hil").save_points([point])`

This keeps benchmark data composition in `modelconverter`, but moves the transport and serialization style into the shared HIL implementation.

Candidate logic to remove from `modelconverter` once the helper is in place:

- `_escape_tag(...)`
- `_normalize_tag(...)`
- `_format_field(...)`
- the raw HTTP request construction
- the explicit Influx URL / token / org handling
- the timeout / `URLError` handling around `urllib.request.urlopen`

The helper should live beside the existing `InfluxClient` / `StabilityInflux` code, but not inside the stability-specific classes.

### 2. Shared HIL metadata helper

If some metadata is better derived from `hil_framework` objects, move that logic out of `tests/test_benchmark/conftest.py` and into a shared helper:

- `influx_metadata(...)`

Derivable from `Testbed` / `camera` / `server`:

- `HIL_TESTBED` from `testbed.config.name` or `camera.config.testbed_name`
- `HIL_CAMERA_MXID`
- `HIL_CAMERA_OS_VERSION` from `camera.get_os_version()` on RVC4 cameras
- `HIL_CAMERA_MODEL`
- `HIL_CAMERA_REVISION`
- `HIL_SERVER_OS` from `camera.server.os`

`HIL_RUNNER` should be ignored.

`DEPTHAI_VERSION` must stay as an explicit input parameter to the benchmark flow.
It should be passed through from the workflow into the benchmark test path and preserved in the Influx payload if needed.

## Do not move

- FPS benchmark model selection
- tolerance math
- pass/fail logic
- workflow dispatch scripts
- benchmark target baselines
- any modelconverter-only fixture data
- benchmark-specific selection of the RVC4 camera to inspect

## Suggested implementation order

1. Add or extend a helper in `hil_framework` that turns benchmark data into `Point` objects and writes them with `InfluxClient("hil").save_points`.
2. Update `modelconverter` to call that helper from the FPS benchmark test.
3. Keep the benchmark math and assertions in `modelconverter`.
4. Remove the now-unused local Influx serialization helpers from `modelconverter`.

## Exclusions

The removed fixture files `fake_fps_metric.csv` and `fake_fps_metric.lp` should stay deleted. They are leftovers, not part of the runtime benchmark flow.
