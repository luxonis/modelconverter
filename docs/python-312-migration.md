# Python 3.11–3.12 migration experiment

This branch moves the host package to `>=3.11.5,<3.13` and uses Python 3.12 by
default. The 3.11.5 lower bound accommodates the Soup Sieve security update
identified in the dependency audit. Security package upgrades are a separate
change: this experiment preserves the existing locked versions wherever they
remain compatible, so it measures the interpreter transition first.

This is **not ready for release**. The host and vendor environments have different
constraints. The Python compatibility workflow runs the locked host tests without
waiting for the existing type-check job, which installs vendor dependencies. The
regular CI workflow and its conversion gates remain in place.

## Local results (Linux x86_64, 2026-11-17)

| Check                                                             | Result                                            |
| ----------------------------------------------------------------- | ------------------------------------------------- |
| Host unit suite, Python 3.11.15                                   | 1,361 passed, 5 skipped                           |
| Host unit suite, Python 3.12.13                                   | 1,361 passed, 5 skipped                           |
| Python 3.12 CLI startup (`python -m modelconverter --help`)       | Passed                                            |
| Python 3.12 wheel and sdist build                                 | Passed                                            |
| Python 3.12 docs, warnings as errors                              | Passed                                            |
| Pre-commit hooks                                                  | Passed                                            |
| Type checking of the changed keyring implementation and its tests | Passed; full vendor type checking remains blocked |

The five skips cover OpenVINO, Hailo and Plotly doctests and a TFLite
metadata case. The host suite mocks vendor behavior; it does not prove real
conversions work. The existing OpenCV override also makes `uv pip check` report
the missing `opencv-python-headless` requirement of Albumentations/Albucore;
`opencv-python` deliberately provides `cv2` instead. That existing dependency
metadata inconsistency was not introduced by this migration.

### Startup deadlock found and partially fixed

The original keyring helper uses `multiprocessing.Process` with a Linux fork.
Python 3.12 warns when forking a process that has multiple threads. LuxonisML
routes that warning through Loguru, whose at-fork locks are held, causing test
collection to hang before the lookup timeout can start. ModelConverter's helper
now runs keyring in a fresh interpreter with a subprocess timeout. Five new tests
exercise success, no password, backend failure, timeout and threaded logging.
Configured keyring backends remain available to the subprocess; a custom backend
registered only in the parent's memory is not inherited.

**An upstream issue remains:** `hubai-sdk==0.3.4` independently performs the same
fork during import. The test harness now preserves pytest's warning handler,
including while importing LuxonisML, and removes duplicate logging setup.
Warnings are still reported. This allows the host suite to run, but does not fix
HubAI SDK's fork for applications that enable fork warnings and route them to
Loguru. The SDK needs the same subprocess fix before unconditional Python 3.12
support can be claimed. Normal CLI help succeeds with the default warning policy.

## Confirmed vendor blockers

| Component           | Evidence                                                                                                                                                                                                           | Consequence                                                                                                                                                                  |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| OpenVINO 2022.3     | The pinned wheel only provides CPython 3.7–3.10 ABIs; installation fails on 3.11 and 3.12. The existing RVC2 image imports the legacy API on 3.10.                                                                 | RVC2/RVC3 image installs and `requirements-typecheck.txt` cannot simply switch interpreters. Moving to a newer OpenVINO also requires replacing `openvino.inference_engine`. |
| SNPE 2.41.0         | `libPyIrGraph.so` from the configured SDK archive imports on 3.10 but raises `ImportError: Python version mismatch: module was compiled for Python 3.10` on both 3.11 and 3.12.                                    | RVC4 must keep SDK commands on a separate supported interpreter or adopt a compatible SDK. Changing the Docker base alone is insufficient.                                   |
| Hailo               | The existing image installs ModelConverter into the vendor interpreter. Its documented TensorFlow 2.12 / NumPy 1 constraints and direct `hailo_sdk_client` imports need separate vendor validation.                | The host test result does not validate Hailo. Neither the vendor image nor the SDK was migrated in this experiment.                                                          |
| All existing images | RVC2/RVC3/RVC4 use a Python 3.10 final stage. Installing the candidate wheel in the existing RVC2 image fails: Python 3.10.22 is outside `>=3.11.5,<3.13`. The Hailo interpreter comes from the vendor base image. | The new package Python requirement cannot be released with the current image recipes unchanged.                                                                              |

Do not bypass `Requires-Python`, copy incompatible native extensions into a newer
interpreter, or interpret mocked host unit tests as SDK compatibility evidence.

## Reproduce the host checks

```sh
uv sync --locked --python 3.12
LUXONIS_TELEMETRY_ENABLED=false uv run --no-sync --python 3.12 pytest tests/unit -n auto
uv run --no-sync --python 3.12 python -m pydoctor \
  --docformat=google --warnings-as-errors \
  --project-name ModelConverter --html-output /tmp/modelconverter-apidocs modelconverter

# Use a separate environment when comparing interpreters.
UV_PROJECT_ENVIRONMENT=.venv311 uv sync --locked --python 3.11
UV_PROJECT_ENVIRONMENT=.venv311 LUXONIS_TELEMETRY_ENABLED=false \
  uv run --no-sync --python 3.11 pytest tests/unit -n auto

# Expected failure: legacy OpenVINO does not publish the required wheels.
uv pip install --python .venv/bin/python --dry-run --no-deps openvino==2022.3.0
```

## Release gates

1. Host unit tests and documentation must pass on both supported interpreters.
2. Fix HubAI SDK's import-time keyring fork and test startup with threaded logging
   and warnings enabled.
3. Select the vendor strategy: supported newer SDKs, or an explicit process
   boundary that retains the vendor interpreter and keeps host dependencies out
   of it. RVC2/RVC3/Hailo currently import vendor APIs inside the ModelConverter
   process; they need more than changing a CLI executable.
4. Update type checking without substituting a newer OpenVINO API that the
   production code does not use.
5. Rebuild every supported RVC2/RVC3/RVC4/Hailo image and run the existing conversion,
   precision and evaluation tests. Inspect the final installed versions after all
   vendor requirement overlays.
6. Apply and validate the security dependency updates separately. A higher Python
   floor alone does not update vulnerable pinned versions.
