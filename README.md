# ModelConverter - Compilation Library

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI](https://img.shields.io/pypi/v/modelconv?label=pypi%20package)](https://pypi.org/project/modelconv/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/modelconv)](https://pypi.org/project/modelconv/)

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Docformatter](https://img.shields.io/badge/%20formatter-docformatter-fedcba.svg)](https://github.com/PyCQA/docformatter)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Convert your **ONNX** models to a format compatible with any generation of Luxonis camera using the **Model Compilation Library**.

## Status

| Package   | Test                                                                                                  | Deploy                                                                                                  |
| --------- | ----------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| **RVC2**  | ![RVC2 Tests](https://github.com/luxonis/modelconverter/actions/workflows/rvc2_test.yaml/badge.svg)   | ![RVC2 Push](https://github.com/luxonis/modelconverter/actions/workflows/rvc2_publish.yaml/badge.svg)   |
| **RVC3**  | ![RVC3 Tests](https://github.com/luxonis/modelconverter/actions/workflows/rvc3_test.yaml/badge.svg)   | ![RVC3 Push](https://github.com/luxonis/modelconverter/actions/workflows/rvc3_publish.yaml/badge.svg)   |
| **RVC4**  | ![RVC4 Tests](https://github.com/luxonis/modelconverter/actions/workflows/rvc4_test.yaml/badge.svg)   | ![RVC4 Push](https://github.com/luxonis/modelconverter/actions/workflows/rvc4_publish.yaml/badge.svg)   |
| **Hailo** | ![Hailo Tests](https://github.com/luxonis/modelconverter/actions/workflows/hailo_test.yaml/badge.svg) | ![Hailo Push](https://github.com/luxonis/modelconverter/actions/workflows/hailo_publish.yaml/badge.svg) |

## Table of Contents

- [MLOps - Compilation Library](#mlops---compilation-library)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
    - [Before You Begin](#before-you-begin)
    - [Instructions](#instructions)
    - [GPU Support](#gpu-support)
  - [Running ModelConverter](#running-modelconverter)
    - [Sharing Files](#sharing-files)
    - [Usage](#usage)
    - [Examples](#examples)
  - [Multi-Stage Conversion](#multi-stage-conversion)
  - [Interactive Mode](#interactive-mode)
  - [Calibration Data](#calibration-data)
  - [Inference](#inference)
    - [Inference Example](#inference-example)
  - [Benchmarking](#benchmarking)

## Installation

### System Requirements

`ModelConverter` requires `docker` to be installed on your system.
It is recommended to use Ubuntu OS for the best compatibility.
On Windows or MacOS, it is recommended to install `docker` using the [Docker Desktop](https://www.docker.com/products/docker-desktop).
Otherwise follow the installation instructions for your OS from the [official website](https://docs.docker.com/engine/install/).

### Before You Begin

`ModelConverter` is in an experimental public beta stage. Some parts might change in the future.

To build the images, you need to download additional packages depending on the selected target.

**RVC2 and RVC3**

Requires `openvino_2022_3_vpux_drop_patched.tar.gz` to be present in `docker/extra_packages`.
You can download the archive [here](https://drive.google.com/file/d/1IXtYi1Mwpsg3pr5cDXlEHdSUZlwJRTVP/view?usp=share_link).

**RVC4**

Requires `snpe.zip` archive to be present in `docker/extra_packages`. You can download an archive with the current version [here](https://softwarecenter.qualcomm.com/api/download/software/qualcomm_neural_processing_sdk/v2.23.0.24.06.24.zip). You only need to rename it to `snpe.zip` and place it in the `docker/extra_packages` directory.

**HAILO**

Requires `hailo_ai_sw_suite_2024-04:1` docker image to be present on the system. You can download the image from the [Hailo website](https://developer.hailo.ai/developer-zone/sw-downloads/).

### Instructions

1. Build the docker image:

   ```bash
   docker build -f docker/<package>/Dockerfile.public -t luxonis/modelconverter-<package>:latest .
   ```
1. For easier use, you can install the ModelConverter CLI. You can install it from PyPI using the following command:

   ```bash
   pip install modelconv
   ```
   For usage instructions, see `modelconverter --help`.

### GPU Support

To enable GPU acceleration for `hailo` conversion, install the [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker).

## Running ModelConverter

Configuration for the conversion predominantly relies on a `yaml` config file. For reference, see [defaults.yaml](shared_with_container/configs/defaults.yaml) and other examples located in the [shared_with_container/configs](shared_with_container/configs) directory.

However, you have the flexibility to modify specific settings without altering the config file itself. This is done using command line arguments. You provide the arguments in the form of `key value` pairs. For better understanding, see [Examples](#examples).

### Sharing Files

When using the supplied `docker-compose.yaml`, the `shared_with_container` directory facilitates file sharing between the host and container. This directory is mounted as `/app/shared_with_container/` inside the container. You can place your models, calibration data, and config files here. The directory structure is:

```txt
shared_with_container/
│
├── calibration_data/
│ └── <calibration data will be downloaded here>
│
├── configs/
│ ├── resnet18.yaml
│ └── <configs will be downloaded here>
│
├── models/
│ ├── resnet18.onnx
│ └── <models will be downloaded here>
│
└── outputs/
  └── <output_dir_name>
    ├── resnet18.onnx
    ├── resnet18.dlc
    ├── logs.txt
    ├── config.yaml
    └── intermediate_outputs/
      └── <intermediate files generated during the conversion>
```

While adhering to this structure is not mandatory as long as the files are visible inside the container, it is advised to keep the files organized.

The converter first searches for files exactly at the provided path. If not found, it searches relative to `/app/shared_with_container/`.

The `output_dir_name` can be specified in the config file. If such a directory already exists, the `output_dir_name` will be appended with the current date and time. If not specified, the `output_dir_name` will be autogenerated in the following format: `<model_name>_to_<target>_<date>_<time>`.

### Usage

You can run the built image either manually using the `docker run` command or using the `modelconverter` CLI.

1. Set your credentials as environment variables (if required):

   ```bash
   export AWS_SECRET_ACCESS_KEY=<your_aws_secret_access_key>
   export AWS_ACCESS_KEY_ID=<your_aws_access_key_id>
   export AWS_S3_ENDPOINT_URL=<your_aws_s3_endpoint_url>
   ```

1. If `shared_with_container` directory doesn't exist on your host, create it.

1. Without remote files, place the model, config, and calibration data in the respective directories (refer [Sharing Files](#sharing-files)).

1. Execute the conversion:

- If using the `docker run` command:
  ```bash
  docker run --rm -it \
    -v $(pwd)/shared_with_container:/app/shared_with_container/ \
    -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
    -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
    -e AWS_S3_ENDPOINT_URL=$AWS_S3_ENDPOINT_URL \
    luxonis/modelconverter-<package>:latest \
    convert <target> \
    --path <s3_url_or_path> [ config overrides ]
  ```
- If using the `modelconverter` CLI:
  ```bash
  modelconverter convert <target> --path <s3_url_or_path> [ config overrides ]
  ```
- If using `docker-compose`:
  ```bash
  docker compose run <target> convert <target> ...
  ```

### Examples

Use `resnet18.yaml` config, but override `calibration.path`:

```bash
modelconverter convert rvc4 --path configs/resnet18.yaml \
                        calibration.path s3://path/to/calibration_data
```

Override inputs and outputs with command line arguments:

```bash
modelconverter convert rvc3 --path configs/resnet18.yaml \
                        inputs.0.name input_1 \
                        inputs.0.shape "[1,3,256,256]" \
                        outputs.0.name output_0
```

Specify all options via the command line without a config file:

```bash
modelconverter convert rvc2 input_model models/yolov6n.onnx \
                        scale_values "[255,255,255]" \
                        reverse_input_channels True \
                        shape "[1,3,256,256]" \
                        outputs.0.name out_0 \
                        outputs.1.name out_1 \
                        outputs.2.name out_2
```

## Multi-Stage Conversion

The converter supports multi-stage conversion. This means conversion of multiple
models where the output of one model is the input to another model. For mulit-stage
conversion you must specify the `stages` section in the config file, see [defaults.yaml](shared_with_container/configs/defaults.yaml)
and [multistage.yaml](shared_with_container/configs/multistage.yaml) for reference.

The output directory structure would be (assuming RVC4 conversion):

```txt
output_path/
├── config.yaml
├── modelconverter.log
├── stage_name1
│   ├── config.yaml
│   ├── intermediate_outputs/
│   ├── model1.onnx
│   └── model1.dlc
└── stage_name2
    ├── config.yaml
    ├── intermediate_outputs/
    ├── model2.onnx
    └── model2.dlc
```

## Interactive Mode

Run the container interactively without any post-target arguments:

```bash
modelconverter shell rvc4
```

Inside, you'll find all the necessary tools for manual conversion.
The `modelconverter` CLI is available inside the container as well.

## Calibration Data

Calibration data can be a mix of images (`.jpg`, `.png`, `.jpeg`) and `.npy`, `.raw` files.
Image files will be loaded and converted to the format specified in the config.
No conversion is performed for `.npy` or `.raw` files, the files are used as provided.
**NOTE for RVC4**: `RVC4` expects images to be provided in `NHWC` layout. If you provide the calibration data in a form of `.npy` or `.raw` format, you need to make sure they have the correct layout.

## Inference

A basic support for inference. To run the inference, use `modelconverter infer <target> <args>`.
For usage instructions, see `modelconverter infer --help`.

The input files must be provided in a specific directory structure.

```txt
input_path/
├── <name of first input node>
│   ├── 0.npy
│   ├── 1.npy
│   └── ...
├── <name of second input node>
│   ├── 0.npy
│   ├── 1.npy
│   └── ...
├── ...
└── <name of last input node>
    ├── 0.npy
    ├── 1.npy
    └── ...
```

**Note**: The numpy files are sent to the model with no preprocessing, so they must be provided in the correct format and shape.

The output files are then saved in a similar structure.

### Inference Example

For `yolov6n` model, the input directory structure would be:

```txt
input_path/
└── images
    ├── 0.npy
    ├── 1.npy
    └── ...
```

To run the inference, use:

```bash
modelconverter infer rvc4 \
  --model_path <path_to_model.dlc> \
  --dest <dest> \
  --input_path <input_path>
  --path <path_to_config.yaml>
```

The output directory structure would be:

```txt
output_path/
├── output1_yolov6r2
│   ├── 0.npy
│   ├── 1.npy
│   └── ...
├── output2_yolov6r2
│   └── <outputs>
└── output3_yolov6r2
    └── <outputs>
```

## Benchmarking

The ModelConverter additionally supports benchmarking of converted models.

To install the package with the benchmarking dependencies, use:

```bash
pip install modelconv[bench]
```

To run the benchmark, use `modelconverter benchmark <target> <args>`.

For usage instructions, see `modelconverter benchmark --help`.

**Example:**

```bash
modelconverter benchmark rvc3 --model-path <path_to_model.xml>
```

The command prints a table with the benchmark results to the console and
optionally saves the results to a `.csv` file.
