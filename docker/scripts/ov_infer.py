import argparse
from pathlib import Path
from typing import Dict, Tuple, TypeVar

import numpy as np
from openvino.inference_engine.ie_api import IECore

T = TypeVar("T")


def parse_args() -> Tuple[Path, Path, Dict[str, Path], Path]:
    parser = argparse.ArgumentParser(description="OpenVINO inference")

    parser.add_argument(
        "--xml-path",
        type=Path,
        metavar="PATH",
        help="Path to the OpenVINO IR XML file",
        required=True,
    )

    parser.add_argument(
        "--bin-path",
        type=Path,
        metavar="PATH",
        help="Path to the OpenVINO IR BIN file. If not provided, "
        "the path to the BIN file is assumed to be the same as the XML file "
        "but with the .bin extension",
        required=False,
        default=None,
    )

    parser.add_argument(
        "--input",
        nargs=2,
        action="append",
        help="Input name and path to the input numpy file. "
        "The array will be directly passed to the network",
        default=[],
        required=True,
    )

    parser.add_argument(
        "--out-path",
        type=Path,
        metavar="PATH",
        help="Path to the directory where the output numpy files will be saved. "
        "The output files will have the same name as the outputs of the network.",
        required=True,
    )

    args = parser.parse_args()

    inputs = {name: Path(path) for name, path in args.input}

    xml_path = args.xml_path

    out_path = args.out_path
    out_path.mkdir(parents=True, exist_ok=True)

    if args.bin_path is None:
        bin_path = args.xml_path.with_suffix(".bin")
    else:
        bin_path = args.bin_path

    return xml_path, bin_path, inputs, out_path


def main():
    xml_path, bin_path, inputs, out_path = parse_args()

    ie = IECore()

    net = ie.read_network(model=xml_path, weights=bin_path)
    exec_net = ie.load_network(network=net, device_name="CPU")
    arr_inputs = {name: np.load(path) for name, path in inputs.items()}

    outputs = exec_net.infer(inputs=arr_inputs)
    for name, output in outputs.items():
        np.save(out_path / f"{name}.npy", output)

    print(f"Outputs saved to {out_path}")


if __name__ == "__main__":
    main()
