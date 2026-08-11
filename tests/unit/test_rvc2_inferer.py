"""``RVC2Inferer.setup`` reading the input layout off the OpenVINO IR.

The config describes the *original* model, whose layout the conversion need not
keep -- a TFLite source yields an NHWC IR from a config that says nothing about
layouts. ``infer`` hands ``read_image`` the IR's shape, and ``read_image`` reads
that shape through the layout, so both have to come from the IR or the image is
transposed the wrong way round.

Driving ``setup`` with a stub keeps this host-side: the real thing needs
OpenVINO, which only exists inside the RVC2 image.
"""

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from modelconverter.packages.rvc2.inferer import RVC2Inferer


def _install_fake_openvino(
    monkeypatch: pytest.MonkeyPatch, net_shapes: dict[str, tuple[int, ...]]
) -> None:
    """Stand in for the IR that ``setup`` reads its shapes from."""
    net = SimpleNamespace(
        input_info={
            name: SimpleNamespace(input_data=SimpleNamespace(shape=shape))
            for name, shape in net_shapes.items()
        }
    )
    module = ModuleType("openvino.inference_engine.ie_api")
    module.IECore = lambda: SimpleNamespace(  # type: ignore[attr-defined]
        read_network=lambda model, weights: net,
        load_network=lambda network, device_name: object(),
    )
    monkeypatch.setitem(
        sys.modules, "openvino.inference_engine.ie_api", module
    )


@pytest.mark.parametrize(
    ("net_shape", "config_shape", "config_layout", "layout"),
    [
        # The case that regressed: the config happens to carry the IR's own
        # NHWC shape but no layout, so there is nothing to fall back on.
        ((1, 224, 224, 3), [1, 224, 224, 3], None, "NHWC"),
        ((1, 3, 224, 224), [1, 3, 224, 224], None, "NCHW"),
        # NCHW config, NHWC IR: the layout follows the shape it belongs to.
        ((1, 224, 224, 3), [1, 3, 224, 224], "NCHW", "NHWC"),
        # Single-channel and RGBA inputs are still channels-first.
        ((1, 1, 32, 32), [1, 1, 32, 32], None, "NCHW"),
        ((1, 4, 32, 32), [1, 4, 32, 32], None, "NCHW"),
        # Nothing to infer from a rank the heuristic says nothing about.
        ((1, 128), [1, 128], None, None),
    ],
)
def test_layout_comes_from_the_ir(
    monkeypatch: pytest.MonkeyPatch,
    net_shape: tuple[int, ...],
    config_shape: list[int],
    config_layout: str | None,
    layout: str | None,
):
    _install_fake_openvino(monkeypatch, {"x": net_shape})
    stub = SimpleNamespace(
        model_path=Path("model.xml"),
        in_shapes={"x": config_shape},
        layout={"x": config_layout},
    )

    RVC2Inferer.setup(stub)  # type: ignore[arg-type]

    assert stub.in_shapes == {"x": list(net_shape)}
    assert stub.layout == {"x": layout}
