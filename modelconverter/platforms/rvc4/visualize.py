"""Visualization of the RVC4 analysis results.

Turns the CSV reports written by the RVC4 analyzer into interactive
``plotly`` figures: one comparing the per-layer outputs of the
quantized model with the original ONNX model, and one comparing the
per-layer inference times. Both are saved as HTML files that load
``plotly.js`` from a CDN.
"""

from collections import deque
from pathlib import Path

import plotly.graph_objects as go
import polars as pl

from modelconverter.platforms.base_visualize import Visualizer
from modelconverter.utils import constants


class RVC4Visualizer(Visualizer):
    """Visualizer of the CSV reports produced by the RVC4 analyzer.

    Collects the layer comparison and layer cycles CSV files lying
    directly in the analysis directory -- keyed by the name of that
    directory, which the analyzer names after the model -- and plots
    one series per key.
    """

    def __init__(self, dir_path: str | None = None) -> None:
        """Initialize the visualizer and locate the CSV reports.

        Args:
            dir_path: Directory containing the analysis results. If
                ``None``, the ``analysis`` subdirectory of the default
                output directory is used.

        """
        super().__init__(dir_path=dir_path)
        self._layer_csvs = self._get_csv_paths(
            dir_path=self._dir_path, comparison_type="layer_comparison"
        )
        self._cycle_csvs = self._get_csv_paths(
            dir_path=self._dir_path, comparison_type="layer_cycles"
        )

    def visualize(self) -> None:
        """Create, save and show the analysis figures.

        Writes ``layer_outputs_visual.html`` and
        ``layer_cycles_visual.html`` into the analysis directory and
        opens both figures.
        """
        fig_layers = self._visualize_layer_outputs()
        fig_layers.write_html(
            self._dir_path / "layer_outputs_visual.html",
            include_plotlyjs="cdn",
        )

        fig_cycles = self._visualize_cycles()
        fig_cycles.write_html(
            self._dir_path / "layer_cycles_visual.html", include_plotlyjs="cdn"
        )
        fig_layers.show()
        fig_cycles.show()

    def _visualize_cycles(self) -> go.Figure:
        frames = {
            model_name: _read_csv(csv_path)
            for model_name, csv_path in self._cycle_csvs.items()
        }
        x_labels = self._create_x_labels(
            [df["layer_name"].to_list() for df in frames.values()]
        )

        metrics = ["time_mean", "Percentage_of_Total_Time"]
        traces_data = {
            model_name: _align_metrics(
                df.with_columns(
                    (
                        pl.col("Percentage_of_Total_Time").cast(pl.Float32())
                        * 100
                    ).alias("Percentage_of_Total_Time")
                ),
                metrics,
                x_labels,
            )
            for model_name, df in frames.items()
        }

        fig = go.Figure()
        for model, data in traces_data.items():
            fig.add_trace(
                go.Bar(
                    x=data["x_axis"],
                    y=data[metrics[0]],
                    name=model,
                    hovertemplate=_hover_template(model, metrics[0]),
                    visible=True,
                )
            )
        _add_metric_buttons(
            fig,
            metrics,
            traces_data,
            title="CPU Cycles per Layer by Model",
        )
        return fig

    def _visualize_layer_outputs(self) -> go.Figure:
        frames = {
            model_name: _read_csv(csv_path)
            for model_name, csv_path in self._layer_csvs.items()
        }
        x_labels = self._create_x_labels(
            [df["layer_name"].to_list() for df in frames.values()]
        )

        metrics = ["max_abs_diff", "MSE", "cos_sim"]
        traces_data = {
            model_name: _align_metrics(df, metrics, x_labels)
            for model_name, df in frames.items()
        }

        fig = go.Figure()
        for model, data in traces_data.items():
            fig.add_trace(
                go.Scatter(
                    x=data["x_axis"],
                    y=data[metrics[0]],
                    mode="markers",
                    name=model,
                    hovertemplate=_hover_template(model, metrics[0]),
                )
            )
        _add_metric_buttons(
            fig,
            metrics,
            traces_data,
            title="Layer Performance Metrics by Model",
        )
        return fig

    def _get_csv_paths(
        self, dir_path: Path, comparison_type: str = "layer_comparison"
    ) -> dict[str, Path]:
        dir_path = dir_path or constants.OUTPUTS_DIR / "analysis"
        csv_paths = {}

        for file in dir_path.glob(f"*{comparison_type}*.csv"):
            csv_paths[file.parent.name] = file

        return csv_paths

    def _create_x_labels(self, layer_lists: list[list[str]]) -> list[str]:
        """Merge the layer orders of several models into one order.

        When all models agree on the next layer, the layer is taken
        once. Otherwise, a next layer that another model does not hold
        further on is an insertion: it is taken, and its model moves
        on.
        """
        queues = [deque(layers) for layers in layer_lists if layers]
        x_labels: list[str] = []
        while queues:
            _take_next_layers(queues, x_labels)
            queues = [queue for queue in queues if queue]
        return x_labels


def _take_next_layers(queues: list[deque[str]], x_labels: list[str]) -> None:
    """Take the next layers into ``x_labels`` and advance their queues."""
    if len({queue[0] for queue in queues}) == 1:
        x_labels.append(queues[0][0])
        moving = queues
    else:
        moving = [
            queue
            for queue in queues
            if any(queue[0] not in other for other in queues)
        ]
        for queue in moving:
            if queue[0] not in x_labels:
                x_labels.append(queue[0])
    for queue in moving:
        queue.popleft()


def _read_csv(csv_path: Path) -> pl.DataFrame:
    """Read a CSV report and strip the spaces around its column names."""
    df = pl.read_csv(csv_path)
    return df.rename({col: col.strip() for col in df.columns})


def _align_metrics(
    df: pl.DataFrame, metrics: list[str], x_labels: list[str]
) -> dict[str, list]:
    """Line up the metric values of a model with the x labels.

    A layer that the model does not hold gets ``None``.
    """
    layer_names = df["layer_name"].to_list()
    data: dict[str, list] = {"x_axis": x_labels}
    for metric in metrics:
        values = dict(zip(layer_names, df[metric].to_list(), strict=True))
        data[metric] = [values.get(layer) for layer in x_labels]
    return data


def _hover_template(model: str, metric: str) -> str:
    """Format the hover text of the points of one model."""
    return (
        f"Model: {model}<br>Layer: %{{x}}<br>{metric}: %{{y}}<extra></extra>"
    )


def _add_metric_buttons(
    fig: go.Figure,
    metrics: list[str],
    traces_data: dict[str, dict[str, list]],
    *,
    title: str,
) -> None:
    """Add buttons that switch the plotted metric, and lay out the figure.

    The figure first shows the first metric.
    """
    buttons = [
        {
            "label": metric,
            "method": "update",
            "args": [
                {
                    "y": [data[metric] for data in traces_data.values()],
                    "hovertemplate": [
                        _hover_template(model, metric) for model in traces_data
                    ],
                },
                {"yaxis": {"title": metric}},
            ],
        }
        for metric in metrics
    ]
    fig.update_layout(
        updatemenus=[
            {
                "type": "buttons",
                "buttons": buttons,
                "direction": "right",
                "showactive": True,
                "x": 0.5,
                "xanchor": "center",
                "y": 1.15,
                "yanchor": "top",
                "pad": {"r": 10, "t": 10},
            }
        ],
        xaxis_title="Layer",
        yaxis_title=metrics[0],
        title=title,
        hoverlabel={"font": {"size": 16}},
        xaxis={"tickfont": {"size": 16}, "tickangle": 45},
    )
