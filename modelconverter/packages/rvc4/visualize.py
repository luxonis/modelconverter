from pathlib import Path

import plotly.graph_objects as go
import polars as pl

from modelconverter.packages.base_visualize import Visualizer
from modelconverter.utils import constants


class RVC4Visualizer(Visualizer):
    def __init__(self, dir_path: str | None = None) -> None:
        super().__init__(dir_path=dir_path)
        self.layer_csvs = self._get_csv_paths(
            dir_path=self.dir_path, comparison_type="layer_comparison"
        )
        self.cycle_csvs = self._get_csv_paths(
            dir_path=self.dir_path, comparison_type="layer_cycles"
        )

    def visualize(self) -> None:
        fig_layers = self._visualize_layer_outputs()
        fig_layers.write_html(
            self.dir_path / "layer_outputs_visual.html",
            include_plotlyjs="cdn",
        )

        fig_cycles = self._visualize_cycles()
        fig_cycles.write_html(
            self.dir_path / "layer_cycles_visual.html", include_plotlyjs="cdn"
        )
        fig_layers.show()
        fig_cycles.show()

    def _visualize_cycles(self) -> go.Figure:
        layer_lists = []
        for model_name, csv_path in self.cycle_csvs.items():
            df = pl.read_csv(csv_path)
            df = df.with_columns(pl.lit(model_name).alias("model_name"))
            df.columns = df.columns
            new_columns = {col: col.strip() for col in df.columns}
            df = df.rename(new_columns)
            layer_lists.append(df["layer_name"].to_list())

        x_labels = self._create_x_labels(layer_lists)

        metrics = ["time_mean", "Percentage_of_Total_Time"]
        initial_metric = metrics[0]

        traces_data = {}
        for model_name, csv_path in self.cycle_csvs.items():
            df = pl.read_csv(csv_path)
            new_columns = {col: col.strip() for col in df.columns}
            df = df.rename(new_columns)
            df = df.with_columns(
                (
                    pl.col("Percentage_of_Total_Time").cast(pl.Float32()) * 100
                ).alias("Percentage_of_Total_Time")
            )
            metric_maps = {
                metric: dict(
                    zip(
                        df["layer_name"].to_list(),
                        df[metric].to_list(),
                        strict=True,
                    )
                )
                for metric in metrics
            }
            model_data = {"x_axis": x_labels}
            for metric in metrics:
                model_data[metric] = [
                    metric_maps[metric].get(layer, None) for layer in x_labels
                ]
            traces_data[model_name] = model_data

        fig = go.Figure()

        for model, data in traces_data.items():
            fig.add_trace(
                go.Bar(
                    x=data["x_axis"],
                    y=data[initial_metric],
                    name=model,
                    hovertemplate=f"Model: {model}<br>Layer: %{{x}}<br>{initial_metric}: %{{y}}<extra></extra>",
                    visible=True,
                )
            )

        buttons = []
        for metric in metrics:
            new_y = []
            new_hovertemplates = []
            for model in self.layer_csvs:
                new_y.append(traces_data[model][metric])
                new_hovertemplates.append(
                    f"Model: {model}<br>Layer: %{{x}}<br>{metric}: %{{y}}<extra></extra>"
                )

            button = {
                "label": metric,
                "method": "update",
                "args": [
                    {"y": new_y, "hovertemplate": new_hovertemplates},
                    {"yaxis": {"title": metric}},
                ],
            }
            buttons.append(button)

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
            yaxis_title=initial_metric,
            title="CPU Cycles per Layer by Model",
            hoverlabel={
                "font": {"size": 16},
            },
            xaxis={"tickfont": {"size": 16}, "tickangle": 45},
        )

        return fig

    def _visualize_layer_outputs(self) -> go.Figure:
        layer_lists = []
        for csv_path in self.layer_csvs.values():
            df = pl.read_csv(csv_path)
            new_columns = {col: col.strip() for col in df.columns}
            df = df.rename(new_columns)
            layer_lists.append(df["layer_name"].to_list())

        x_labels = self._create_x_labels(layer_lists)

        metrics = ["max_abs_diff", "MSE", "cos_sim"]
        initial_metric = metrics[0]

        traces_data = {}
        for model_name, csv_path in self.layer_csvs.items():
            df = pl.read_csv(csv_path)
            new_columns = {col: col.strip() for col in df.columns}
            df = df.rename(new_columns)
            metric_maps = {
                metric: dict(
                    zip(
                        df["layer_name"].to_list(),
                        df[metric].to_list(),
                        strict=True,
                    )
                )
                for metric in metrics
            }
            model_data = {"x_axis": x_labels}
            for metric in metrics:
                model_data[metric] = [
                    metric_maps[metric].get(layer, None) for layer in x_labels
                ]
            traces_data[model_name] = model_data

        fig = go.Figure()
        for model in self.layer_csvs:
            fig.add_trace(
                go.Scatter(
                    x=traces_data[model]["x_axis"],
                    y=traces_data[model][initial_metric],
                    mode="markers",
                    name=model,
                    hovertemplate=f"Model: {model}<br>Layer: %{{x}}<br>{initial_metric}: %{{y}}<extra></extra>",
                )
            )
        buttons = []
        for metric in metrics:
            new_y = []
            new_hovertemplates = []
            for model in self.layer_csvs:
                new_y.append(traces_data[model][metric])
                new_hovertemplates.append(
                    f"Model: {model}<br>Layer: %{{x}}<br>{metric}: %{{y}}<extra></extra>"
                )

            button = {
                "label": metric,
                "method": "update",
                "args": [
                    {"y": new_y, "hovertemplate": new_hovertemplates},
                    {"yaxis": {"title": metric}},
                ],
            }
            buttons.append(button)

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
            yaxis_title=initial_metric,
            title="Layer Performance Metrics by Model",
            hoverlabel={"font": {"size": 16}},
            xaxis={"tickfont": {"size": 16}, "tickangle": 45},
        )

        return fig

    def _get_csv_paths(
        self, dir_path: Path, comparison_type: str = "layer_comparison"
    ) -> dict[str, str]:
        dir_path = dir_path if dir_path else constants.OUTPUTS_DIR / "analysis"
        csv_paths = {}

        for file in dir_path.glob(f"*{comparison_type}*.csv"):
            csv_paths[file.parent.name] = file

        return csv_paths

    def _create_x_labels(self, layer_lists: list) -> list:
        pointers = [0] * len(layer_lists)
        x_labels = []

        while any(
            pointers[i] < len(layer_lists[i]) for i in range(len(layer_lists))
        ):
            layer_lists = [
                layer_lists[i]
                for i in range(len(layer_lists))
                if pointers[i] < len(layer_lists[i])
            ]
            pointers = [
                pointers[i]
                for i in range(len(layer_lists))
                if pointers[i] < len(layer_lists[i])
            ]

            current_candidates = [
                layer_lists[i][pointers[i]] for i in range(len(layer_lists))
            ]

            if len(set(current_candidates)) == 1:
                x_labels.append(current_candidates[0])
                pointers = [p + 1 for p in pointers]
                continue

            candidate_is_insertion = [False] * len(current_candidates)
            for i, candidate in enumerate(current_candidates):
                candidate_is_insertion[i] = any(
                    candidate not in layer_lists[j][pointers[j] :]
                    for j in range(len(layer_lists))
                )
                # true if any model list of layers does not contain the candidate layer

            for i in range(len(current_candidates)):
                if (
                    candidate_is_insertion[i]
                    and current_candidates[i] not in x_labels
                ):
                    x_labels.append(candidate)  # is insertion, not added yet

                if candidate_is_insertion[i]:
                    pointers[i] += 1  # increase index for all insertions

        return x_labels
