import csv
import os
from collections import defaultdict
from pathlib import Path

import plotly
import plotly.graph_objects as go
import plotly.subplots


def create_figure(x_axis_name: str) -> go.Figure:
    colors = plotly.colors.colorbrewer.Set1

    x_axis_title = {"epoch": "Training epochs", "time": "Wall-clock time (s)"}

    fig = plotly.subplots.make_subplots(rows=1, cols=2)
    fig.update_xaxes(title_text=x_axis_title[x_axis_name])
    fig.update_yaxes(title_text="Cross-entropy loss", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy", row=1, col=2)

    font_dict = dict(family="Arial", size=18, color="black")
    fig.update_layout(
        font=font_dict,
        plot_bgcolor="white",
        margin=dict(r=20, t=20, b=10),
        uirevision=True,
        hovermode="x unified",
    )

    fig.update_xaxes(
        zeroline=False,
        linecolor="black",
        gridcolor="rgb(200,200,200)",
        griddash="5px,2px",
        ticks="outside",
        tickfont=font_dict,
        title_font=font_dict,
        mirror=True,
        tickcolor="black",
    )
    fig.update_yaxes(
        zeroline=False,
        linecolor="black",
        gridcolor="rgb(200,200,200)",
        griddash="5px,2px",
        ticks="outside",
        tickfont=font_dict,
        title_font=font_dict,
        mirror=True,
        tickcolor="black",
    )

    log_dir = Path("runs")
    csv_paths = log_dir.rglob("log.csv")
    csv_paths = sorted(csv_paths, key=lambda path: os.path.getmtime(path.parent))

    for color_idx, path in enumerate(csv_paths):
        test_name = str(path.relative_to(log_dir).parent)

        data = defaultdict(list)
        with open(path, "r") as log_file:
            reader = csv.DictReader(log_file)
            for row in reader:
                for key, value in row.items():
                    data[key].append(float(value))

        xs = data[x_axis_name]
        for col, subplot_name in enumerate(["loss", "acc"], 1):
            for mode, dash in [("train", "dot"), ("test", "solid")]:
                trace_name = f"{test_name}/{mode}"
                ys = data[f"{mode}_{subplot_name}"]
                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=ys,
                        line=dict(dash=dash, color=colors[color_idx % len(colors)]),
                        mode="lines",
                        name=trace_name,
                        legendgroup=test_name,
                        showlegend=col == 1,
                    ),
                    1,
                    col,
                )

    return fig


def main() -> None:
    save_dir = Path("figures")
    save_dir.mkdir(exist_ok=True)
    fig = create_figure(x_axis_name="epoch")
    fig.write_html(save_dir / "figure.html")
    fig.write_image(save_dir / "figure.png", width=1500, height=700)


if __name__ == "__main__":
    main()
