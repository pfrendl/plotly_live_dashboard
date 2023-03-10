import dash
from dash import dcc, html
from dash.dependencies import Input, Output

from plotting import create_figure

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div(
    [
        html.H4("Classification"),
        "x axis",
        dcc.RadioItems(
            ["epochs", "time"],
            "epochs",
            id="xaxis-type",
        ),
        dcc.Graph(id="live-update-graph"),
        dcc.Interval(id="interval-component", interval=10 * 1000, n_intervals=0),
    ],
)


@app.callback(
    Output("live-update-graph", "figure"),
    Input("interval-component", "n_intervals"),
    Input("xaxis-type", "value"),
)
def update_graph_live(n_intervals: int, x_axis_radio_item: str):
    x_axis_name = {"epochs": "epoch", "time": "time"}
    fig = create_figure(x_axis_name[x_axis_radio_item])
    fig.update_layout(uirevision=x_axis_radio_item)
    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
