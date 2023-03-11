# plotly_live_dashboard

This repository implements a lightweight machine learning training tracking solution.
It serves live and interactive [Plotly](https://plotly.com/python/) plots using [Dash](https://dash.plotly.com/) through the browser.
The plotting logic can be modified on the fly thanks to the hot reloading feature of Dash:

https://user-images.githubusercontent.com/6968154/224454106-50746641-8c5d-4eda-aa86-93ce4835e672.mp4

\
Run a training:

`python train.py <training_name>`

Run the dashboard:

`python dashboard.py`

Generate and save figures of the logged trainings:

`python plotting.py`

You can also use the in-browser UI to save snapshot images of the figure.
