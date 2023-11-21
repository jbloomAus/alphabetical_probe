import plotly.graph_objects as go
import numpy as np
from typing import Dict, List


def basic_bar_graph(data: Dict[int, int], ymin=0, ymax=1):
    """Creates a basic graph from a set of {x_1: y_1, ... x_n: y_n} data."""
    fig = go.Figure(go.Bar(x=list(data.keys()), y=list(data.values())))
    fig.update_layout(yaxis=dict(range=[ymin, ymax]))
    fig.show()
 
    
def nested_bar_graph(data: Dict[int, Dict[int, int]], shades: List[str], title='', xtitle='', ytitle=''):
    """Creates a nested bar graph from a nested dictionary of data.
    The bar graph has each column consisting of multiple bars, each of which is a different color.
    If less colors than bars are provided, the colors will repeat."""
    fig = go.Figure()
    # The number of keys in the inner dictionaries determines the number of bars in each group
    num_inner_keys = len(next(iter(data.values())))
    bar_width = 0.15
    gap = 0.02  # Gap between groups of bars

    # Generate the x-axis positions for the groups
    group_positions = np.linspace(0, len(data) - 1, len(data))

    # Add bars for each key in the nested dictionary
    for i, (_, values) in enumerate(data.items()):
        for j, (_, value) in enumerate(values.items()):
            # Position each bar within the group based on its sub_key
            bar_position = group_positions[i] + (j - num_inner_keys / 2) * (bar_width + gap) + bar_width / 2
            fig.add_trace(go.Bar(x=[bar_position], y=[value], width=bar_width, marker_color=shades[j % len(shades)]))

    fig.update_layout(
        barmode='group',
        title=title,
        xaxis=dict(title=xtitle, tickmode='array', tickvals=group_positions, ticktext=list(data.keys())),
        yaxis=dict(title=ytitle),
    )
    fig.show()