from evals.spelling_by_grade import SpellingEvalDict
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go


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
 
   
def create_table(dataset: Dict[int, SpellingEvalDict], n_pairs=5):
    """Create a color-coded table from a dataset of {grade: SpellingEvalDict} dicts."""
    table, current_row, columns = [], [], []
    for i in range(n_pairs):
        columns.append(f"Expected {i+1}")
        columns.append(f"Result {i+1}")

    for key in dataset:
        for item in dataset[key]:
            current_row.extend([item['answer'], item['formatted_response']])
            if len(current_row) == n_pairs * 2:  # Each row should have 2 * n_pairs columns
                table.append(current_row)
                current_row = []

    if current_row:  # Handle the last incomplete row
        current_row.extend([""] * ((n_pairs*2) - len(current_row)))  # Fill the rest of the row with empty strings
        table.append(current_row)

    
    df = pd.DataFrame(table, columns=columns)
    df = df.style.apply(lambda x: [style_answers(val, x.iloc[i-1]) if i % 2 else '' for i, val in enumerate(x)], axis=1)
    display(df)


def default_table_metric(expected, actual):
    return expected.strip() == actual.strip()


def style_answers(expected, actual, metric_fn=default_table_metric, true_color='lightgreen', false_color='lightcoral'):
    color = true_color if metric_fn(expected, actual) else false_color
    return f'background-color: {color}'