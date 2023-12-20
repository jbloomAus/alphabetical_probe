from evals.spelling_by_grade import SpellingEvalDict
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
import string


def basic_bar_graph(data: Dict[int, int], ymin=0, ymax=1, title='', xtitle='', ytitle=''):
    """Creates a basic graph from a set of {x_1: y_1, ... x_n: y_n} data."""
    fig = go.Figure(go.Bar(x=list(data.keys()), y=list(data.values())))
    fig.update_layout(title=title, xaxis=dict(title=xtitle), yaxis=dict(range=[ymin, ymax], title=ytitle))
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
    

def create_confusion_matrix(data: Dict[Any, Dict], title='', xtitle='', ytitle=''):
    """Create a confusion matrix for a dictionary of dictionaries representing data."""

    characters = list(string.ascii_uppercase) + ['_'] + ['Other'] # _ is a blank space, 'Other' is miscellaneous.
    confusion_matrix = np.zeros((len(characters), len(characters)))

    # Mapping characters to indices
    char_to_index = {char: index for index, char in enumerate(characters)}

    # Populate the confusion matrix
    for answer_char, resp_dict in data.items():
        answer_index = char_to_index[answer_char]
        for resp_char, count in resp_dict.items():
            resp_index = char_to_index[resp_char]
            confusion_matrix[answer_index][resp_index] = count

    max_value = int(np.max(confusion_matrix))

    fig = ff.create_annotated_heatmap(
        z=confusion_matrix,
        zmin=-max_value, zmax=max_value,
        x=characters, y=characters, 
        annotation_text=confusion_matrix.astype(int),
        colorscale='RdBu',
        font_colors=['black']
    )

    fig.update_layout(
        title=title,
        xaxis=dict(title=xtitle),
        yaxis=dict(title=ytitle, autorange='reversed'),
        autosize=False, width=900, height=900
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.show()



def default_table_metric(expected, actual):
    """Return the metric for when an answer is correct in a table if no metric is provided."""
    return expected.strip() == actual.strip()


def style_answers(expected, actual, metric_fn=default_table_metric, true_color='lightgreen', false_color='lightcoral'):
    """Decides on background color of a table's cell based on whether the answer is correct or not."""
    color = true_color if metric_fn(expected, actual) else false_color
    return f'background-color: {color}'