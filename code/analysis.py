"""
File Name: analysis.py

Authors: Kyle Seidenthal

Date: 19-11-2019

Description: Do some analysis on the results

"""

import os
import json
import pandas as pd
import plotly.graph_objects as go
import sys
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates

RESULT_DIR = "../results/"


def analyze(test_prefix):
    """
    Perform some analysis on the tests that start with test_prefix

    :param test_prefix: The test number prefix, can be 1 or 2, corresponding to
                        generality tests and hyperparameter tests, repsectively
    :returns: None
    """

    result_files = get_result_files(test_prefix)

    if test_prefix == "1":
        data = read_test_1_results_to_pandas(result_files)

        create_parallel_coord_plot(data)
        create_bar_chart(data)

def get_result_files(test_prefix):
    """
    Get a list of the paths to result files

    :param test_prefix: The test prefix to get files for
    :returns: A list of paths to result files
    """

    result_paths = []

    for filename in os.listdir(RESULT_DIR):
        if filename.startswith(test_prefix):
            result_paths.append(os.path.join(RESULT_DIR, filename))

    return result_paths


def read_test_1_results_to_pandas(result_file_paths):
    """
    Reads in the list of files and converts them into a Pandas Dataframe

    :param result_file_path: A list of paths to result files
    :returns: A Pandas Dataframe representing the data
    """

    cols = ["train_set", "train_set2", "test_set", "label/mean", "loss", "mae",
            "mre", "prediction/mean", "rho"]

    data_lists = []

    for path in result_file_paths:

        with open(path) as json_file:
            data = json.load(json_file)
            test_id = os.path.split(path)[-1]
            test_id = os.path.splitext(test_id)[0]

            row = []

            for i in range(len(cols)):
                if "set" in cols[i]:
                    try:
                        data[test_id]["inputs"][cols[i]] =\
                            data[test_id]["inputs"][cols[i]].split("/")[-1]
                    except Exception as e:
                        e = None
                        e
                        pass
                try:
                    row.append(data[test_id]["inputs"][cols[i]])
                except Exception as e:
                    e = None
                    e
                    row.append(data[test_id]["outputs"][cols[i]])

            data_lists.append(row)

    df = pd.DataFrame(data_lists, columns=cols)
    df.replace(to_replace=[None], value="None", inplace=True)
    df['test_label'] = df[['train_set',
                           'train_set2',
                           'test_set']].apply(lambda x: '-'.join(x), axis=1)
    df = df.drop(columns=[x for x in cols if "set" in x], axis=1)

    return df


def create_bar_chart(data):
    data_copy = data.copy()
    data_copy = data_copy.set_index('test_label')
    print(data_copy)
    plt.figure()
    data_copy.plot.bar(rot=0)
    plt.savefig("data_bar_chart.png")

def create_parallel_coord_plot(data):
    """
    Creates a parallel coordinates plot from the data

    :param data: {% A parameter %}
    :returns: {% A thing %}
    """
    plt.figure()
    parallel_coordinates(data, 'test_label')
    plt.savefig("data_parcoord_plot.png")
    #fig = go.Figure(data=go.Parcoords(
    #    line=dict(
    #        color=data['index'],
    #        colorscale='Electric',
    #        showscale=True,
    #        cmin=0,
    #        cmax=len(data)),
    #    dimensions=list([
    #                dict(
    #                    label='Label/Mean', values=data['label/mean']),
    #                dict(
    #                    label='Loss', values=data['loss']),
    #                dict(
    #                    label='MAE', values=data['mae']),
    #                dict(
    #                    label='MRE', values=data['mre']),
    #                dict(
    #                    label='Prediction Mean',
    #                    values=data['prediction/mean']),
    #                dict(
    #                    label='Streaming Pearson Correlation',
    #                    values=data['rho']),
    #                ])
    #    ))

    #fig.update_layout(
    #        plot_bgcolor='white',
    #        paper_bgcolor='white'
    #        )
    #print("Showing plot")
    #fig.show()


if __name__ == "__main__":

    test_prefix = sys.argv[1]

    analyze(test_prefix)
