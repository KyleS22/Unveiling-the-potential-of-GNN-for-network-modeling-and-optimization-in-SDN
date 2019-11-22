"""
File Name: analysis.py

Authors: Kyle Seidenthal

Date: 19-11-2019

Description: Do some analysis on the results

"""

import os
import json
import pandas as pd
import sys
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates

import matplotlib.style as style

style.use('ggplot')

# The path to where the result.json files are stored
RESULT_DIR = "../results/"


def analyze(test_prefix):
    """
    Perform some analysis on the tests that start with test_prefix

    :param test_prefix: The test number prefix, can be 1 or 2, corresponding to
                        generality tests and hyperparameter tests, repsectively
    :returns: None
    """

    # Get the paths to the result files
    result_files = get_result_files(test_prefix)

    # Handle the first set of tests
    if test_prefix == "1":

        data = read_test_1_results_to_pandas(result_files)

        output_data_to_table(data, "test1_table.tex")

        create_parallel_coord_plot(data)
        create_test1_bar_chart(data)


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

    # Make sure the test_label column comes first
    cols = list(df)
    cols.insert(0, cols.pop(cols.index('test_label')))

    df = df.ix[:, cols]

    return df


def output_data_to_table(data, name):
    """
    Output the given dataframe to a .tex file formatted as a table

    :param data: The dataframe containing the data
    :param name: The name for the tex file
    :returns: None
    """

    create_figures_dir()

    out_path = os.path.join(RESULT_DIR, "figures", name)

    with open(out_path, 'w') as outfile:
        outfile.write(data.to_latex())


def create_figures_dir():
    """
    Create a figures directory in the results folder, if not already there

    :returns: None
    """

    figs_path = os.path.join(RESULT_DIR, "figures")

    if not os.path.exists(figs_path):
        os.mkdir(figs_path)


def create_test1_bar_chart(data):
    """
    Create a bar chart formatted for data from test1

    :param data: The dataframe with the test1 data
    :returns: None
    """

    create_figures_dir()

    data_copy = data.copy()
    data_copy = data_copy.set_index('test_label')
    data_copy = data_copy.drop(['label/mean', 'mre'], axis=1)
    plt.figure()

    data_copy = data_copy.sort_values('loss', ascending=False)
    data_copy.plot.bar(rot=10, figsize=(10, 6),
                       title="Relationships of Results with Different " +
                             "Training Sets",
                       colormap="RdYlGn")

    out_path = os.path.join(RESULT_DIR, "figures", "test1_bar_chart.png")

    plt.savefig(out_path)


def create_parallel_coord_plot(data):
    """
    Creates a parallel coordinates plot from the data

    :param data: {% A parameter %}
    :returns: {% A thing %}
    """

    create_figures_dir()

    plt.figure(figsize=(10, 6))
    plt.title("Parallel Coordinates Plot for Results With Different " +
              "Training Sets")

    parallel_coordinates(data, 'test_label', colormap="RdYlGn")

    out_path = os.path.join(RESULT_DIR, "figures", "test1_parcoord_plot.png")

    plt.savefig(out_path)


if __name__ == "__main__":

    test_prefix = sys.argv[1]

    analyze(test_prefix)
