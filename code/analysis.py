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

from matplotlib.colors import LogNorm

import matplotlib.style as style
import numpy as np
import seaborn as sns
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

        create_test1_parallel_coord_plot(data)
        create_test1_bar_chart(data)

    elif test_prefix == "2":

        data = read_test_2_results_to_pandas(result_files)
        data = reduce_df(data)
        output_data_to_table(data, "test2_table.tex")
        create_test2_charts(data)
        create_test2_heatmap(data)
    else:
        print("Invalid test prefix")


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
    ax = data_copy.plot.bar(rot=10, figsize=(10, 8),
                       title="Relationships of Results with Different " +
                             "Training Sets",
                       colormap="RdYlGn")

    ax.set_ylim(-1, 5)
    ax.set_xlabel("Datasets (Train1 - Train2 - Eval)")
    out_path = os.path.join(RESULT_DIR, "figures", "test1_bar_chart.png")

    plt.savefig(out_path)
    plt.cla()

def create_test1_parallel_coord_plot(data):
    """
    Creates a parallel coordinates plot from the data

    :param data: {% A parameter %}
    :returns: {% A thing %}
    """

    create_figures_dir()

    plt.figure(figsize=(10, 6))
    plt.title("Parallel Coordinates Plot for Results With Different " +
              "Training Sets")

    plt.ylim(-1, 5)

    parallel_coordinates(data, 'test_label', colormap="RdYlGn")

    out_path = os.path.join(RESULT_DIR, "figures", "test1_parcoord_plot.png")

    plt.savefig(out_path)
    plt.cla()

def read_test_2_results_to_pandas(result_file_paths):
    """
    Reads in the list of files and converts them into a Pandas Dataframe

    :param result_file_path: A list of paths to result files
    :returns: A Pandas Dataframe representing the data
    """

    cols = ["dropout_rate", "link_state_dim", "path_state_dim", "label/mean",
            "loss", "mae", "mre", "prediction/mean", "rho"]

    data_lists = []

    for path in result_file_paths:

        with open(path) as json_file:
            data = json.load(json_file)
            test_id = os.path.split(path)[-1]
            test_id = os.path.splitext(test_id)[0]

            row = []

            for i in range(len(cols)):

                try:
                    row.append(data[test_id]["inputs"][cols[i]])
                except Exception as e:
                    e = None
                    e
                    row.append(data[test_id]["outputs"][cols[i]])

            data_lists.append(row)

    df = pd.DataFrame(data_lists, columns=cols)
    df.replace(to_replace=[None], value="None", inplace=True)

    df = df.sort_values(by=['dropout_rate', 'link_state_dim', 'path_state_dim',
                            'mae'])

    return df


def reduce_df(data, dropout_rate=0.6):
    """
    Reduce the dataframe to a smaller dataframe containing only entries with
    the given dropout_rate

    :param data: The dataframe to reduce
    :param dropout_rate: The dropout rate to use  The default value is 0.5.
    :returns: The reduced dropout rate
    """
    print(dropout_rate)
    print(data.dtypes)




    reduced = data.loc[round(data['dropout_rate'].astype(str).astype(float),2) == dropout_rate]

    print(reduced)
    reduced = reduced.sort_values(by=['mae'])

    return reduced


def create_test2_charts(data):
    """
    Create bar charts for the data in test2

    :param data: A dataframe containing the data
    :returns: None
    """

    # Get the unique link_state values
    link_state_dims = set(data['link_state_dim'].tolist())

    for link_state_dim in link_state_dims:
        df = data.loc[data['link_state_dim'] == link_state_dim]

        create_test2_bar_chart(df)
        create_test2_parcoords(df)

def create_test2_bar_chart(data):
    """
    Create a single bar chart for the test2 case

    :param data: A dataframe for a single link_state_dim
    :returns: None


    """
    create_figures_dir()

    # all the link state dims should be the same, so we just grab one
    link_state_dim = str(data['link_state_dim'].tolist()[0])

    # The integers are strings for some reason
    data = data.astype({'path_state_dim': 'int32'})
    data = data.drop(['label/mean', 'mre'], axis=1)
    data = data.set_index('path_state_dim')

    plt.figure()

    data = data.sort_values('path_state_dim', ascending=True)
    ax = data.plot.bar(rot=0, figsize=(10, 6),
                       title="Results for Link State Dim = " + link_state_dim,
                       colormap="RdYlGn")

    ax.set_ylim(-1, 5)
    file_name = "test2_bar_chart_" + link_state_dim + ".png"
    out_path = os.path.join(RESULT_DIR, "figures", file_name)

    plt.savefig(out_path)
    plt.cla()


def create_test2_parcoords(data):
    """
    Create a parallel coordinates plot for the test2 case

    :param data: A dataframe
    :returns: None


    """

    create_figures_dir()

    # Get link_state_dim
    link_state_dim = str(data['link_state_dim'].tolist()[0])

    # Convert path_state_dim to int32 for sorting
    data = data.astype({'path_state_dim': 'int32'})

    data = data.drop(['dropout_rate',
                      'link_state_dim'], axis=1)

    data = data.sort_values('path_state_dim', ascending=True)

    plt.figure(figsize=(10, 6))
    plt.ylim(-1, 2.5)
    plt.title("Path-State-Dim behaviour for Link-State-Dim = " +
              link_state_dim)

    parallel_coordinates(data, 'path_state_dim', colormap="RdYlGn")

    file_name = "test2_parcoord_plot_" + link_state_dim + ".png"
    out_path = os.path.join(RESULT_DIR, "figures", file_name)

    plt.savefig(out_path)
    plt.cla()

def create_test2_heatmap(data):

    data = data.drop(['dropout_rate', 'label/mean', 'loss', 'prediction/mean',
        'rho', 'mre'], axis=1)
    data = data.astype({"path_state_dim": int, "link_state_dim": int, "mae":
        np.float32})


    Index = [i for i in range(8, 64 + 8, 8)]
    Cols = Index

    mae_data = np.zeros((8, 8))

    for _, row in data.iterrows():
        link_state_dim = int(row['link_state_dim'])
        path_state_dim = int(row['path_state_dim'])
        mae = row['mae']

        mae_data[int((link_state_dim / 8) - 1)][int((path_state_dim / 8) - 1)] = mae

    heat_df = pd.DataFrame(mae_data, index=Index, columns=Cols)

    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(heat_df, annot=True, linewidths=.5,
            norm=LogNorm(np.min(mae_data),
        np.max(mae_data)))

    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    ax.xaxis.tick_top()
    ax.set_title("Heatmap of MAE for Different Values of path_state_dim and"+
            "link_state_dim")

    ax.set_xlabel("link_state_dim")
    ax.set_ylabel("path_state_dim")

    file_name = "test2_heatmap.png"
    out_path = os.path.join(RESULT_DIR, "figures", file_name)


    plt.savefig(out_path)

if __name__ == "__main__":

    test_prefix = sys.argv[1]

    analyze(test_prefix)
