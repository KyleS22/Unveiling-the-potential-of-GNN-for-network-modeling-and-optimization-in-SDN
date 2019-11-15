"""
File Name: run_tests.py

Authors: Kyle Seidenthal

Date: 13-11-2019

Description: A script to run a bunch of tests on the model

"""
import logging
import numpy as np
import time
import itertools
import tensorflow as tf
import os
import json
import subprocess

# Set up logging
logger = logging.getLogger('test_runner')
logger.setLevel(logging.DEBUG)

logger.propagate = False

fh = logging.FileHandler('test_runner.log')
fh.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - \
        %(levelname)s - %(message)s')

fh.setFormatter(formatter)
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)

# Some global variables to keep track of things

# The total number of tests to run
NUM_TESTS = 0

# The number of tests left to be run
NUM_TEST_REMAINING = 0

# The average runtime of one test
AVG_RUNTIME = 0

# The path to where the datasets live
DATASET_PATH = "../datasets/"

# The path to the place to store the results
RESULTS_PATH = "../results/"

# The number of steps to train for
NUM_STEPS = 50000

# The directory that model checkpoints will be stored in
CHECKPOINT_DIR = "./CheckPoints"


def make_hyperparam_string(l2, dropout_rate, link_state_dim, path_state_dim,
                           readout_units, learning_rate, T):
    """
    Creates a string that represents the input to the --hparam option in
    routenet_with_link_cap.py

    :param l2: The l2 loss parameter
    :param dropout_rate: The dropout rate.  A float between 0 and 1
    :param link_state_dim: The dimensions for the link state
    :param path_state_dim: The dimensions for the path state
    :param readout_units: The number of readout units
    :param learning_rate: The learning rate
    :param T: T
    :returns: A string in the correct format to be passed to the --hparam
              option in routenet_with_link_cap.py
    """

    out_str = ""

    out_str += "l2=" + str(l2)
    out_str += ",dropout_rate=" + str(dropout_rate)
    out_str += ",link_state_dim=" + str(link_state_dim)
    out_str += ",path_state_dim=" + str(path_state_dim)
    out_str += ",readout_units=" + str(readout_units)
    out_str += ",learning_rate=" + str(learning_rate)
    out_str += ",T=" + str(T)

    return out_str


def test1(dataset_names):
    """
    Run the first set of tests

    :param dataset_names: a list of the names of the datasets that are
                          possible to use.
    :returns: None
    """

    test_id = 1000
    completed_tests = load_completed_tests()

    hyperparam_string = make_hyperparam_string(0.1, 0.5, 32, 32,
                                               256, 0.001, 8)

    # Get all the single train experiments
    for train in range(len(dataset_names)):
        for test in range(len(dataset_names)):

            if train != test:
                # Make sure we have not already tried this test previously
                if test_id not in completed_tests:

                    train_set = dataset_names[train]
                    test_set = dataset_names[test]

                    train_set_path = DATASET_PATH + train_set
                    test_set_path = DATASET_PATH + test_set

                    logger.debug(str(test_id) + " " + train_set_path + " " +
                                 test_set_path + " " + hyperparam_string)

                    run_single_test(test_id, train_set_path, None,
                                    test_set_path,  hyperparam_string)
                else:
                    logger.info("Test " + str(test_id) +
                                " already complete. Skipping.")
                test_id += 1

    # Get all the double train experiments
    combos = itertools.combinations(dataset_names, 2)

    for combo in combos:
        for test in range(len(dataset_names)):
            if (dataset_names[test] != combo[0] and
                    dataset_names[test] != combo[1]):

                # Make sure we have not already run this test
                if test_id not in completed_tests:
                    train_set = combo[0]
                    train_set2 = combo[1]
                    test_set = dataset_names[test]

                    train_set_path = DATASET_PATH + train_set
                    train_set2_path = DATASET_PATH + train_set2
                    test_set_path = DATASET_PATH + test_set

                    logger.debug(str(test_id) + " " + train_set_path + " " +
                                 test_set_path + " " + hyperparam_string)

                    run_single_test(test_id, train_set_path, train_set2_path,
                                    test_set_path, hyperparam_string)
                else:
                    logger.info("Test " + str(test_id) +
                                " already complete. Skipping.")

                test_id += 1


def test2(hyperparams):
    """
    Run all the second test experiments

    :param hyperparams: The hyperparameter dictionary, which specifies tuples
                        of (min, max, step) for each hyperparam.
    :returns: None
    """

    completed_tests = load_completed_tests()
    test_id = 2000

    train_set = "nsfnetbw"
    train_set2 = "synth50bw"
    test_set = "geant2bw"

    train_set_path = DATASET_PATH + train_set
    train_set2_path = DATASET_PATH + train_set2
    test_set_path = DATASET_PATH + test_set

    # Parse the hyperparam inputs to create lists of values to try
    l2_range = np.arange(hyperparams["L2"][0], hyperparams["L2"][1],
                         hyperparams["L2"][2])

    dr_range = np.arange(hyperparams["drop_rate"][0],
                         hyperparams["drop_rate"][1],
                         hyperparams["drop_rate"][2])

    lsd_range = np.arange(hyperparams["link_state_dim"][0],
                          hyperparams["link_state_dim"][1],
                          hyperparams["link_state_dim"][2])

    psd_range = np.arange(hyperparams["path_state_dim"][0],
                          hyperparams["path_state_dim"][1],
                          hyperparams["path_state_dim"][2])

    ru_range = np.arange(hyperparams["readout_units"][0],
                         hyperparams["readout_units"][1],
                         hyperparams["readout_units"][2])

    lr_range = np.arange(hyperparams["learning_rate"][0],
                         hyperparams["learning_rate"][1],
                         hyperparams["learning_rate"][2])

    T_range = np.arange(hyperparams["T"][0], hyperparams["T"][1],
                        hyperparams["T"][2])

    # Try every possible combination of the specified values for each
    # parameter in a big ugly loop
    for l2 in l2_range:
        for dr in dr_range:
            for lsd in lsd_range:
                for psd in psd_range:
                    for ru in ru_range:
                        for lr in lr_range:
                            for T in T_range:

                                # Make sure we have not already run this test
                                if test_id not in completed_tests:
                                    hyperparam_string = \
                                            make_hyperparam_string(l2, dr,
                                                                   lsd, psd,
                                                                   ru, lr, T)

                                    run_single_test(test_id, train_set_path,
                                                    train_set2_path,
                                                    test_set_path,
                                                    hyperparam_string)
                                else:
                                    logger.info("Test " + str(test_id) +
                                                " already complete.\
                                                Skipping.")

                                test_id += 1


def calculate_num_tests(inputs_dict):
    """
    Calculate the total number of tests that will be run

    :param inputs_dict: The dictionary of input values
    :returns: The total number of tests that will be run
    """

    num_tests = 0
    dataset_names = inputs_dict["dataset_names"]

    # Get all the single train experiments
    for train in range(len(dataset_names)):
        for test in range(len(dataset_names)):

            if train != test:
                num_tests += 1

    # Get all double train experiments

    combos = itertools.combinations(dataset_names, 2)

    for combo in combos:
        for test in range(len(dataset_names)):
            if (dataset_names[test] != combo[0] and
                    dataset_names[test] != combo[1]):
                    num_tests += 1

    hyperparams = inputs_dict["hyperparams"]

    l2_range = np.arange(hyperparams["L2"][0], hyperparams["L2"][1],
                         hyperparams["L2"][2])

    dr_range = np.arange(hyperparams["drop_rate"][0],
                         hyperparams["drop_rate"][1],
                         hyperparams["drop_rate"][2])

    lsd_range = np.arange(hyperparams["link_state_dim"][0],
                          hyperparams["link_state_dim"][1],
                          hyperparams["link_state_dim"][2])

    psd_range = np.arange(hyperparams["path_state_dim"][0],
                          hyperparams["path_state_dim"][1],
                          hyperparams["path_state_dim"][2])

    ru_range = np.arange(hyperparams["readout_units"][0],
                         hyperparams["readout_units"][1],
                         hyperparams["readout_units"][2])

    lr_range = np.arange(hyperparams["learning_rate"][0],
                         hyperparams["learning_rate"][1],
                         hyperparams["learning_rate"][2])

    T_range = np.arange(hyperparams["T"][0], hyperparams["T"][1],
                        hyperparams["T"][2])

    for l2 in l2_range:
        for dr in dr_range:
            for lsd in lsd_range:
                for psd in psd_range:
                    for ru in ru_range:
                        for lr in lr_range:
                            for T in T_range:

                                num_tests += 1

    return num_tests


def run_single_test(test_id, train_set, train_set2, test_set,
                    hyperparam_string):
    """
    Run a single test with the given parameters

    :param test_id: The id for the test, an integer
    :param train_set: The path to the training set to use
    :param train_set2: The path to the second training set to use.  Can be
                       None.
    :param test_set: The path to the test set to use
    :param hyperparam_string: The hyperparam string to use with
                              routenet_with_link_cap.py
    :returns: None
    """

    out_dir = os.path.join(CHECKPOINT_DIR, str(test_id))

    # log start of test
    logger.info('Starting test ' + str(test_id))

    # start timer
    start_time = time.time()

    # call run experiment.sh with subprocess
    if train_set2 is None:
        res = subprocess.call(["sh", "./run_experiment.sh", "train",
                               hyperparam_string,
                               train_set, test_set, out_dir])
    else:
        logger.debug("Test Set: {}".format(test_set))
        res = subprocess.call(["sh", "./run_experiment.sh", "train_multiple",
                               hyperparam_string, train_set, test_set,
                               out_dir, train_set2])
    # Catch the result of subprocess and log an error if necessary
    if res != 0:
        logger.error("Error occured with test " + str(test_id))
        log_test_fail(test_id)
        return

    # need to give tensorflow a minute to clear memory
    time.sleep(60)
    # stop timer
    runtime = time.time() - start_time

    # Update logs and get estimated total runtime rematining
    global NUM_TESTS_REMAINING
    NUM_TESTS_REMAINING -= 1

    global AVG_RUNTIME
    AVG_RUNTIME = (AVG_RUNTIME + runtime)/2
    logger.debug("AVG_RUNTIME: " + str(AVG_RUNTIME))
    logger.debug("NUM_TESTS_REMAINING: " + str(NUM_TESTS_REMAINING))

    est_time_remaining = AVG_RUNTIME * NUM_TESTS_REMAINING

    est_seconds = int((est_time_remaining) % 60)
    if est_seconds < 10:
        est_seconds = "0" + str(est_seconds)

    est_minutes = int((est_time_remaining / (60)) % 60)

    if est_minutes < 10:
        est_minutes = "0" + str(est_minutes)

    est_hours = int((est_time_remaining / (60 * 60)) % 24)

    # Create the result file and log success if it worked
    if create_result_file(test_id, train_set, train_set2, test_set,
                          hyperparam_string, CHECKPOINT_DIR):

        log_test_completion(test_id)

    # Otherwise, mark this one down as a failure
    else:
        log_test_fail(test_id)

    # log end of test, time it took and estimate time left
    logger.info('Test ' + str(test_id) +
                " finished.  Estimated time remaining: " +
                str(est_hours) + ":" + str(est_minutes) + ":" +
                str(est_seconds))


def create_result_file(test_id, train_set, train_set2, test_set,
                       hyperparam_string, checkpoint_dir):
    """
    Create the results file for this test

    :param test_id: The id of the test
    :param train_set: The path to the training set
    :param train_set2: The path to the second training set
    :param test_set: The path to the testing set
    :param hyperparam_string: The hyperparam string used
    :param checkpoint_dir: The path to the directory containin the model
                           checkpoints
    :returns: True if the file is created successfully
    """

    # Get the full result path
    eval_res_path = os.path.join(checkpoint_dir, str(test_id),  "eval")

    # Make sure the checkpoint directory exists
    if not os.path.exists(eval_res_path):
        logger.error("The checkpoint path does not exist: " + eval_res_path)
        return False

    # try to get the event file.  It should be the only filr in the checkpoint
    # eval directory
    try:
        event_file = os.path.join(eval_res_path, os.listdir(eval_res_path)[0])
    except Exception as e:
        logger.error("The event file does not exist: " + eval_res_path)
        return False

    outputs = {}

    # Parse the event file for the results we want
    for event in tf.train.summary_iterator(event_file):
        for value in event.summary.value:
            if value.HasField('simple_value'):
                outputs[value.tag] = value.simple_value

    hyperparam_parts = hyperparam_string.split(',')

    # Split the hyperparam string to get the inputs we used
    inputs = {}

    for part in hyperparam_parts:
        key = part.split('=')[0]
        value = part.split('=')[1]

        inputs[key] = value

    inputs["train_set"] = train_set
    inputs["train_set2"] = train_set2
    inputs["test_set"] = test_set
    inputs["num_steps"] = NUM_STEPS

    # Create the results dictionary
    results_dict = {"inputs": inputs, "outputs": outputs}

    # Add the test ID as the first key
    output_file = {test_id: results_dict}

    # Make sure the results path exists
    if not os.path.exists(RESULTS_PATH):
        os.mkdir(RESULTS_PATH)

    # Write out the JSON
    with open(os.path.join(RESULTS_PATH, str(test_id) + ".json"),
              'w') as outfile:
        json.dump(output_file, outfile)

    return True


def log_test_completion(test_id):
    """
    Add the test ID to a file containing completed tests

    :param test_id: The ID of the test to mark as complete
    :returns: None
    """
    with open("completed_tests.txt", 'a') as outfile:
        outfile.write(str(test_id) + "\n")


def log_test_fail(test_id):
    """
    Add the test ID to a file containinf failed tests

    :param test_id: The ID of the test to mark as failed
    :returns: None
    """
    with open("failed_tests.txt", 'a') as outfile:
        outfile.write(str(test_id) + "\n")

    logger.error("Test " + str(test_id) + " failed. Written to log.")


def load_completed_tests():
    """
    Load the completed tests file

    :returns: A list of test IDs that have been marked completed
    """
    if os.path.exists("completed_tests.txt"):
        with open("completed_tests.txt", 'r') as infile:
            ids = infile.readlines()
            ids = [int(x.strip()) for x in ids]

        return ids
    return []


def run_tests(inputs_dict):
    """
    Run the tests with the given inputs

    :param inputs_dict: A dictionary representing the inputs to use for tests
    :returns: None
    """

    global NUM_TESTS
    NUM_TESTS = calculate_num_tests(inputs_dict)
    logger.debug("NUM_TESTS: " + str(NUM_TESTS))
    global NUM_TESTS_REMAINING
    NUM_TESTS_REMAINING = NUM_TESTS

    test1(inputs_dict["dataset_names"])

    test2(inputs_dict["hyperparams"])


if __name__ == "__main__":

    inputs_dict = {
            "dataset_names": ["geant2bw", "nsfnetbw", "synth50bw"],
            # TUPLES are (min, max, step)
            # NOTE min is inclusive, max is exclusive
            "hyperparams": {
                "L2": (0.1, 0.2, 0.1),
                "drop_rate": (0.0, 1.0 + 0.2, 0.2),
                "link_state_dim": (8, 64 + 8, 8),
                "path_state_dim": (8, 64 + 8, 8),
                "readout_units": (256, 257, 1),
                "learning_rate": (0.001, 0.002, 0.001),
                "T": (8, 8 + 1, 1)
                }
            }

    run_tests(inputs_dict)
