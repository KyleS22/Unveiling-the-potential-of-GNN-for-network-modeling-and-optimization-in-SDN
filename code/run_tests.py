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

#logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)

logger = logging.getLogger('test_runner')
logger.setLevel(logging.DEBUG)

logger.propagate = False

fh = logging.FileHandler('test_runner.log')
fh.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

fh.setFormatter(formatter)
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)

NUM_TESTS = 0
NUM_TEST_REMAINING = 0
AVG_RUNTIME = 0
DATASET_PATH = "../datasets/"
RESULTS_PATH = "../results/"

NUM_STEPS = 50000
CHECKPOINT_DIR = "./CheckPoints"

def make_hyperparam_string(l2, dropout_rate, link_state_dim, path_state_dim,
        readout_units, learning_rate, T):

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

    test_id = 1000
    completed_tests = load_completed_tests()

    hyperparam_string = make_hyperparam_string(0.1, 0.5, 32, 32,
                        256, 0.001, 8)

    # Get all the single train experiments
    for train in range(len(dataset_names)):
        for test in range(len(dataset_names)):

            if train != test:
                if test_id not in completed_tests:

                    train_set = dataset_names[train]
                    test_set = dataset_names[test]

                    train_set_path = DATASET_PATH + train_set
                    test_set_path = DATASET_PATH + test_set

                    run_single_test(test_id, train_set_path, None, test_set_path,  hyperparam_string)

                test_id +=1

    combos = itertools.combinations(dataset_names, 2)

    for combo in combos:
        for test in range(len(dataset_names)):
            if dataset_names[test] != combo[0] and dataset_names[test] != combo[1]:
                if test_id not in completed_tests:
                    train_set = combo[0]
                    train_set2 = combo[1]
                    test_set = dataset_names[test]

                    train_set_path = DATASET_PATH + train_set
                    train_set2_path = DATASET_PATH + train_set2
                    test_set_path = DATASET_PATH + test_set

                    run_single_test(test_id, train_set_path, train_set2_path,
                            test_set_path,
                            hyperparam_string)

                test_id += 1

def test2(hyperparams):

    completed_tests = load_completed_tests()
    test_id = 2000

    train_set = "nsfnetbw"
    train_set2 = "synth50bw"
    test_set = "geant2bw"

    train_set_path = DATASET_PATH + train_set
    train_set2_path = DATASET_PATH + train_set2
    test_set_path = DATASET_PATH + test_set

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
                                if test_id not in completed_tests:
                                    hyperparam_string = make_hyperparam_string(l2,
                                        dr, lsd, psd, ru, lr, T)

                                    run_single_test(test_id, train_set_path,
                                        train_set2_path, test_set_path,
                                        hyperparam_string)

                                test_id += 1

def calculate_num_tests(inputs_dict):

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
            if dataset_names[test] != combo[0] and dataset_names[test] != combo[1]:
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

def run_single_test(test_id, train_set, train_set2, test_set, hyperparam_string):

    out_dir = os.path.join(CHECKPOINT_DIR, str(test_id))

    # log start of test
    logger.info('Starting test ' + str(test_id))

    # start timer
    start_time = time.time()

    # call run experiment.sh with subprocess
    if train_set2 is None:
        res = subprocess.call(["sh", "./run_experiment.sh", "train", hyperparam_string,
            train_set, test_set, out_dir])
    else:
        logger.debug("Test Set: {}".format(test_set))
        res = subprocess.call(["sh", "./run_experiment.sh", "train_multiple",
            hyperparam_string, train_set, test_set, out_dir, train_set2])

    # Catch the result of subprocess and log an error if necessary
    if res != 0:
        logger.error("Error occured with test " + str(test_id))
        log_test_fail(test_id)
        return

    # stop timer
    runtime = time.time() - start_time


    global NUM_TESTS_REMAINING
    NUM_TESTS_REMAINING -= 1

    global AVG_RUNTIME
    AVG_RUNTIME = (AVG_RUNTIME + runtime)/2
    logger.debug("AVG_RUNTIME: " + str(AVG_RUNTIME))
    logger.debug("NUM_TESTS_REMAINING: " + str(NUM_TESTS_REMAINING))

    est_time_remaining = AVG_RUNTIME * NUM_TESTS_REMAINING

    est_seconds = int((est_time_remaining)%60)
    if est_seconds < 10:
        est_seconds = "0" + str(est_seconds)

    est_minutes = int((est_time_remaining/(60))%60)

    if est_minutes < 10:
        est_minutes = "0" + str(est_minutes)

    est_hours = int((est_time_remaining/(60 * 60))%24)


    if create_result_file(test_id, train_set, train_set2, test_set,
            hyperparam_string, CHECKPOINT_DIR):
        log_test_completion(test_id)
    else:
        log_test_fail(test_id)

    # log end of test, time it took and estimate time left
    logger.info('Test ' + str(test_id) + " finished.  Estimated time remaining: " +
            str(est_hours) + ":" + str(est_minutes) + ":" + str(est_seconds))


def create_result_file(test_id, train_set, train_set2, test_set,
        hyperparam_string, checkpoint_dir):

    eval_res_path = os.path.join(checkpoint_dir, str(test_id),  "eval")

    if not os.path.exists(eval_res_path):
        logger.error("The checkpoint path does not exist: " + eval_res_path)
        return False

    try:
        event_file = os.path.join(eval_res_path, os.listdir(eval_res_path)[0])
    except:
        logger.error("The event file does not exist: " + eval_res_path)
        return False

    outputs = {}

    for event in tf.train.summary_iterator(event_file):
        for value in event.summary.value:
            if value.HasField('simple_value'):
                outputs[value.tag] = value.simple_value


    hyperparam_parts = hyperparam_string.split(',')

    inputs = {}

    for part in hyperparam_parts:
        key = part.split('=')[0]
        value = part.split('=')[1]

        inputs[key] = value

    inputs["train_set"] = train_set
    inputs["train_set2"] = train_set2
    inputs["test_set"] = test_set
    inputs["num_steps"] = NUM_STEPS

    results_dict = {"inputs": inputs, "outputs": outputs}

    output_file = {test_id, results_dict}

    if not os.path.exists(RESULTS_PATH):
        os.mkdir(RESULTS_PATH)

    with open(os.path.join(RESULTS_PATH, test_id + ".json")) as outfile:
        json.dump(output_file, outfile)

    return True



def log_test_completion(test_id):
    with open("completed_tests.txt", 'a') as outfile:
        outfile.write(str(test_id) + "\n")


def log_test_fail(test_id):
    with open("failed_tests.txt", 'a') as outfile:
        outfile.write(str(test_id) + "\n")

    logger.error("Test " + str(test_id) + " failed. Written to log.")

def load_completed_tests():
    if os.path.exists("completed_tests.txt"):
        with open("completed_tests.txt", 'r') as infile:
            ids = infile.readlines()
            ids = [int(x.strip()) for x in ids]

        return ids
    return []

def run_tests(inputs_dict):

    global NUM_TESTS
    NUM_TESTS= calculate_num_tests(inputs_dict)
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


