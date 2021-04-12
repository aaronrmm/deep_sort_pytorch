import os
import os.path as osp
import logging
import argparse
from pathlib import Path

from utils.log import get_logger
from deepsort import VideoTracker
from utils.parser import get_config

import motmetrics as mm

mm.lap.default_solver = "lap"
from utils.evaluation import Evaluator


def mkdir_if_missing(dir):
    os.makedirs(dir, exist_ok=True)


def main(sequence_data, output_path, args=""):
    logger = get_logger()
    logger.setLevel(logging.INFO)
    data_type = "mot"

    cfg = get_config()

    # run tracking
    accs = []
    for seq in sequence_data.keys():
        logger.info("start seq: {}".format(seq))
        groundtruth_filename = sequence_data[seq]["gt"]
        assert os.path.isfile(
            groundtruth_filename
        ), f"No ground truth at {os.path.abspath(groundtruth_filename)}"
        result_filename = sequence_data[seq]["prediction"]
        assert os.path.isfile(
            result_filename
        ), f"No results at {os.path.abspath(result_filename)}"

        # eval
        logger.info("Evaluate seq: {}".format(seq))
        evaluator = Evaluator(data_type, groundtruth_filename)
        accs.append(evaluator.eval_file(result_filename))

    # get summary
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, list(sequence_data.keys()), metrics)
    strsummary = mm.io.render_summary(
        summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)
    Evaluator.save_summary(summary, output_path)


def parse_args():
    parser = argparse.ArgumentParser()
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Get predictions
    prediction_root = "./advanced_visdrone_results"
    predicted_sequences = {}
    with open(os.path.join(prediction_root, "sequences.txt"), "r") as sequence_file:
        for line in sequence_file.readlines():
            sequence_id, sequence_prediction_file = line.split(" ")
            predicted_sequences[sequence_id] = sequence_prediction_file.strip()
    # Get ground truths
    ground_truth_root = "/mnt/z/Data/AerialCars/VisDrone2019-MOT-test-dev"
    ground_truth_sequences = {}
    with open(os.path.join(ground_truth_root, "sequences.txt"), "r") as sequence_file:
        for line in sequence_file.readlines():
            sequence_id, sequence_prediction_file = line.split(" ")
            ground_truth_sequences[sequence_id] = sequence_prediction_file.strip()
    for sequence_id in predicted_sequences.keys():
        assert (
            sequence_id in ground_truth_sequences.keys()
        ), f"No ground truth file listed for sequence {sequence_id}."
    # Combine
    sequence_data = {
        sequence_id: {
            "gt": os.path.join(ground_truth_root, ground_truth_sequences[sequence_id]),
            "prediction": os.path.join(
                prediction_root, predicted_sequences[sequence_id]
            ),
        }
        for sequence_id in predicted_sequences.keys()
    }
    main(sequence_data, os.path.join(prediction_root, "summary_global.xlsx"))
