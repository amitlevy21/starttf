import sys
import os
import numpy as np
import math

from starttf.utils.hyperparams import load_params
from starttf.tfrecords.autorecords import write_data

from opendatalake.detection.bosch_tlr import bosch_tlr
from opendatalake.detection.utils import augment_detections


def preprocess_feature(hyper_params, feature):
    return feature


def preprocess_label(hyper_params, feature, label):
    img_h, img_w, img_c = feature["image"].shape
    region_size = 4.0
    class_mapping = hyper_params.problem.class_mapping.__dict__
    direction_mapping = hyper_params.problem.direction_mapping.__dict__

    # Determine shape of output layer given input
    up_scale = int(32.0 / region_size)
    feature_h, feature_w = int(math.ceil(img_h / 32.0)) * up_scale, int(math.ceil(img_w / 32.0)) * up_scale

    # Create output labels
    processed_label = {}
    processed_label["class_id"] = np.zeros(shape=(feature_h, feature_w, 1), dtype=np.uint8)
    processed_label["direction"] = np.zeros(shape=(feature_h, feature_w, 1), dtype=np.uint8)
    processed_label["rect"] = np.zeros(shape=(feature_h, feature_w, 4), dtype=np.float32)

    # Set everything to background
    processed_label["class_id"][:, :, 0] = 0
    processed_label["direction"][:, :, 0] = hyper_params.problem.number_of_directions

    cells_occupied = 0
    for detection in label["detections_2d"]:
        # If background there is nothing to do
        if detection.class_id == 0:
            continue

        cx, cy = detection.cx, detection.cy
        w, h, = detection.w, detection.h
        x1 = cx - w / 2.0 * hyper_params.problem.shrink
        x2 = cx + w / 2.0 * hyper_params.problem.shrink
        y1 = cy - h / 2.0 * hyper_params.problem.shrink
        y2 = cy + h / 2.0 * hyper_params.problem.shrink

        for x in range(int(round(x1 / region_size)), int(round(x2 / region_size)), 1):
            for y in range(int(round(y1 / region_size)), int(round(y2 / region_size)), 1):
                anchor_x = x * region_size
                anchor_y = y * region_size

                if 0 <= x < feature_w and 0 <= y < feature_h:
                    cells_occupied += 1
                    processed_label["class_id"][y][x][0] = class_mapping[detection.class_id]
                    processed_label["direction"][y][x][0] = direction_mapping[detection.class_id]

                    # Set rect correctly
                    processed_label["rect"][y][x][0] = detection.cx - anchor_x
                    processed_label["rect"][y][x][1] = detection.cy - anchor_y
                    processed_label["rect"][y][x][2] = detection.w
                    processed_label["rect"][y][x][3] = detection.h

    if cells_occupied == 0 and hyper_params.problem.omit_not_annotated:
        return None
    return processed_label


if __name__ == "__main__":
    # Load the hyper parameters.
    hyper_params_path = "starttf/examples/bosch_tlr/hyper_params.json"
    if len(sys.argv) > 1:
        hyper_params_path = sys.argv[1] + "/" + hyper_params_path
    hyper_params = load_params(hyper_params_path)

    # Get a generator and its parameters
    train_gen, train_gen_params = bosch_tlr(base_dir=hyper_params.problem.data_path, phase="train")
    validation_gen, validation_gen_params = bosch_tlr(base_dir=hyper_params.problem.data_path, phase="validation")

    # Create the paths where to write the records from the hyper parameter file.
    train_record_path = os.path.join(hyper_params.train.tf_records_path, "train")
    validation_record_path = os.path.join(hyper_params.train.tf_records_path, "validation")

    # Write the data
    write_data(hyper_params, train_record_path, train_gen, train_gen_params, 8, preprocess_feature=preprocess_feature, preprocess_label=preprocess_label, augment_data=augment_detections)
    write_data(hyper_params, validation_record_path, validation_gen, validation_gen_params, 8, preprocess_feature=preprocess_feature, preprocess_label=preprocess_label, augment_data=augment_detections)
