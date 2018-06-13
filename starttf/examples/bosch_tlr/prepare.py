import os
import numpy as np
import math
import random

from starttf.utils.hyperparams import load_params
from starttf.tfrecords.autorecords import write_data
from starttf.utils.image_manipulation import crop

from opendatalake.detection.bosch_tlr import bosch_tlr
from opendatalake.texture_augmentation import full_texture_augmentation


def preprocess_feature(hyper_params, feature):
    feature["image"] = full_texture_augmentation(feature["image"])
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


def move_detections(label, dy, dx):
    for k in label.keys():
        detections = label[k]
        for detection in detections:
            detection.cy += dy
            detection.cx += dx


def augment_data(hyper_params, feature, label):
    # Do not augment these ways:
    # 1) Rotation is not possible
    # 3) Scaling is not possible, because it ruins depth perception
    # However, random crops can improve performance. (Training speed and accuracy)
    if hyper_params.problem.get_or_default("augmentation", None) is None:
        return feature, label

    img_h, img_w, img_c = feature["image"].shape
    augmented_feature = {}
    augmented_label = {}
    augmented_feature["image"] = feature["image"].copy()

    for k in label.keys():
        augmented_label[k] = [detection.copy() for detection in label[k]]

    if hyper_params.problem.augmentation.get_or_default("random_crop", None) is not None and len(augmented_label["detections_2d"]) != 0:
        img_h, img_w, img_c = augmented_feature["image"].shape
        target_w = hyper_params.problem.augmentation.random_crop.shape.width
        target_h = hyper_params.problem.augmentation.random_crop.shape.height

        idx = random.randint(0, len(augmented_label["detections_2d"]) - 1)
        detection = augmented_label["detections_2d"][idx]

        # Compute start point so that crop fit's into image and random crop contains detection
        start_x = int(detection.cx - random.random() * (target_w-20) / 2.0 - 10)
        start_y = int(detection.cy - random.random() * (target_h-20) / 2.0 - 10)
        if start_x < 0:
            start_x = 0
        if start_y < 0:
            start_y = 0
        if start_x >= img_w - target_w:
            start_x = img_w - target_w - 1
        if start_y >= img_h - target_h:
            start_y = img_h - target_h - 1

        # Crop image
        augmented_feature["image"] = crop(augmented_feature["image"], start_y, start_x, target_h, target_w)

        # Crop labels
        move_detections(augmented_label, -start_y, -start_x)

    return augmented_feature, augmented_label


if __name__ == "__main__":
    # Load the hyper parameters.
    hyper_params = load_params("D:/Data/Projects/starttf/starttf/examples/bosch_tlr/hyper_params.json")

    # Get a generator and its parameters
    train_gen, train_gen_params = bosch_tlr(base_dir=hyper_params.problem.data_path, phase="train")
    validation_gen, validation_gen_params = bosch_tlr(base_dir=hyper_params.problem.data_path, phase="validation")

    # Create the paths where to write the records from the hyper parameter file.
    train_record_path = os.path.join(hyper_params.train.tf_records_path, "train")
    validation_record_path = os.path.join(hyper_params.train.tf_records_path, "validation")

    # Write the data
    write_data(hyper_params, train_record_path, train_gen, train_gen_params, 8, preprocess_feature=preprocess_feature, preprocess_label=preprocess_label, augment_data=augment_data)
    write_data(hyper_params, validation_record_path, validation_gen, validation_gen_params, 8, preprocess_feature=preprocess_feature, preprocess_label=preprocess_label, augment_data=augment_data)
