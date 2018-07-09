from starttf.models.simple_detector_v2 import create_model as huval_model
from starttf.layers.tile_2d import tile_2d

import tensorflow as tf


def create_model(input_tensor, mode, hyper_params):
    model = huval_model(input_tensor, mode, hyper_params)
    with tf.variable_scope("simple_detector"):
        num_classes = hyper_params.problem.number_of_directions

        # Classification
        model["direction_logits_raw"] = tf.layers.conv2d(inputs=model["conv7"], filters=num_classes * 64, kernel_size=(1, 1),
                                               strides=(1, 1),
                                               padding="same", activation=None, name="direction_logits_raw")
        model["direction_logits"] = tile_2d(model["direction_logits_raw"], 8, 8, name="direction_logits", reorder_required=True)
        model["direction_probs"] = tf.nn.softmax(model["direction_logits"], name="direction_probs")

    return model
