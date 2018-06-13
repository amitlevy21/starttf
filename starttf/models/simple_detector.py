from starttf.models.vgg16_encoder import create_model as vgg_model
from starttf.layers.tile_2d import tile_2d

import tensorflow as tf


def create_model(input_tensor, mode, hyper_params):
    model = vgg_model(input_tensor, mode, hyper_params)
    with tf.variable_scope("simple_detector"):
        num_classes = hyper_params.get_or_dummy("problem").get_or_default("number_of_categories", 2)

        conv6_size = hyper_params.get_or_dummy("simple_detector").get_or_default("conv6_kernel_size", 7)
        model["conv6"] = tf.layers.conv2d(inputs=model["pool5"], filters=4096, kernel_size=(conv6_size, conv6_size),
                                          strides=(1, 1), padding="same", activation=tf.nn.relu, name="conv6")
        model["conv7"] = tf.layers.conv2d(inputs=model["conv6"], filters=4096, kernel_size=(1, 1), strides=(1, 1),
                                          padding="valid", activation=tf.nn.relu, name="conv7")

        # Classification
        model["logits_raw"] = tf.layers.conv2d(inputs=model["conv7"], filters=num_classes * 64, kernel_size=(1, 1),
                                               strides=(1, 1),
                                               padding="valid", activation=None, name="logits_raw")
        model["logits"] = tile_2d(model["logits_raw"], 8, 8, name="logits", reorder_required=True)
        model["probs"] = tf.nn.softmax(model["logits"], name="probs")

        # Bounding rect regression
        model["rect_raw"] = tf.layers.conv2d(inputs=model["conv7"], filters=4 * 64, kernel_size=(1, 1),
                                             strides=(1, 1),
                                             padding="valid", activation=None, name="rect_raw")
        model["rect"] = tile_2d(model["rect_raw"], 8, 8, name="rect", reorder_required=True)

    return model
