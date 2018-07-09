# MIT License
# 
# Copyright (c) 2017 Marvin Teichmann
# Copyright (c) 2018 Modified by Michael Fuerst
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
An implementation of rezoom layer that meets the standarts of the starttf project.

For description of the rezoom layer see:
https://arxiv.org/abs/1612.07695

The original code can be found here:
https://github.com/MarvinTeichmann/KittiBox
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def _to_idx(vec, w_shape):
    """
    vec = (idn, idh, idw)
    w_shape = [n, h, w, c]
    """
    return vec[:, 2] + w_shape[2] * (vec[:, 1] + w_shape[1] * vec[:, 0])


def _interp(w, i, channel_dim):
    """
    Input:
        w: A 4D block tensor of shape (n, h, w, c)
        i: A list of 3-tuples [(x_1, y_1, z_1), (x_2, y_2, z_2), ...],
            each having type (int, float, float)
        The 4D block represents a batch of 3D image feature volumes with c channels.
        The input i is a list of points  to index into w via interpolation. Direct
        indexing is not possible due to y_1 and z_1 being float values.
    Output:
        A list of the values: [
            w[x_1, y_1, z_1, :]
            w[x_2, y_2, z_2, :]
            ...
            w[x_k, y_k, z_k, :]
        ]
        of the same length == len(i)
    """
    w_as_vector = tf.reshape(w, [-1, channel_dim])  # gather expects w to be 1-d
    upper_l = tf.to_int32(tf.concat(axis=1, values=[i[:, 0:1], tf.floor(i[:, 1:2]), tf.floor(i[:, 2:3])]))
    upper_r = tf.to_int32(tf.concat(axis=1, values=[i[:, 0:1], tf.floor(i[:, 1:2]), tf.ceil(i[:, 2:3])]))
    lower_l = tf.to_int32(tf.concat(axis=1, values=[i[:, 0:1], tf.ceil(i[:, 1:2]), tf.floor(i[:, 2:3])]))
    lower_r = tf.to_int32(tf.concat(axis=1, values=[i[:, 0:1], tf.ceil(i[:, 1:2]), tf.ceil(i[:, 2:3])]))

    upper_l_idx = _to_idx(upper_l, tf.shape(w))
    upper_r_idx = _to_idx(upper_r, tf.shape(w))
    lower_l_idx = _to_idx(lower_l, tf.shape(w))
    lower_r_idx = _to_idx(lower_r, tf.shape(w))

    upper_l_value = tf.gather(w_as_vector, upper_l_idx)
    upper_r_value = tf.gather(w_as_vector, upper_r_idx)
    lower_l_value = tf.gather(w_as_vector, lower_l_idx)
    lower_r_value = tf.gather(w_as_vector, lower_r_idx)

    alpha_lr = tf.expand_dims(i[:, 2] - tf.floor(i[:, 2]), 1)
    alpha_ud = tf.expand_dims(i[:, 1] - tf.floor(i[:, 1]), 1)

    upper_value = (1 - alpha_lr) * upper_l_value + alpha_lr * upper_r_value
    lower_value = (1 - alpha_lr) * lower_l_value + alpha_lr * lower_r_value
    value = (1 - alpha_ud) * upper_value + alpha_ud * lower_value
    return value


def _bilinear_select(hyper_params, pred_boxes, w_offset, h_offset):
    """
    Function used for rezooming high level feature maps. Uses bilinear interpolation
    to select all channels at index (x, y) for a high level feature map, where x and y are floats.
    """
    batch_size, h, w, c = pred_boxes.get_shape().as_list()
    grid_size = h * w
    outer_size = grid_size * batch_size

    fine_stride = 8.  # pixels per 60x80 grid cell in 480x640 image
    coarse_stride = hyper_params['region_size']  # pixels per 15x20 grid cell in 480x640 image
    batch_ids = []
    x_offsets = []
    y_offsets = []
    for n in range(batch_size):
        for i in range(h):
            for j in range(w):
                batch_ids.append([n])
                x_offsets.append([coarse_stride / 2. + coarse_stride * j])
                y_offsets.append([coarse_stride / 2. + coarse_stride * i])

    batch_ids = tf.constant(batch_ids)
    x_offsets = tf.constant(x_offsets)
    y_offsets = tf.constant(y_offsets)

    pred_boxes_r = tf.reshape(pred_boxes, [outer_size, 4])
    scale_factor = coarse_stride / fine_stride  # scale difference between 15x20 and 60x80 features

    pred_x_center = (pred_boxes_r[:, 0:1] + w_offset * pred_boxes_r[:, 2:3] + x_offsets) / fine_stride
    pred_y_center = (pred_boxes_r[:, 1:2] + h_offset * pred_boxes_r[:, 3:4] + y_offsets) / fine_stride

    # Clip value to fit into larger feature layer.
    pred_x_center_clip = tf.clip_by_value(pred_x_center, 0, scale_factor * w - 1)
    pred_y_center_clip = tf.clip_by_value(pred_y_center, 0, scale_factor * h - 1)

    interp_indices = tf.concat(axis=1, values=[tf.to_float(batch_ids), pred_y_center_clip, pred_x_center_clip])
    return interp_indices


def _rezoom(hyper_params, pred_boxes, early_feat, early_feat_channels,
            w_offsets, h_offsets):
    """
    Rezoom into a feature map at multiple interpolation points
    in a grid.
    If the predicted object center is at X, len(w_offsets) == 3,
    and len(h_offsets) == 5,
    the rezoom grid will look as follows:
    [o o o]
    [o o o]
    [o X o]
    [o o o]
    [o o o]
    Where each letter indexes into the feature map with bilinear interpolation
    """
    with tf.name_scope('rezoom'):
        batch_size, h, w, c = pred_boxes.get_shape().as_list()

        # Create a list of interpolation indices.
        indices = []
        for w_offset in w_offsets:
            for h_offset in h_offsets:
                indices.append(_bilinear_select(hyper_params, pred_boxes, w_offset, h_offset))
        interp_indices = tf.concat(axis=0, values=indices)

        # Get the correct values from the early feature layer.
        rezoom_features = _interp(early_feat, interp_indices, early_feat_channels)

        # Restore channels, reorder channels and then squash last two together.
        rezoom_features = tf.reshape(rezoom_features, [-1, batch_size, h, w, early_feat_channels])
        rezoom_features_t = tf.transpose(rezoom_features, [1, 2, 3, 0, 4])
        return tf.reshape(rezoom_features_t, [batch_size, h, w, len(w_offsets) * len(h_offsets) * early_feat_channels])


def rezoom_layer(hyper_params, pred_boxes, pred_logits, early_feat, hidden_output, mode):
    with tf.name_scope('rezoom_layer'):
        batch_size, h, w, c = pred_boxes.get_shape().as_list()
        grid_size = h * w
        outer_size = grid_size * batch_size

        early_feat_channels = hyper_params['early_feat_channels']
        early_feat = early_feat[:, :, :, :early_feat_channels]

        w_offsets = hyper_params['rezoom_w_coords']
        h_offsets = hyper_params['rezoom_h_coords']
        num_offsets = len(w_offsets) * len(h_offsets)
        rezoom_features = _rezoom(hyper_params, pred_boxes, early_feat, early_feat_channels, w_offsets, h_offsets)
        if mode == tf.estimator.ModeKeys.Train:
            rezoom_features = tf.nn.dropout(rezoom_features, 0.5)

        delta_features = tf.concat(axis=1, values=[hidden_output, rezoom_features / 1000.])  # Why / 1000.?
        dim = 128
        shape = [hyper_params['num_inner_channel'] +
                 early_feat_channels * num_offsets,
                 dim]
        delta_weights1 = tf.get_variable('delta1',
                                         shape=shape)
        # TODO: maybe adding dropout here?
        ip1 = tf.nn.relu(tf.matmul(delta_features, delta_weights1))
        if mode == tf.estimator.ModeKeys.Train:
            ip1 = tf.nn.dropout(ip1, 0.5)
        delta_confs_weights = tf.get_variable(
            'delta2', shape=[dim, hyper_params['num_classes']])
        delta_boxes_weights = tf.get_variable('delta_boxes', shape=[dim, 4])

        rere_feature = tf.matmul(ip1, delta_boxes_weights) * 5
        pred_boxes_delta = (tf.reshape(rere_feature, [outer_size, 1, 4]))

        scale = hyper_params.get('rezoom_conf_scale', 50)
        feature2 = tf.matmul(ip1, delta_confs_weights) * scale
        pred_confs_delta = tf.reshape(feature2, [outer_size, 1,
                                                 hyper_params['num_classes']])

        pred_confs_delta = tf.reshape(pred_confs_delta,
                                      [outer_size, hyper_params['num_classes']])

        pred_confidences_squash = tf.nn.softmax(pred_confs_delta)
        pred_confidences = tf.reshape(pred_confidences_squash,
                                      [outer_size, hyper_params['rnn_len'],
                                       hyper_params['num_classes']])

        return pred_boxes, pred_logits, pred_confidences, \
            pred_confs_delta, pred_boxes_delta


def _add_rezoom_loss_histograms(hyper_params, pred_boxes_deltas):
    """
    Add some histograms to tensorboard.
    """
    tf.summary.histogram(
        '/delta_hist0_x', pred_boxes_deltas[:, 0, 0])
    tf.summary.histogram(
        '/delta_hist0_y', pred_boxes_deltas[:, 0, 1])
    tf.summary.histogram(
        '/delta_hist0_w', pred_boxes_deltas[:, 0, 2])
    tf.summary.histogram(
        '/delta_hist0_h', pred_boxes_deltas[:, 0, 3])


def rezoom_loss(hyper_params,
                true_boxes, true_classes,
                pred_boxes, pred_confs_deltas, pred_boxes_deltas,
                box_mask, conf_mask):
    """
    Computes loss for delta output. Only relevant
    if rezoom layers are used.
    """

    # Confidence loss
    if hyper_params['rezoom_change_loss'] == 'center':
        error = (true_boxes[:, :, 0:2] - pred_boxes[:, :, 0:2]) \
            / tf.maximum(true_boxes[:, :, 2:4], 1.)
        square_error = tf.reduce_sum(tf.square(error), 2)

        inside = tf.to_int64(
            tf.logical_and(tf.less(square_error, 0.2**2),
                           tf.greater(true_classes, 0)))
    else:
        assert not hyper_params['rezoom_change_loss']
        inside = tf.to_int64((tf.greater(true_classes, 0)))

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=pred_confs_deltas, labels=inside)
    delta_confs_loss = tf.reduce_mean(cross_entropy * conf_mask)

    # Regression Loss
    delta_residual = (true_boxes - (pred_boxes + pred_boxes_deltas)) * box_mask
    sqrt_delta = tf.minimum(tf.square(delta_residual), 10. ** 2)
    delta_boxes_loss = tf.reduce_mean(sqrt_delta)

    return delta_confs_loss, delta_boxes_loss
