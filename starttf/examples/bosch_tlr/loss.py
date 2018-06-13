import tensorflow as tf
from starttf.losses.basic_losses import smooth_l1_distance
from starttf.losses.loss_processors import mask_loss, multiloss, batch_alpha_balance_loss, interpolate_loss
from starttf.losses.utils import overlay_classification_on_image


def create_loss(model, labels, mode, hyper_params):
    """
    Create a cross entropy loss with the loss as the only metric.

    :param model: A dictionary containing all output tensors of your model.
    :param labels: A dictionary containing all label tensors.
    :param mode: tf.estimators.ModeKeys defining if you are in eval or training mode.
    :param hyper_params: A hyper parameters object.
    :return: All the losses (tensor dict, "loss" is the loss that is used for minimization)
            and all the metrics(tensor dict) that should be logged for debugging.
    """
    metrics = {}
    losses = {}
    k = hyper_params.problem.number_of_categories
    d = hyper_params.problem.number_of_directions
    with tf.variable_scope("losses"):
        # Create one hot encoding from sparse label. (Simplifies usage of mask)
        with tf.variable_scope("preprocessing"):
            one_hot = tf.one_hot(labels["class_id"], k+1)
            shape = one_hot.get_shape().as_list()
            class_label = tf.reshape(one_hot, shape=[-1, shape[1], shape[2], shape[4]])
            one_hot = tf.one_hot(labels["direction"], d+1)
            shape = one_hot.get_shape().as_list()
            direction_label = tf.reshape(one_hot, shape=[-1, shape[1], shape[2], shape[4]])

            # Create mask for correctly masking loss (background and/or don't care)
            mask = 0
            for i in range(1, k, 1):
                mask += class_label[:, :, :, i]
            direction_mask = 0
            for i in range(d):
                direction_mask += direction_label[:, :, :, i]

        # Add loss
        with tf.variable_scope("class_id"):
            ce_1 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=model["logits"], labels=class_label[:, :, :, :k])
            ce_2 = batch_alpha_balance_loss(labels=class_label[:, :, :, :k], loss=ce_1)
            ce_3 = interpolate_loss(labels=class_label[:, :, :, :k], loss1=ce_1, loss2=ce_2,
                                    interpolation_values=hyper_params.train.normalization_strength.class_id)
            masked_ce = mask_loss(input_tensor=ce_3, binary_tensor=class_label[:, :, :, 0] + mask)
            losses["class_id_loss"] = tf.reduce_mean(masked_ce)

            tf.summary.image("image/class_id_loss", overlay_classification_on_image(classification=tf.expand_dims(masked_ce, axis=-1),
                                                                                     rgb_image=model["image"], scale=4))
            tf.summary.image("image/class_id_preds", overlay_classification_on_image(classification=model["probs"][:, :, :, 1:min(k, 4)],
                                                                                     rgb_image=model["image"], scale=4))
            tf.summary.image("image/class_id_labels", overlay_classification_on_image(classification=class_label[:, :, :, 1:min(k, 4)],
                                                                                     rgb_image=model["image"], scale=4))

        # Add direction loss
        with tf.variable_scope("direction"):
            ce_1 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=model["direction_logits"], labels=direction_label[:, :, :, :d])
            ce_2 = batch_alpha_balance_loss(labels=direction_label[:, :, :, :d], loss=ce_1)
            ce_3 = interpolate_loss(labels=direction_label[:, :, :, :d], loss1=ce_1, loss2=ce_2,
                                    interpolation_values=hyper_params.train.normalization_strength.direction)
            losses["direction_loss"] = tf.reduce_mean(mask_loss(input_tensor=ce_3, binary_tensor=direction_mask))

            tf.summary.image("image/direction_preds", overlay_classification_on_image(classification=model["direction_probs"][:, :, :, 0:min(d, 3)],
                                                                                     rgb_image=model["image"], scale=4))
            tf.summary.image("image/direction_labels", overlay_classification_on_image(classification=direction_label[:, :, :, 0:min(d, 3)],
                                                                                     rgb_image=model["image"], scale=4))

        # Rect loss
        with tf.variable_scope("rect"):
            l1_loss = smooth_l1_distance(labels=labels["rect"][:, :, :], preds=model['rect'][:, :, :])
            losses["rect_loss"] = tf.reduce_mean(mask_loss(input_tensor=l1_loss, binary_tensor=mask))

        # Add losses to dict. "loss" is the primary loss that is optimized.
        if not hyper_params.train.loss_weights.get_or_default("automatic", False):
            losses["loss"] = hyper_params.train.loss_weights.class_id * losses["class_id_loss"] + \
                             hyper_params.train.loss_weights.direction * losses["direction_loss"] + \
                             hyper_params.train.loss_weights.rect * losses["rect_loss"]
        else:
            losses["loss"] = multiloss([losses["class_id_loss"], losses["direction_loss"], losses["rect_loss"]],
                                       logging_namespace="multiloss")

    return losses, metrics
