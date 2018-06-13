import tensorflow as tf


def alpha_balance_loss(labels, loss, alpha_weights):
    """
    Calculate the alpha balanced cross_entropy.

    This means for each sample the cross entropy is calculated and then weighted by the class specific weight.

    :param labels: A float tensor of shape [batch_size, ..., num_classes] representing the label class probabilities.
    :param loss: A float tensor of shape [batch_size, ...] representing the loss that should be focused.
    :param alpha_weights: A float tensor of shape [1, ..., num_classes] (... is filled with ones to match number
                              of dimensions to labels tensor) representing the weights for each class.
    :return: A tensor representing the weighted cross entropy.
    """
    with tf.variable_scope("alpha_balance"):
        # Broadcast multiply labels with alpha weights to select weights and then reduce them along last axis.
        weights = tf.reduce_sum(labels * alpha_weights, axis=-1)
        return weights * loss


def focus_loss(labels, probs, loss, gamma):
    """
    Calculate the alpha balanced focal loss.

    See the focal loss paper: "Focal Loss for Dense Object Detection" [by Facebook AI Research]

    :param labels: A float tensor of shape [batch_size, ..., num_classes] representing the label class probabilities.
    :param probs: A float tensor of shape [batch_size, ..., num_classes] representing the probs (after softmax).
    :param loss: A float tensor of shape [batch_size, ...] representing the loss that should be focused.
    :param gamma: The focus parameter.
    :return: A tensor representing the weighted cross entropy.
    """
    with tf.variable_scope("focus_loss"):
        # Compute p_t that is used in paper.
        # FIXME is it possible that the 1-p term does not make any sense?
        p_t = tf.reduce_sum(probs * labels, axis=-1)# + tf.reduce_sum((1.0 - probs) * (1.0 - labels), axis=-1)

        focal_factor = tf.pow(1.0 - p_t, gamma) if gamma > 0 else 1  # Improve stability for gamma = 0
        return focal_factor * loss


def interpolate_loss(labels, loss1, loss2, interpolation_values):
    """
    Interpolate two losses linearly.

    :param labels: A float tensor of shape [batch_size, ..., num_classes] representing the label class probabilities.
    :param loss1: A float tensor of shape [batch_size, ...] representing the loss1 for interpolation.
    :param loss2: A float tensor of shape [batch_size, ...] representing the loss2 for interpolation.
    :param interpolation_values: The values for each class how much focal loss should be interpolated in.
    :return: A tensor representing the weighted cross entropy.
    """
    with tf.variable_scope("interpolate_focus_loss"):
        # Select the probs or weights with the labels.
        t = tf.reduce_sum(labels * interpolation_values, axis=-1)
        return (1 - t) * loss1 + t * loss2


def batch_alpha_balance_loss(labels, loss):
    """
    Calculate the alpha balanced cross_entropy.

    This means for each sample the cross entropy is calculated and then weighted by the class specific weight.

    :param labels: A float tensor of shape [batch_size, ..., num_classes] representing the label class probabilities.
    :param loss: A float tensor of shape [batch_size, ...] representing the loss that should be focused.
    :param alpha_weights: A float tensor of shape [1, ..., num_classes] (... is filled with ones to match number
                              of dimensions to labels tensor) representing the weights for each class.
    :return: A tensor representing the weighted cross entropy.
    """
    with tf.variable_scope("batch_alpha_balance"):
        # Compute the occurrence probability for each class
        mu, _ = tf.nn.moments(labels, [0, 1, 2])

        # For weighting a class should be down weighted by its occurrence probability.
        not_mu = 1 - mu

        # Select the class specific not_mu
        not_mu_class = tf.reduce_sum(labels * not_mu, axis=-1)
        return not_mu_class * loss


def mask_loss(input_tensor, binary_tensor):
    """
    Mask a loss by using a tensor filled with 0 or 1.

    :param input_tensor: A float tensor of shape [batch_size, ...] representing the loss/cross_entropy
    :param binary_tensor: A float tensor of shape [batch_size, ...] representing the mask.
    :return: A float tensor of shape [batch_size, ...] representing the masked loss.
    """
    with tf.variable_scope("mask_loss"):
        mask = tf.cast(tf.cast(binary_tensor, tf.bool), tf.float32)

        return input_tensor * mask


def multiloss(losses, logging_namespace="multiloss"):
    """
    Create a loss from multiple losses my mixing them.
    This multi-loss implementation is inspired by the Paper "Multi-Task Learning Using Uncertainty to Weight Losses
    for Scene Geometry and Semantics" by Kendall, Gal and Cipolla.
    :param losses: A list containing all losses that should be merged.
    :param logging_namespace: Variable scope in which multiloss lives.
    :return: A single loss.
    """
    with tf.variable_scope(logging_namespace):
        sum_loss = 0
        for i, loss in enumerate(losses):
            with tf.variable_scope(str(i)) as scope:
                sigma = tf.get_variable(name="sigma", dtype=tf.float32, initializer=tf.constant(1.0), trainable=True)
                sigma_2 = tf.pow(sigma, 2)
                tf.summary.scalar("sigma2", sigma_2)
                sum_loss += 0.5 / sigma_2 * loss + tf.log(sigma_2 + 1.0)
        return sum_loss
