import tensorflow as tf


def tile_2d(input, k_x, k_y, name, reorder_required=True):
    """
    A tiling layer like introduced in overfeat and huval papers.
    :param input: Your input tensor.
    :param k_x: The tiling factor in x direction.
    :param k_y: The tiling factor in y direction.
    :param name: The name of the layer.
    :param reorder_required: To implement an exact huval tiling you need reordering.
      However not using it is more efficient and when training from scratch setting this to false is highly recommended.
    :return: The output tensor.
    """
    size = input.get_shape().as_list()
    c, h, w = size[3], size[1], size[2]
    batch_size = size[0]
    if batch_size is None:
        batch_size = -1

    # Check if tiling is possible and define output shape.
    assert c % (k_x * k_y) == 0

    tmp = input

    if reorder_required:
        output_channels = int(c / (k_x * k_y))
        channels = tf.unstack(tmp, axis=-1)
        reordered_channels = [None for _ in range(len(channels))]
        for o in range(output_channels):
            for i in range(k_x * k_y):
                target = o + i * output_channels
                source = o * (k_x * k_y) + i
                reordered_channels[target] = channels[source]
        tmp = tf.stack(reordered_channels, axis=-1)

    # Actual tilining
    with tf.variable_scope(name) as scope:
        tmp = tf.transpose(tmp, [0, 2, 1, 3])
        tmp = tf.reshape(tmp, (batch_size, w, int(h * k_y), int(c / (k_y))))
        tmp = tf.transpose(tmp, [0, 2, 1, 3])
        tmp = tf.reshape(tmp, (batch_size, int(h * k_y), int(w * k_x), int(c / (k_y * k_x))))
    
    return tmp


def inverse_tile_2d(input, k_x, k_y, name):
    """
        An inverse tiling layer.

        An inverse to the tiling layer can be of great use, since you can keep the resolution of your output low,
        but harness the benefits of the resolution of a higher level feature layer.
        If you insist on a source you can call it very lightly inspired by yolo9000 "passthrough layer".

        :param input: Your input tensor. (Assert input.shape[1] % k_y = 0 and input.shape[2] % k_x = 0)
        :param k_x: The tiling factor in x direction [int].
        :param k_y: The tiling factor in y direction [int].
        :param name: The name of the layer.
        :return: The output tensor of shape [batch_size, inp.height / k_y, inp.width / k_x, inp.channels * k_x * k_y].
        """

    batch_size, h, w, c = input.get_shape().as_list()
    if batch_size is None:
        batch_size = -1

    # Check if tiling is possible and define output shape.
    assert w % k_x == 0 and h % k_y == 0

    # Actual inverse tilining
    with tf.variable_scope(name) as scope:
        tmp = input
        tmp = tf.reshape(tmp, (batch_size, int(h * k_y), w, int(c * k_x)))
        tmp = tf.transpose(tmp, [0, 2, 1, 3])
        tmp = tf.reshape(tmp, (batch_size, w, h, int(c * k_y * k_x)))
        tmp = tf.transpose(tmp, [0, 2, 1, 3])

    return tmp


def feature_passthrough(early_feat, late_feat, k_x, k_y, outputs, name):
    """
    A feature passthrough layer inspired by yolo9000 and the inverse tiling layer.

    It can be proven, that this layer does the same as conv1x1(concat(inverse_tile(early_feat), late_feat)).
    This layer has no activation function.

    :param early_feat: The early feature layer of shape [batch_size, h * k_y, w * k_x, _].
    :param late_feat:  The late feature layer of shape [batch_size, h, w, _].
    :param k_x: The tiling factor in x direction [int]. Also scale difference between early and late.
    :param k_y: The tiling factor in y direction [int]. Also scale difference between early and late.
    :param outputs: The number of output features.
    :param name: The name of the layer.
    :return: The output tensor of shape [batch_size, h, w, outputs]
    """
    with tf.variable_scope(name) as scope:
        early_conv = tf.layers.conv2d(early_feat, filters=outputs, kernel_size=(k_x, k_y), strides=(k_x, k_y))
        late_conv = tf.layers.conv2d(late_feat, filters=outputs, kernel_size=(1, 1), strides=(1, 1))
        return early_conv + late_conv
