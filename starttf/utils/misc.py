import sys
import tensorflow as tf
PYTHON_VERSION = sys.version_info[0]
if PYTHON_VERSION == 2:
    from urllib import urlretrieve
elif PYTHON_VERSION == 3:
    from urllib.request import urlretrieve


def mode_to_str(mode):
    """
    Converts a tf.estimator.ModeKeys in a nice readable string.
    :param mode: The mdoe as a tf.estimator.ModeKeys
    :return: A human readable string representing the mode.
    """
    if mode == tf.estimator.ModeKeys.TRAIN:
        return "train"
    if mode == tf.estimator.ModeKeys.EVAL:
        return "eval"
    if mode == tf.estimator.ModeKeys.PREDICT:
        return "predict"
    return "unknown"


def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z


def download(url, filename):
    """
    Download the url into a file.
    :param url: The url where to get the weights.
    :param filename: The filename where to store it.
    :return:
    """
    urlretrieve(url, filename)


def tf_if(condition, a, b):
    """
    Implements an if condition in tensorflow.
    :param condition: A boolean condition.
    :param a: Case a.
    :param b: Case b.
    :return: A if condition was true, b otherwise.
    """
    int_condition = tf.to_float(tf.to_int64(condition))
    return a * int_condition + (1 - int_condition) * b