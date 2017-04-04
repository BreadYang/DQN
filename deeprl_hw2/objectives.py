"""Loss functions."""

import tensorflow as tf
import semver


def huber_loss(y_true, y_pred, max_grad=1.):
    """Calculate the huber loss.

    See https://en.wikipedia.org/wiki/Huber_loss

    Parameters
    ----------
    y_true: np.array, tf.Tensor
      Target value.
    y_pred: np.array, tf.Tensor
      Predicted value.
    max_grad: float, optional
      Positive floating point value. Represents the maximum possible
      gradient magnitude.

    Returns
    -------
    tf.Tensor
      The huber loss.
    """
    diff = y_pred - y_true
    if semver.match(tf.__version__, '<1.0.0'):
        huber_val = tf.select(tf.abs(diff) < max_grad, 0.5 * tf.square(diff), max_grad * (tf.abs(diff) - 0.5*max_grad))
    else:
        huber_val = tf.where(tf.abs(diff) < max_grad, 0.5 * tf.square(diff), max_grad * (tf.abs(diff) - 0.5*max_grad))
    return huber_val


def mean_huber_loss(y_true, y_pred, max_grad=1.):
    """Return mean huber loss.

    Same as huber_loss, but takes the mean over all values in the
    output tensor.

    Parameters
    ----------
    y_true: np.array, tf.Tensor
      Target value.
    y_pred: np.array, tf.Tensor
      Predicted value.
    max_grad: float, optional
      Positive floating point value. Represents the maximum possible
      gradient magnitude.

    Returns
    -------
    tf.Tensor
      The mean huber loss.
    """
    diff = y_pred - y_true
    if semver.match(tf.__version__, '<1.0.0'):
        huber_val = tf.select(tf.abs(diff) < max_grad, 0.5 * tf.square(diff), max_grad * (tf.abs(diff) - 0.5*max_grad))
    else:
        huber_val = tf.where(tf.abs(diff) < max_grad, 0.5 * tf.square(diff), max_grad * (tf.abs(diff) - 0.5*max_grad))
    return tf.reduce_mean(huber_val)
