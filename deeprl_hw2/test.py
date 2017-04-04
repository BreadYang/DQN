import tensorflow as tf
import numpy as np
import semver


def huber_loss(y_true, y_pred, max_grad=1.):

    diff = y_pred - y_true
    if semver.match(tf.__version__, '<1.0.0'):
        huber_val = tf.select(tf.abs(diff) < max_grad, 0.5 * tf.square(diff), max_grad * (tf.abs(diff) - 0.5*max_grad))
    else:
        huber_val = tf.where(tf.abs(diff) < max_grad, 0.5 * tf.square(diff), max_grad * (tf.abs(diff) - 0.5*max_grad))
    return  huber_val



x = np.array([2.0,3.0,1.0,0.0])
y = np.array([2.0,5.0,1.0,0.0])

result = huber_loss(x,y)
result2 = tf.reduce_sum(result)
result3 = tf.reduce_mean(result)

sess = tf.Session()
print sess.run([result, result2, result3])

print result
