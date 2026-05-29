import tensorflow as tf


def PLCCloss(y_true, y_pred):
    
    y_p = y_pred - tf.reduce_mean(y_pred)
    y_t = y_true - tf.reduce_mean(y_true)

    loss = tf.reduce_sum(y_p*y_t) / (tf.sqrt(tf.reduce_sum(y_p**2)) * tf.sqrt(tf.reduce_sum(y_t**2)))
    
    return loss