import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

error = tf.convert_to_tensor( [5], dtype=tf.float32)
a = tf.Variable(np.asarray([-9, -6, -3, 0, 3, 6, 9]), dtype=tf.float32)
mu_error = tf.Variable(np.asarray([-9, -6, -3, 0, 3, 6, 9]), dtype=tf.float32)
sigma_error = tf.Variable(np.asarray([1,1,1,1,1,1,1]), dtype=tf.float32)

rul_error =tf.exp(-tf.square(tf.subtract(tf.tile(error, [5]), mu_error[1:6])) / tf.square(sigma_error[1:6]))
error_first = 1 / (1 + tf.exp(-mu_error[0]*(error -mu_error[0])))

error_first = tf.expand_dims(error_first, -1)  # 在最后增加一个维度
error_first = tf.expand_dims(error_first, -1)  # 在最后增加一个维度
rul_error = tf.expand_dims(rul_error, axis=-1)
a = tf.expand_dims(a, -1)
mf_error = tf.concat([error_first,rul_error,error_first], axis=0)

aa = tf.minimum(tf.expand_dims(a, axis=1), tf.expand_dims(mf_error, axis=0))
print(error_first)