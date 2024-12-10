import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 假设 mf_torque 和 sigma_torque 是提前定义好的
# mf_torque = tf.Variable(np.asarray([0.28, 0.45, 0.62]), dtype=tf.float64)
# sigma_torque = tf.Variable(np.asarray([30, 0.2, 30]), dtype=tf.float64)
# #
# mf_torque = tf.Variable(np.asarray([0.28, 0.45, 0.62]), dtype=tf.float64)
# sigma_torque = tf.Variable(np.asarray([30, 0.2, 30]), dtype=tf.float64)
# mf_delta_torque = tf.Variable(np.asarray([-0.2, -0.15, -0.05, 0, 0.05, 0.15, 0.2]), dtype=tf.float64)
# sigma_delta_torque = tf.Variable(np.asarray([150, 0.04, 0.04, 0.02, 0.04, 0.04, 150]), dtype=tf.float64)
# torque_range = np.arange(0, 0.9, 0.01)
#
# mf_torque = tf.Variable(np.asarray([50, 60, 70]), dtype=tf.float64)
# sigma_torque = tf.Variable(np.asarray([0.5, 10, 0.5]), dtype=tf.float64)
# mf_delta_torque = tf.Variable(np.asarray([-0.25, -0.2, -0.1, 0, 0.1, 0.2, 0.25]), dtype=tf.float64)
# sigma_delta_torque = tf.Variable(np.asarray([130, 0.05, 0.05, 0.05, 0.05, 0.05, 130]), dtype=tf.float64)
# torque_range = np.arange(30, 90, 0.1)

mf_torque = tf.Variable(np.asarray([-8, -7, -3, 0, 3, 7, 8]), dtype=tf.float64)
sigma_torque = tf.Variable(np.asarray([2, 1.5, 1.5, 1.5, 1.5, 1.5, 2]), dtype=tf.float64)
mf_delta_torque = tf.Variable(np.asarray([-9.5, -7, -3.5, 0, 3.5, 7, 9.5]), dtype=tf.float64)
sigma_delta_torque = tf.Variable(np.asarray([2, 1.5, 1.5, 1.5, 1.5, 1.5, 2]), dtype=tf.float64)
torque_range = np.arange(-12, 12, 0.1)

# mf_delta_torque = tf.Variable(np.asarray([-8.5, -5, 0, 5, 8.5]), dtype=tf.float64)
# sigma_delta_torque = tf.Variable(np.asarray([2, 2.5, 2.5, 2.5, 2]), dtype=tf.float64)

# 定义 self.torque 的范围，比如从 0.5 到 2.0，步长为 0.1
# torque_range = np.arange(0, 0.9, 0.01)
# torque = tf.convert_to_tensor(torque_range, dtype=tf.float64)

# 定义 delta_torque 的范围，比如从 -0.5 到 0.5，步长为 0.01
delta_torque_range = np.arange(-12, 12, 0.1)
delta_torque = tf.convert_to_tensor(delta_torque_range, dtype=tf.float64)
plt.figure(figsize=(10, 6))
for i in range(5):
    if i == 0:
        # 对头部使用特定的隶属度函数（例如 sigmoid 函数，递增的曲线）
        rul_delta = 1 / (1 + tf.exp(sigma_delta_torque[i] * (delta_torque - mf_delta_torque[i])))
    elif i == 4:
        # 对尾部使用特定的隶属度函数（例如 sigmoid 函数，递减的曲线）
        rul_delta = 1 / (1 + tf.exp(-sigma_delta_torque[i] * (delta_torque - mf_delta_torque[i])))
    else:
        # 对中间的使用高斯隶属度函数    
        rul_delta = tf.exp(-tf.square(tf.subtract(delta_torque, mf_delta_torque[i])) / tf.square(sigma_delta_torque[i]))
    
    plt.plot(delta_torque_range, rul_delta.numpy())

plt.legend()

# 显示图形
plt.grid(True)
plt.tight_layout()


# 计算 torque_first, torque_last 和 rul_error
# torque_first = 1 / (1 + tf.exp(sigma_torque[0] * (torque - mf_torque[0])))
# torque_last = 1 / (1 + tf.exp(-sigma_torque[-1] * (torque - mf_torque[-1])))
# rul_error = tf.exp(-tf.square(tf.subtract(torque, mf_torque[1])) / tf.square(sigma_torque[1]))

# # 绘制 torque_first, torque_last 和 rul_error 的变化
# plt.figure(figsize=(10, 6))

# # 绘制 torque_first
# plt.plot(torque_range, torque_first.numpy(), color='red')

# # 绘制 torque_last
# plt.plot(torque_range, torque_last.numpy(), color='blue')

# # 绘制 rul_error
# plt.plot(torque_range, rul_error.numpy(), color='green')

# 设置图形标题和标签
plt.xlabel("Delta Error (Deg/s)")
plt.ylabel("Membership Value")

# # 显示图形
# plt.grid(True)
plt.tight_layout()


plt.show()
