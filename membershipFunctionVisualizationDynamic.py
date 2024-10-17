import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

file_path = './exp/2024_10_08_2006/1/down_model.npy' #1kg
# file_path = './exp/2024_10_10_1535/1/up_model.npy' #2kg
# file_path = './exp/2024_10_10_1723/1/up_model.npy' #1kg
loaded_data = np.load(file_path)
# print(len(loaded_data)/42)
# # loaded_data = loaded_data[-42:]
# print(loaded_data[6])
# 初始化参数
kg = 1
if kg ==1 :
    mf_torque = tf.Variable(np.asarray([50, 60, 70]), dtype=tf.float64)
    # mf_torque.assign(np.asarray([0.2, 0.39, 0.58]))

    sigma_torque = tf.Variable(np.asarray([0.5, 10, 0.5]), dtype=tf.float64)
    mf_delta_torque = tf.Variable(np.asarray([-0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15]), dtype=tf.float64)
    sigma_delta_torque = tf.Variable(np.asarray([130, 0.05, 0.05, 0.05, 0.05, 0.05, 130]), dtype=tf.float64)
    # 定义 torque 和 delta_torque 的范围
    torque_range = np.arange(30, 90, 0.1)
    torque = tf.convert_to_tensor(torque_range, dtype=tf.float64)

    delta_torque_range = np.arange(-0.3, 0.3, 0.001)
    delta_torque = tf.convert_to_tensor(delta_torque_range, dtype=tf.float64)
elif kg ==2:
    mf_torque = tf.Variable(np.asarray([0.08,0.12, 0.16]), dtype=tf.float64)
    # mf_torque.assign(np.asarray([0.2, 0.39, 0.58]))

    sigma_torque = tf.Variable(np.asarray([100, 0.04, 100]), dtype=tf.float64)
    mf_delta_torque = tf.Variable(np.asarray([-0.2, -0.15, -0.05, 0, 0.05, 0.15, 0.2]), dtype=tf.float64)
    sigma_delta_torque = tf.Variable(np.asarray([150, 0.04, 0.04, 0.02, 0.04, 0.04, 150]), dtype=tf.float64)
    # 定义 torque 和 delta_torque 的范围
    torque_range = np.arange(0, 0.25, 0.001)
    torque = tf.convert_to_tensor(torque_range, dtype=tf.float64)

    delta_torque_range = np.arange(-0.3, 0.3, 0.001)
    delta_torque = tf.convert_to_tensor(delta_torque_range, dtype=tf.float64)
    [0.43,0.935, 1.44]

# 初始化两个独立的 figure
fig1, ax1 = plt.subplots(figsize=(10, 6))  # 第一个图: torque 曲线
fig2, ax2 = plt.subplots(figsize=(10, 6))  # 第二个图: delta_torque 曲线

# 第一张图：torque 曲线
torque_first_line, = ax1.plot([], [], color='red')
torque_last_line, = ax1.plot([], [], color='blue')
rul_error_line, = ax1.plot([], [], color='green')

# 第二张图：delta_torque 曲线
delta_torque_lines = [ax2.plot([], [])[0] for _ in range(7)]  # 7条线

# 设置图1标题和标签
ax1.set_xlabel("Torque")
ax1.set_ylabel("Membership Value")
ax1.grid(True)

# 设置图2标题和标签
ax2.set_xlabel("Delta Torque")
ax2.set_ylabel("Membership Value")
ax2.grid(True)

# 更新函数，用于更新 torque 和 delta_torque 曲线
def update(idx):
    # 动态更新 mf_torque 和 sigma_torque
    data = loaded_data[idx:idx+42]
    data = data.reshape(-1, 6)
    mf_torque.assign(data[~np.isnan(data[:, 0]), 0])  # Remove NaN from mf_torque
    sigma_torque.assign(data[~np.isnan(data[:, 1]), 1])  # Remove NaN from sigma_torque
    mf_delta_torque.assign(data[~np.isnan(data[:, 2]), 2])  # Remove NaN from mf_delta_torque
    sigma_delta_torque.assign(data[~np.isnan(data[:, 3]), 3])  # Remove NaN from sigma_delta_torque
    # 计算 torque 部分
    torque_first = 1 / (1 + tf.exp(sigma_torque[0] * (torque - mf_torque[0])))
    torque_last = 1 / (1 + tf.exp(-sigma_torque[-1] * (torque - mf_torque[-1])))
    rul_error = tf.exp(-tf.square(tf.subtract(torque, mf_torque[1])) / tf.square(sigma_torque[1]))
    # 更新 torque 相关的曲线
    torque_first_line.set_data(torque_range, torque_first.numpy())
    torque_last_line.set_data(torque_range, torque_last.numpy())
    rul_error_line.set_data(torque_range, rul_error.numpy())

    # 计算 delta_torque 部分并更新对应曲线
    for i in range(7):
        if i == 0:
            rul_delta = 1 / (1 + tf.exp(sigma_delta_torque[i] * (delta_torque - mf_delta_torque[i])))
        elif i == 6:
            rul_delta = 1 / (1 + tf.exp(-sigma_delta_torque[i] * (delta_torque - mf_delta_torque[i])))
        else:
            rul_delta = tf.exp(-tf.square(tf.subtract(delta_torque, mf_delta_torque[i])) / tf.square(sigma_delta_torque[i]))
        
        # 更新 delta_torque 曲线
        delta_torque_lines[i].set_data(delta_torque_range, rul_delta.numpy())

    ax1.relim()  # 重新计算第1张图的坐标轴范围
    ax1.autoscale_view()  # 自动缩放视图
    
    ax2.relim()  # 重新计算第2张图的坐标轴范围
    ax2.autoscale_view()  # 自动缩放视图

    return torque_first_line, torque_last_line, rul_error_line, delta_torque_lines

# 动画设置
# print(int(len(loaded_data)/42)-100)
# print(int(len(loaded_data)/42)-225)
# loaded_data = loaded_data[475*42:600*42] #1kg down
# loaded_data = loaded_data[1000*42:1250*42] #1kg up
# loaded_data = loaded_data[0*125*42:1*125*42] #2kg up
ani1 = FuncAnimation(fig1, update, frames=np.arange(0, len(loaded_data)-42,420), interval=50, repeat=False)
ani2 = FuncAnimation(fig2, update, frames=np.arange(0, len(loaded_data)-42,420), interval=50, repeat=False)

plt.tight_layout()

# ani1.save('./gif/2kg_down_torque_animation.gif', writer=PillowWriter(fps=10))
# ani2.save('./gif/2kg_down_delta_torque_animation.gif', writer=PillowWriter(fps=10))
# 显示两个独立的图
plt.show()


# import tensorflow as tf
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation, PillowWriter

# # file_path = './exp/2024_10_08_2006/1/up_model.npy' #1kg
# # file_path = './exp/2024_10_10_1535/1/up_model.npy' #2kg
# file_path = './exp/2024_10_10_2038/1/up_model.npy' #2kg 5rule
# loaded_data = np.load(file_path)
# # print(len(loaded_data)/42)
# # # loaded_data = loaded_data[-42:]
# # print(loaded_data[6])
# # 初始化参数
# kg = 1
# if kg ==1 :
#     mf_torque = tf.Variable(np.asarray([50, 60, 70]), dtype=tf.float64)
#     # mf_torque.assign(np.asarray([0.2, 0.39, 0.58]))

#     sigma_torque = tf.Variable(np.asarray([0.5, 10, 0.5]), dtype=tf.float64)
#     mf_delta_torque = tf.Variable(np.asarray([-0.25, -0.15, 0, 0.15, 0.25]), dtype=tf.float64)
#     sigma_delta_torque = tf.Variable(np.asarray([130, 0.07, 0.07, 0.07, 130]), dtype=tf.float64)
#     # 定义 torque 和 delta_torque 的范围
#     torque_range = np.arange(30, 90, 0.1)
#     torque = tf.convert_to_tensor(torque_range, dtype=tf.float64)

#     delta_torque_range = np.arange(-0.3, 0.3, 0.001)
#     delta_torque = tf.convert_to_tensor(delta_torque_range, dtype=tf.float64)
# elif kg ==2:
#     mf_torque = tf.Variable(np.asarray([0.08,0.12, 0.16]), dtype=tf.float64)
#     # mf_torque.assign(np.asarray([0.2, 0.39, 0.58]))

#     sigma_torque = tf.Variable(np.asarray([100, 0.04, 100]), dtype=tf.float64)
#     mf_delta_torque = tf.Variable(np.asarray([-0.2, -0.15, -0.05, 0, 0.05, 0.15, 0.2]), dtype=tf.float64)
#     sigma_delta_torque = tf.Variable(np.asarray([150, 0.04, 0.04, 0.02, 0.04, 0.04, 150]), dtype=tf.float64)
#     # 定义 torque 和 delta_torque 的范围
#     torque_range = np.arange(0, 0.25, 0.001)
#     torque = tf.convert_to_tensor(torque_range, dtype=tf.float64)

#     delta_torque_range = np.arange(-0.3, 0.3, 0.001)
#     delta_torque = tf.convert_to_tensor(delta_torque_range, dtype=tf.float64)
#     [0.43,0.935, 1.44]

# # 初始化两个独立的 figure
# fig1, ax1 = plt.subplots(figsize=(10, 6))  # 第一个图: torque 曲线
# fig2, ax2 = plt.subplots(figsize=(10, 6))  # 第二个图: delta_torque 曲线

# # 第一张图：torque 曲线
# torque_first_line, = ax1.plot([], [], color='red')
# torque_last_line, = ax1.plot([], [], color='blue')
# rul_error_line, = ax1.plot([], [], color='green')

# # 第二张图：delta_torque 曲线
# delta_torque_lines = [ax2.plot([], [])[0] for _ in range(7)]  # 7条线

# # 设置图1标题和标签
# ax1.set_xlabel("Torque")
# ax1.set_ylabel("Membership Value")
# ax1.grid(True)

# # 设置图2标题和标签
# ax2.set_xlabel("Delta Torque")
# ax2.set_ylabel("Membership Value")
# ax2.grid(True)

# # 更新函数，用于更新 torque 和 delta_torque 曲线
# def update(idx):
#     # 动态更新 mf_torque 和 sigma_torque
#     data = loaded_data[idx:idx+42]
#     data = data.reshape(-1, 6)
#     mf_torque.assign(data[~np.isnan(data[:, 0]), 0])  # Remove NaN from mf_torque
#     sigma_torque.assign(data[~np.isnan(data[:, 1]), 1])  # Remove NaN from sigma_torque
#     mf_delta_torque.assign(data[~np.isnan(data[:, 2]), 2])  # Remove NaN from mf_delta_torque
#     sigma_delta_torque.assign(data[~np.isnan(data[:, 3]), 3])  # Remove NaN from sigma_delta_torque
#     # 计算 torque 部分
#     torque_first = 1 / (1 + tf.exp(sigma_torque[0] * (torque - mf_torque[0])))
#     torque_last = 1 / (1 + tf.exp(-sigma_torque[-1] * (torque - mf_torque[-1])))
#     rul_error = tf.exp(-tf.square(tf.subtract(torque, mf_torque[1])) / tf.square(sigma_torque[1]))
#     # 更新 torque 相关的曲线
#     torque_first_line.set_data(torque_range, torque_first.numpy())
#     torque_last_line.set_data(torque_range, torque_last.numpy())
#     rul_error_line.set_data(torque_range, rul_error.numpy())

#     # 计算 delta_torque 部分并更新对应曲线
#     for i in range(5):
#         if i == 0:
#             rul_delta = 1 / (1 + tf.exp(sigma_delta_torque[i] * (delta_torque - mf_delta_torque[i])))
#         elif i == 4:
#             rul_delta = 1 / (1 + tf.exp(-sigma_delta_torque[i] * (delta_torque - mf_delta_torque[i])))
#         else:
#             rul_delta = tf.exp(-tf.square(tf.subtract(delta_torque, mf_delta_torque[i])) / tf.square(sigma_delta_torque[i]))
        
#         # 更新 delta_torque 曲线
#         delta_torque_lines[i].set_data(delta_torque_range, rul_delta.numpy())

#     ax1.relim()  # 重新计算第1张图的坐标轴范围
#     ax1.autoscale_view()  # 自动缩放视图
    
#     ax2.relim()  # 重新计算第2张图的坐标轴范围
#     ax2.autoscale_view()  # 自动缩放视图

#     return torque_first_line, torque_last_line, rul_error_line, delta_torque_lines

# # 动画设置
# # print(int(len(loaded_data)/42)-100)
# # print(int(len(loaded_data)/42)-225)
# # loaded_data = loaded_data[475*42:600*42] #1kg down
# # loaded_data = loaded_data[1000*42:1250*42] #1kg up
# loaded_data = loaded_data[0*125*42:1*125*42] #2kg up
# ani1 = FuncAnimation(fig1, update, frames=np.arange(0, len(loaded_data)-42,42), interval=50, repeat=False)
# ani2 = FuncAnimation(fig2, update, frames=np.arange(0, len(loaded_data)-42,42), interval=50, repeat=False)

# plt.tight_layout()

# # ani1.save('./gif/2kg_down_5rule_torque_animation.gif', writer=PillowWriter(fps=10))
# # ani2.save('./gif/2kg_down_5rule_delta_torque_animation.gif', writer=PillowWriter(fps=10))
# # 显示两个独立的图
# plt.show()


