## 模型訓練過程
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

file_path = './exp/2024_10_21_2158/1/up_model.npy' #1kg
# file_path = './exp/2024_10_10_1535/1/up_model.npy' #2kg
# file_path = './exp/2024_10_10_1723/1/up_model.npy' #1kg
loaded_data = np.load(file_path)
# print(len(loaded_data)/42)
# # loaded_data = loaded_data[-42:]
# print(loaded_data[6])
# 初始化参数
kg = 1
if kg ==1 :
    mf_torque = tf.Variable(np.asarray([-9.5, -7, -3.5, 0, 3.5, 7, 9.5]), dtype=tf.float64)
    sigma_torque = tf.Variable(np.asarray([2, 1.5, 1.5, 1.5, 1.5, 1.5, 2]), dtype=tf.float64)

    mf_delta_torque = tf.Variable(np.asarray([-9.5, -7, -3.5, 0, 3.5, 7, 9.5]), dtype=tf.float64)
    sigma_delta_torque = tf.Variable(np.asarray([2, 1.5, 1.5, 1.5, 1.5, 1.5, 2]), dtype=tf.float64)
    # 定义 torque 和 delta_torque 的范围
    torque_range = np.arange(-12, 12, 0.01)
    torque = tf.convert_to_tensor(torque_range, dtype=tf.float64)

    delta_torque_range = np.arange(-12, 12, 0.01)
    delta_torque = tf.convert_to_tensor(delta_torque_range, dtype=tf.float64)


# 初始化两个独立的 figure
fig1, ax1 = plt.subplots(figsize=(10, 6))  # 第一个图: torque 曲线
fig2, ax2 = plt.subplots(figsize=(10, 6))  # 第二个图: delta_torque 曲线
ax1.margins(x=0, y=0.01)
ax2.margins(x=0, y=0.01)

# 第一张图：torque 曲线
torque_lines = [ax1.plot([], [])[0] for _ in range(7)]  # 7条线

# 第二张图：delta_torque 曲线
delta_torque_lines = [ax2.plot([], [])[0] for _ in range(7)]  # 7条线

# 设置图1标题和标签
ax1.set_xlabel("error")
ax1.set_ylabel("Membership Value")

ax1.grid(True)

# 设置图2标题和标签
ax2.set_xlabel("delta error")
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
    # torque_first = 1 / (1 + tf.exp(sigma_torque[0] * (torque - mf_torque[0])))
    # torque_last = 1 / (1 + tf.exp(-sigma_torque[-1] * (torque - mf_torque[-1])))
    # rul_error = tf.exp(-tf.square(tf.subtract(torque, mf_torque[1])) / tf.square(sigma_torque[1]))
    # # 更新 torque 相关的曲线
    # torque_first_line.set_data(torque_range, torque_first.numpy())
    # torque_last_line.set_data(torque_range, torque_last.numpy())
    # rul_error_line.set_data(torque_range, rul_error.numpy())

    # 计算 delta_torque 部分并更新对应曲线
    for i in range(7):
        if i == 0:
            rul = 1 / (1 + tf.exp(sigma_torque[i] * (torque - mf_torque[i])))
            rul_delta = 1 / (1 + tf.exp(sigma_delta_torque[i] * (delta_torque - mf_delta_torque[i])))
        elif i == 6:
            rul = 1 / (1 + tf.exp(-sigma_torque[i] * (torque - mf_torque[i])))
            rul_delta = 1 / (1 + tf.exp(-sigma_delta_torque[i] * (delta_torque - mf_delta_torque[i])))
        else:
            rul = tf.exp(-tf.square(tf.subtract(torque, mf_torque[i])) / tf.square(sigma_torque[i]))
            rul_delta = tf.exp(-tf.square(tf.subtract(delta_torque, mf_delta_torque[i])) / tf.square(sigma_delta_torque[i]))
        
        # 更新 delta_torque 曲线
        delta_torque_lines[i].set_data(delta_torque_range, rul_delta.numpy())
        torque_lines[i].set_data(torque_range, rul.numpy())

    ax1.relim()  # 重新计算第1张图的坐标轴范围
    ax1.autoscale_view()  # 自动缩放视图
    
    ax2.relim()  # 重新计算第2张图的坐标轴范围
    ax2.autoscale_view()  # 自动缩放视图

    return torque_lines, delta_torque_lines

# 动画设置
# print(int(len(loaded_data)/42)-100)
# print(int(len(loaded_data)/42)-225)
# loaded_data = loaded_data[475*42:600*42] #1kg down
# # loaded_data = loaded_data[1000*42:1250*42] #1kg up
# loaded_data = loaded_data[0*60*42:1*60*42] + loaded_data[1*125*42+0*60*42:1*125*42+1*60*42]+ loaded_data[2*125*42+0*60*42:2*125*42+1*60*42]
# loaded_data = np.concatenate([loaded_data[0*60*42:1*60*42], loaded_data[1*125*42:1*125*42+1*60*42],loaded_data[2*125*42:2*125*42+1*60*42],
#                                 loaded_data[3*125*42:3*125*42+1*60*42],loaded_data[4*125*42:4*125*42+1*60*42],
#                                  loaded_data[5*125*42:5*125*42+1*60*42],loaded_data[6*125*42:6*125*42+1*60*42],
#                                  loaded_data[7*125*42:7*125*42+1*60*42],loaded_data[8*125*42:8*125*42+1*60*42],
#                                  loaded_data[9*125*42:9*125*42+1*60*42],loaded_data[10*125*42:10*125*42+1*60*42]])

ani1 = FuncAnimation(fig1, update, frames=np.arange(0, len(loaded_data)-42,42*50), interval=50, repeat=False)
ani2 = FuncAnimation(fig2, update, frames=np.arange(0, len(loaded_data)-42,42*50), interval=50, repeat=False)

# plt.tight_layout()

# ani1.save('./gif/2kg_up_error_1FNN.gif', writer=PillowWriter(fps=10))
# ani2.save('./gif/2kg_up_delta_error_1FNN.gif', writer=PillowWriter(fps=10))
# 显示两个独立的图
plt.show()
