import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy import interpolate

# 读取两个 NumPy 数组
EXP_DIR   = './exp'
data_date = '2024_08_22_1836'

voltage_path = f'{EXP_DIR}/{data_date}/1/voltage.npy'
angle_path = f'{EXP_DIR}/{data_date}/1/angle.npy'
voltage = np.load(voltage_path)
angle = np.load(angle_path)
voltage = voltage[1+25*4:]  # 删除第一个元素（如果需要）
angle = angle[1+25*4:]      # 删除第一个元素（如果需要）

# 输出数据长度
print(len(voltage))
print(len(angle))

# 假设采样率为 20 Hz
sampling_rate = 25  # 20 Hz

# 根据实际数据生成时间轴
duration = len(angle) / sampling_rate  # 根据 angle 数据长度计算持续时间
time = np.linspace(0, duration, num=len(angle))

# 生成一个sin_wave，并从第10秒开始循环，前面用30填充
sine_angle = np.asarray([30,30,30,30,30,30,30,30,30,30,30,30,30,30,31,31,31,31,31,31,32,32,32,32,33,33,33,34,34,35,35,35,36,36,37,37,38,38,39,39,40,41,41,42,42,43,44,44,45,45,46,47,48,48,49,50,50,51,52,53,53,54,55,55,56,57,58,58,59,60,60,61,62,62,63,64,65,65,66,67,67,68,69,70,70,71,72,72,73,74,75,75,76,76,77,78,78,79,79,80,81,81,82,82,83,83,84,84,85,85,85,86,86,87,87,87,88,88,88,88,89,89,89,89,89,89,90,90,90,90,90,90,90,90,90,90,90,90,90,90,89,89,89,89,89,88,88,88,88,87,87,87,86,86,86,85,85,84,84,83,83,82,82,81,81,80,80,79,79,78,77,77,76,75,75,74,74,73,72,71,71,70,69,69,68,67,66,66,65,64,63,63,62,61,60,60,59,58,57,57,56,55,54,54,53,52,51,51,50,49,49,48,47,46,46,45,45,44,43,43,42,41,41,40,40,39,39,38,38,37,37,36,36,35,35,34,34,34,33,33,33,32,32,32,32,31,31,31,31,31])
x = np.linspace(0, 1, 250)
f = interpolate.interp1d(x, sine_angle)
x_10 = np.linspace(0, 1, 250)
x_8 = np.linspace(0, 1, 200)
x_6 = np.linspace(0, 1, 150)
x_4 = np.linspace(0, 1, 100)
period_10 = f(x_10)
period_8 = f(x_8)
period_6 = f(x_6)
period_4 = f(x_4)


# 生成与 angle 数据长度相同的 sin_wave
sin_wave = np.zeros(len(angle))

# # 前10秒填充30度
# sin_wave[:int(4 * sampling_rate)] = 30

# # 从10秒开始循环 sine_angle
# sin_wave[int(4 * sampling_rate):] = np.tile(sine_angle, int(np.ceil((len(angle) - int(4 * sampling_rate)) / len(sine_angle))))[:len(angle) - int(4 * sampling_rate)]


# 生成与 angle 数据长度相同的 sin_wave
target_wave = period_10
sin_wave = np.tile(target_wave, int(np.ceil(len(angle) / len(target_wave))))[:len(angle)]



# 创建一个新的图形对象
plt.figure(figsize=(10, 8))

# 在第一个子图中绘制 voltage 数组
plt.subplot(2, 1, 1)  # (行数, 列数, 子图编号)
plt.plot(time, voltage)
plt.ylabel('Voltage')
plt.grid(True)

# 在第二个子图中绘制 angle 数据和叠加了 sin_wave 的数据
plt.subplot(2, 1, 2)  # (行数, 列数, 子图编号)
plt.plot(time, angle, label='Angle')
plt.plot(time, sin_wave, color=(0, 0, 0, 0.5), linestyle='--', label='Desire Angle')
plt.xlabel('Time (seconds)')
plt.ylabel('Angle')
plt.grid(True)
plt.legend()

# 调整布局，使子图不重叠
plt.tight_layout()


# # 计算加速度（角度的二阶导数）
# velocity = np.gradient(angle, time)  # 计算速度（角度的一阶导数）
# acceleration = np.gradient(velocity, time)  # 计算加速度

# # 设计低通滤波器
# def butter_lowpass_filter(data, cutoff, fs, order=4):
#     nyquist = 0.5 * fs
#     normal_cutoff = cutoff / nyquist
#     b, a = butter(order, normal_cutoff, btype='low', analog=False)
#     y = filtfilt(b, a, data)
#     return y

# # 应用低通滤波器
# cutoff_frequency = 1  # 设置低通滤波器的截止频率（Hz）
# filtered_acceleration = butter_lowpass_filter(acceleration, cutoff_frequency, sampling_rate)



# # 创建第二个图形对象 fig2，绘制加速度
# fig2 = plt.figure(figsize=(10, 6))

# # 绘制原始加速度和滤波后的加速度图
# # plt.plot(time, acceleration, label='Original Acceleration', color='r', linestyle='--')
# plt.plot(time, filtered_acceleration, label='Filtered Acceleration', color='b')
# plt.plot(time, angle, label='Angle')
# plt.xlabel('Time (seconds)')
# plt.ylabel('Acceleration')
# plt.grid(True)
# plt.legend()

# # 调整布局
# plt.tight_layout()

# 显示图形
plt.show()