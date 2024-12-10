import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy import interpolate
import os

from encoder_function import moving_average_with_padding

# 读取两个 NumPy 数组
EXP_DIR   = './sineWaveHistory'
EXP_DIR   = './exp'

# 0kg
data_date_0 = '2024_10_24_1706' #0kg1FNN
data_date_02 = '2024_10_24_1704'
data_date_1 = '2024_10_24_1701'
data_date_12 = '2024_10_24_1702'
data_date_2 = '2024_10_24_1658'
data_date_22 = '2024_10_24_1655'

angle_path_0 = f'{EXP_DIR}/{data_date_0}/1/angle.npy'
angle_path_02 = f'{EXP_DIR}/{data_date_02}/1/angle.npy'
angle_path_1 = f'{EXP_DIR}/{data_date_1}/1/angle.npy'
angle_path_12 = f'{EXP_DIR}/{data_date_12}/1/angle.npy'
angle_path_2 = f'{EXP_DIR}/{data_date_2}/1/angle.npy'
angle_path_22 = f'{EXP_DIR}/{data_date_22}/1/angle.npy'

angle_0 = np.load(angle_path_0)
angle_02 = np.load(angle_path_02)
angle_1 = np.load(angle_path_1)
angle_12 = np.load(angle_path_12)
angle_2 = np.load(angle_path_2)
angle_22 = np.load(angle_path_22)

angle_0 = angle_0[25*4:-1]
angle_02 = angle_02[25*4:-1]
angle_1 = angle_1[25*4:-1]
angle_12 = angle_12[25*4:-1]
angle_2 = angle_2[25*4:-1]
angle_22 = angle_22[25*4:-1]


# print("force : ",len(force))

# 假设采样率为 20 Hz
sampling_rate = 25  # 20 Hz

# 根据实际数据生成时间轴
duration = len(angle_0) / sampling_rate  # 根据 angle 数据长度计算持续时间
time = np.linspace(0, duration, num=len(angle_0))

# 生成一个sin_wave，并从第10秒开始循环，前面用30填充
first_part = 125
middle_part = 250
last_part = 125

# 创建列表
data = [30] * first_part + [90.2] * middle_part + [30] * last_part

def calculate_rise_time(y, sampling_rate, start=0.05, end=0.95):
    """计算从10%到90%的上升时间"""
    final_value = y[-1]
    start_value = 33
    end_value = 87
    start_index = np.where(y >= start_value)[0][0]
    end_index = np.where(y >= end_value)[0][0]
    rise_time = (end_index - start_index) / sampling_rate
    return rise_time

def calculate_fall_time(y, sampling_rate, start=0.9, end=0.1):
    """计算从90%到10%的下降时间"""
    # 这里假设方波的高点为峰值
    peak_value = y[0]
    start_value = 87
    end_value = 33
    
    # 从y数组中找到第一次降至start_value以下的索引
    try:
        start_index = np.where(y <= start_value)[0][0]
        # 从start_index之后找到第一次降至end_value以下的索引
        end_index = np.where(y[start_index:] <= end_value)[0][0] + start_index
        fall_time = (end_index - start_index) / sampling_rate
        return fall_time
    except IndexError:
        return None  # 如果找不到符合条件的点，返回None

def calculate_overshoot(y):
    """计算超调量"""
    overshoot = np.max(y)
    # overshoot = ((peak_value - set_point) / set_point) * 100 if set_point != 0 else 0
    return overshoot

angles = [angle_0, angle_02, angle_1, angle_12, angle_2, angle_22]
labels = ["0kg_FNN1", "0kg_FNN2", "1kg_FNN1", "1kg_FNN2", "2kg_FNN1", "2kg_FNN2"]


for angle, label in zip(angles, labels):
    rt = calculate_rise_time(angle[:int(len(angle)/2)], sampling_rate)
    rt_f = calculate_fall_time(angle[int(len(angle)/2):], sampling_rate)
    overshoot = calculate_overshoot(angle)
    print(f"{label} - 上升时间: {rt:.2f} 秒, 超调量: {overshoot:.2f}%")
    print(f"{label} - 下降时间: {rt_f:.2f} 秒")


plt.figure(figsize=(10, 8))
# angle_2 = moving_average_with_padding(angle_2,5)
plt.plot(time, data,'k--')
# plt.plot(time, angle,label="Ori" ,color='green')
plt.plot(time, angle_0,label="0kg_FNN1", color='blue')
plt.plot(time, angle_02,label="0kg_FNN2")
plt.plot(time, angle_1,label="1kg_FNN1")
plt.plot(time, angle_12,label="1kg_FNN2")
plt.plot(time, angle_2,label="2kg_FNN1")
plt.plot(time, angle_22,label="2kg_FNN2")
# plt.plot(time, angle_2,label="2-FNN", color='red')
# plt.plot(time, angle,label="Cycle 1" ,color='green')
# plt.plot(time, angle_1,label="Cycle 6", color='blue')
# plt.plot(time, angle_2,label="Cycle 40", color='red')

plt.xlabel('Time (seconds)')
plt.ylabel('Angle (Deg)')
plt.grid(True)
# plt.legend(loc='upper right')

plt.show()