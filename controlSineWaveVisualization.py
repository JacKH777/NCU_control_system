# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import butter, filtfilt
# from scipy import interpolate
# import os

# from encoder_function import moving_average_with_padding

# # 读取两个 NumPy 数组
# EXP_DIR   = './sineWaveHistory'
# EXP_DIR   = './exp'
# data_date = '2024_09_26_2158'
# # data_date = '2024_10_11_0945'
# # data_date = '2024_10_11_1007'
# # data_date = '2024_10_12_1153'
# # data_date = '2024_10_15_2037' #1kg 10sec 2FNN
# # data_date = '2024_10_15_2044' #1kg 6sec 2FNN
# # data_date_1 = '2024_10_15_2110' #1kg 10sec 1FNN
# # data_date_1 = '2024_10_15_2116' #1kg 6sec 1FNN
# data_date_1 = '2024_10_16_1608' #2kg 10sec 1FNN
# data_date = '2024_10_16_1614' #2kg 10sec 2FNN
# # data_date = '2024_10_16_1620' #2kg 6sec 2FNN
# # data_date_1 = '2024_10_16_1631' #2kg 6sec 1FNN

# data_date_1 = '2024_10_17_1843' #jack 10sec 1FNN
# data_date = '2024_10_17_1543' #jack 10sec 2FNN
# voltage_path = f'{EXP_DIR}/{data_date}/1/voltage.npy'
# angle_path = f'{EXP_DIR}/{data_date}/1/angle.npy'
# voltage_path_1 = f'{EXP_DIR}/{data_date_1}/1/voltage.npy'
# angle_path_1 = f'{EXP_DIR}/{data_date_1}/1/angle.npy'
# # force_path = f'{EXP_DIR}/{data_date}/1/force.npy'
# voltage = np.load(voltage_path)
# angle = np.load(angle_path)
# voltage_1 = np.load(voltage_path_1)
# angle_1 = np.load(angle_path_1)
# # force = np.load(force_path)
# # voltage = voltage[25*4:25*4+1500]  # 删除第一个元素（如果需要）
# # angle = angle[25*4:25*4+1500]      # 删除第一个元素（如果需要）
# # voltage_1 = voltage_1[25*4:25*4+1500]  # 删除第一个元素（如果需要）
# # angle_1 = angle_1[25*4:25*4+1500]      # 删除第一个元素（如果需要）
# voltage = voltage[25*4:-1]  # 删除第一个元素（如果需要）
# angle = angle[25*4:-1]      # 删除第一个元素（如果需要）
# voltage_1 = voltage_1[25*4:-1]  # 删除第一个元素（如果需要）
# angle_1 = angle_1[25*4:-1]      # 删除第一个元素（如果需要）
# # force = force[1+25*4:] 

# # 输出数据长度
# print("voltage : ",len(voltage))
# print("angle : ",len(angle))
# # print("force : ",len(force))

# # 假设采样率为 20 Hz
# sampling_rate = 25  # 20 Hz

# # 根据实际数据生成时间轴
# duration = len(angle) / sampling_rate  # 根据 angle 数据长度计算持续时间
# time = np.linspace(0, duration, num=len(angle))

# # 生成一个sin_wave，并从第10秒开始循环，前面用30填充
# sine_angle = np.asarray([30,30,30,30,30,30,30,30,30,30,30,30,30,30,31,31,31,31,31,31,32,32,32,32,33,33,33,34,34,35,35,35,36,36,37,37,38,38,39,39,40,41,41,42,42,43,44,44,45,45,46,47,48,48,49,50,50,51,52,53,53,54,55,55,56,57,58,58,59,60,60,61,62,62,63,64,65,65,66,67,67,68,69,70,70,71,72,72,73,74,75,75,76,76,77,78,78,79,79,80,81,81,82,82,83,83,84,84,85,85,85,86,86,87,87,87,88,88,88,88,89,89,89,89,89,89,90,90,90,90,90,90,90,90,90,90,90,90,90,90,89,89,89,89,89,88,88,88,88,87,87,87,86,86,86,85,85,84,84,83,83,82,82,81,81,80,80,79,79,78,77,77,76,75,75,74,74,73,72,71,71,70,69,69,68,67,66,66,65,64,63,63,62,61,60,60,59,58,57,57,56,55,54,54,53,52,51,51,50,49,49,48,47,46,46,45,45,44,43,43,42,41,41,40,40,39,39,38,38,37,37,36,36,35,35,34,34,34,33,33,33,32,32,32,32,31,31,31,31,31])
# x = np.linspace(0, 1, 250)
# f = interpolate.interp1d(x, sine_angle)
# x_10 = np.linspace(0, 1, 250)
# x_8 = np.linspace(0, 1, 200)
# x_6 = np.linspace(0, 1, 150)
# x_4 = np.linspace(0, 1, 100)
# period_10 = f(x_10)
# period_8 = f(x_8)
# period_6 = f(x_6)
# period_4 = f(x_4)


# # 生成与 angle 数据长度相同的 sin_wave
# sin_wave = np.zeros(len(angle))

# # # 前10秒填充30度
# # sin_wave[:int(4 * sampling_rate)] = 30

# # # 从10秒开始循环 sine_angle
# # sin_wave[int(4 * sampling_rate):] = np.tile(sine_angle, int(np.ceil((len(angle) - int(4 * sampling_rate)) / len(sine_angle))))[:len(angle) - int(4 * sampling_rate)]


# # 生成与 angle 数据长度相同的 sin_wave
# target_wave = period_10
# sin_wave = np.tile(target_wave, int(np.ceil(len(angle) / len(target_wave))))[:len(angle)]
# # sin_wave = np.tile(target_wave,10)[:len(angle)]
# # 分段函数
# def split_data(data, segments):
#     # n = len(data)
#     return np.array_split(data, segments)

# def rmse(predictions, targets):
#     return np.sqrt(((predictions - targets) ** 2).mean())

# # 将实际值和预测值分成10段
# angle_10 = split_data(angle, 10)
# sin_wave_10 = split_data(sin_wave, 10)
# voltage_10 = split_data(voltage, 10)

# angle_10_1 = split_data(angle_1, 10)
# voltage_10_1 = split_data(voltage_1, 10)

# sin_wave_10 = np.array(sin_wave_10)
# angle_10_1 = np.array(angle_10_1)
# error_10_1 = sin_wave_10 - angle_10_1
# rmses_1 = [rmse(pred, act) for pred, act in zip(sin_wave_10, angle_10_1)]
# # ## force 濾波
# # def moving_average_with_padding(data, window_size):
# #     # 计算填充大小
# #     pad_size = window_size // 2
    
# #     # 使用反射填充模式在数据前后进行填充
# #     padded_data = np.pad(data, pad_size, mode='reflect')
    
# #     # 对填充后的数据进行卷积
# #     convolved_data = np.convolve(padded_data, np.ones(window_size) / window_size, mode='valid')
    
# #     # 移除前后填充的部分，保证输出与输入长度相同
# #     return convolved_data

# # # force = moving_average_with_padding(force, 25)
# # def get_torque(angle,force):
# #         lenth = np.sqrt(0.20**2+0.255**2-2*0.20*0.255*np.cos(np.radians(180 - angle)))
# #         angle_of_force_radians = np.arcsin(0.20*np.sin(np.radians(180 - angle))/lenth)
# #         torque = force * np.sin(angle_of_force_radians)*0.25
# #         return torque
# # # force =  force * 0.32
# # # lenth = np.sqrt(0.20**2+0.255**2-2*0.20*0.255*np.cos(np.radians(180 - angle)))
# # # angle_of_force_redius = np.arcsin(0.20*np.sin(np.radians(180 - angle))/lenth)
# # # torque = force * np.sin(angle_of_force_redius)*0.255
# # # force = force*0.3
# # # torque = get_torque(angle,force)
# # # torque = moving_average_with_padding(torque, 25)

# # angle = moving_average_with_padding(angle,11)
# # voltage = moving_average_with_padding(voltage,11)
# # 创建一个新的图形对象
# sin_wave_10 = np.array(sin_wave_10)
# angle_10 = np.array(angle_10)
# error_10 = sin_wave_10 - angle_10
# rmses = [rmse(pred, act) for pred, act in zip(sin_wave_10, angle_10)]
# error = np.concatenate((error_10))
# # error_diff = np.diff(error_10)
# error_diff = [np.diff(segment)  for segment in error_10]
# # delta_error = np.insert(error_diff, 0, 0)  # 在差分误差前面插入一个0
# delta_error = [np.insert(segment, 0, 0)  for segment in error_diff]
# delta_error = [moving_average_with_padding(segment, 25)*25  for segment in delta_error]
# delta_error = np.concatenate((delta_error))
# delta_voltage = [np.diff(segment)  for segment in voltage_10]
# delta_voltage = [np.insert(segment, 0, 0)  for segment in delta_voltage]
# # print(len(delta_error))
# # print(error_10)
# # delta_error = np.diff(error)
# # delta_error = np.insert(delta_error, 0, 0)  # 在差分误差前面插入一个0
# # # delta_error = moving_average_with_padding(delta_error, 25) *25
# # delta_error = delta_error *25

# plt.figure(figsize=(10, 8))

# a = 10
# # 在第一个子图中绘制 voltage 数组
# plt.subplot(4, 1, 1)  # (行数, 列数, 子图编号)
# # plt.plot(time, voltage)
# plt.scatter(time, error,s=5)
# plt.ylabel('Error (Deg)')
# plt.axvline(x=a*1, color='gray', linestyle='--')
# plt.axvline(x=a*2, color='gray', linestyle='--')
# plt.axvline(x=a*3, color='gray', linestyle='--')
# plt.axvline(x=a*4, color='gray', linestyle='--')
# plt.axvline(x=a*5, color='gray', linestyle='--')
# plt.axvline(x=a*6, color='gray', linestyle='--')
# plt.axvline(x=a*7, color='gray', linestyle='--')
# plt.axvline(x=a*8, color='gray', linestyle='--')
# plt.axvline(x=a*9, color='gray', linestyle='--')
# plt.grid(True)

# plt.subplot(4, 1, 2)  # (行数, 列数, 子图编号)
# # plt.plot(time, voltage)
# plt.scatter(time,delta_error,s=5)
# plt.ylabel('Delta Error (Deg/s)')
# plt.axvline(x=a*1, color='gray', linestyle='--')
# plt.axvline(x=a*2, color='gray', linestyle='--')
# plt.axvline(x=a*3, color='gray', linestyle='--')
# plt.axvline(x=a*4, color='gray', linestyle='--')
# plt.axvline(x=a*5, color='gray', linestyle='--')
# plt.axvline(x=a*6, color='gray', linestyle='--')
# plt.axvline(x=a*7, color='gray', linestyle='--')
# plt.axvline(x=a*8, color='gray', linestyle='--')
# plt.axvline(x=a*9, color='gray', linestyle='--')
# plt.grid(True)

# plt.subplot(4, 1, 3)  # (行数, 列数, 子图编号)
# # plt.plot(time, voltage)
# plt.scatter(time, voltage,s=5)
# plt.ylabel('Voltage (V)')
# plt.axvline(x=a*1, color='gray', linestyle='--')
# plt.axvline(x=a*2, color='gray', linestyle='--')
# plt.axvline(x=a*3, color='gray', linestyle='--')
# plt.axvline(x=a*4, color='gray', linestyle='--')
# plt.axvline(x=a*5, color='gray', linestyle='--')
# plt.axvline(x=a*6, color='gray', linestyle='--')
# plt.axvline(x=a*7, color='gray', linestyle='--')
# plt.axvline(x=a*8, color='gray', linestyle='--')
# plt.axvline(x=a*9, color='gray', linestyle='--')
# plt.grid(True)

# # plt.subplot(4, 1, 3)  # (行数, 列数, 子图编号)
# # # plt.plot(time, voltage)
# # plt.scatter(time, voltage,s=5)
# # plt.ylabel('Voltage (V)')
# # plt.axvline(x=a*1, color='gray', linestyle='--')
# # plt.axvline(x=a*2, color='gray', linestyle='--')
# # plt.axvline(x=a*3, color='gray', linestyle='--')
# # plt.axvline(x=a*4, color='gray', linestyle='--')
# # plt.axvline(x=a*5, color='gray', linestyle='--')
# # plt.axvline(x=a*6, color='gray', linestyle='--')
# # plt.axvline(x=a*7, color='gray', linestyle='--')
# # plt.axvline(x=a*8, color='gray', linestyle='--')
# # plt.axvline(x=a*9, color='gray', linestyle='--')
# # plt.grid(True)

# # 在第二个子图中绘制 angle 数据和叠加了 sin_wave 的数据
# plt.subplot(4, 1, 4)  # (行数, 列数, 子图编号)
# # plt.plot(time, angle, label='Angle')
# plt.scatter(time, angle, s=10, label='Angle')
# plt.plot(time, sin_wave, color=(0, 0, 0, 0.5), linestyle='--', label='Desire Angle')
# plt.axvline(x=a*1, color='gray', linestyle='--')
# plt.axvline(x=a*2, color='gray', linestyle='--')
# plt.axvline(x=a*3, color='gray', linestyle='--')
# plt.axvline(x=a*4, color='gray', linestyle='--')
# plt.axvline(x=a*5, color='gray', linestyle='--')
# plt.axvline(x=a*6, color='gray', linestyle='--')
# plt.axvline(x=a*7, color='gray', linestyle='--')
# plt.axvline(x=a*8, color='gray', linestyle='--')
# plt.axvline(x=a*9, color='gray', linestyle='--')
# plt.ylabel('Angle (Deg)')
# plt.xlabel('Time (Sec)')
# plt.grid(True)
# plt.legend()

# # plt.subplot(4, 1, 3)  # (行数, 列数, 子图编号)
# # plt.plot(time,torque)
# # plt.ylabel('Torque (N*m)')
# # plt.grid(True)

# # plt.subplot(4, 1, 4)  # (行数, 列数, 子图编号)
# # plt.plot(time, moving_average_with_padding(force, 25))
# # plt.xlabel('Time (seconds)')
# # plt.ylabel('Force (N)')
# # plt.grid(True)

# plt.tight_layout()

# # plt.figure(figsize=(10, 8))
# # plt.plot(time, force)
# # plt.ylabel('Force')
# # plt.grid(True)

# plt.figure(figsize=(10, 8))
# plt.plot(range(1,len(rmses)+1),rmses,label = '2-FNN') 
# plt.plot(range(1,len(rmses)+1),rmses_1,label = '1-FNN') 
# plt.grid(True)
# plt.xticks(range(1,len(rmses)+1))
# plt.ylabel('RMSE (Deg)')
# plt.xlabel('Cycle')
# plt.legend(loc='upper left')
# # # 调整布局，使子图不重叠
# # plt.tight_layout()
# plt.show()

# # ## 生成一個sin
# # period_length = len(target_wave)
# # start_index = len(sin_wave) - 3 * period_length

# # # 提取倒数第二个波形段
# # penultimate_wave = angle[start_index:start_index + int(period_length)]
# # penultimate_wave_v = voltage[start_index:start_index + int(period_length)]
# # save_dir = './combine4To10SecSineWave/1kg_angle'
# # save_path = os.path.join(save_dir, 'sine_1kg_4.npy')
# # save_dir_v = './combine4To10SecSineWave/1kg_voltage'
# # save_path_v = os.path.join(save_dir_v, 'sine_1kg_4.npy')
# # np.save(save_path, penultimate_wave)
# # np.save(save_path_v, penultimate_wave_v)

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy import interpolate
import os

from encoder_function import moving_average_with_padding

# 读取两个 NumPy 数组
EXP_DIR   = './sineWaveHistory'
EXP_DIR   = './exp'
data_date = '2024_09_26_2158'
# data_date = '2024_10_11_0945'
# data_date = '2024_10_11_1007'
# data_date = '2024_10_12_1153'
# data_date = '2024_10_15_2037' #1kg 10sec 2FNN
# data_date = '2024_10_15_2044' #1kg 6sec 2FNN
# data_date_1 = '2024_10_15_2110' #1kg 10sec 1FNN
# data_date_1 = '2024_10_15_2116' #1kg 6sec 1FNN
# data_date_1 = '2024_10_16_1608' #2kg 10sec 1FNN
# data_date = '2024_10_16_1614' #2kg 10sec 2FNN
# data_date = '2024_10_16_1620' #2kg 6sec 2FNN
# data_date_1 = '2024_10_16_1631' #2kg 6sec 1FNN

data_date_1 = '2024_10_20_1234' #jack 10sec 1FNN
# data_date_1 = '2024_10_20_1217' #1kg 10*30 2FNN
# data_date_1 = '2024_10_20_1144' #0kg 10*30 2FNN
data_date = '2024_10_20_1309' #2kg 10*30 1FNN
# data_date = '2024_10_20_1321' #1kg 10*30 1FNN
# data_date = '2024_10_20_1332' #0kg 10*30 1FNN

##20241022
data_date_1 = '2024_10_21_1919' #0kg 10*40 2FNN
data_date = '2024_10_21_2042' #0kg 10*40 1FNN
# data_date = '2024_10_21_2103' #1kg 10*40 1FNN
# data_date_1 = '2024_10_21_2115' #1kg 10*40 2FNN
data_date = '2024_10_21_2158' #2kg 10*40 1FNN
data_date_1 = '2024_10_21_2143' #2kg 10*40 2FNN


# data_date = '2024_10_21_1919' #0kg 10*40 2FNN
# data_date_1 = '2024_11_04_0853' #0kg 10*40 2FNN 0.01learning Rate
data_date_2 = '2024_11_04_0932' #0kg 10*40 2FNN 0.0001learning Rate

voltage_path = f'{EXP_DIR}/{data_date}/1/voltage.npy'
angle_path = f'{EXP_DIR}/{data_date}/1/angle.npy'
voltage_path_1 = f'{EXP_DIR}/{data_date_1}/1/voltage.npy'
angle_path_1 = f'{EXP_DIR}/{data_date_1}/1/angle.npy'

voltage_path_2 = f'{EXP_DIR}/{data_date_2}/1/voltage.npy'
angle_path_2 = f'{EXP_DIR}/{data_date_2}/1/angle.npy'
# force_path = f'{EXP_DIR}/{data_date}/1/force.npy'
voltage = np.load(voltage_path)
angle = np.load(angle_path)
voltage_1 = np.load(voltage_path_1)
angle_1 = np.load(angle_path_1)
voltage_2 = np.load(voltage_path_2)
angle_2 = np.load(angle_path_2)
# force = np.load(force_path)
# voltage = voltage[25*4:25*4+1500]  # 删除第一个元素（如果需要）
# angle = angle[25*4:25*4+1500]      # 删除第一个元素（如果需要）
# voltage_1 = voltage_1[25*4:25*4+1500]  # 删除第一个元素（如果需要）
# angle_1 = angle_1[25*4:25*4+1500]      # 删除第一个元素（如果需要）
voltage = voltage[25*4:-1]  # 删除第一个元素（如果需要）
angle = angle[25*4:-1]      # 删除第一个元素（如果需要）
voltage_1 = voltage_1[25*4:-1]  # 删除第一个元素（如果需要）
angle_1 = angle_1[25*4:-1]      # 删除第一个元素（如果需要）
voltage_2 = voltage_2[25*4:-1]  # 删除第一个元素（如果需要）
angle_2 = angle_2[25*4:-1]      # 删除第一个元素（如果需要）
# force = force[1+25*4:] 

# 输出数据长度
print("voltage : ",len(voltage))
print("angle : ",len(angle))
# print("force : ",len(force))

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
# sin_wave = np.tile(target_wave,10)[:len(angle)]
# 分段函数
def split_data(data, segments):
    # n = len(data)
    return np.array_split(data, segments)

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

# 将实际值和预测值分成10段
angle_10 = split_data(angle, 40)
sin_wave_10 = split_data(sin_wave, 40)
voltage_10 = split_data(voltage, 40)

angle_10_1 = split_data(angle_1, 40)
# voltage_10_1 = split_data(voltage_1, 10)

sin_wave_10 = np.array(sin_wave_10)
angle_10_1 = np.array(angle_10_1)
error_10_1 = sin_wave_10 - angle_10_1
rmses_1 = [rmse(pred, act) for pred, act in zip(sin_wave_10, angle_10_1)]

angle_10_2 = split_data(angle_2, 40)
angle_10_2 = np.array(angle_10_2)
error_10_2 = sin_wave_10 - angle_10_2
rmses_2 = [rmse(pred, act) for pred, act in zip(sin_wave_10, angle_10_2)]
# ## force 濾波
# def moving_average_with_padding(data, window_size):
#     # 计算填充大小
#     pad_size = window_size // 2
    
#     # 使用反射填充模式在数据前后进行填充
#     padded_data = np.pad(data, pad_size, mode='reflect')
    
#     # 对填充后的数据进行卷积
#     convolved_data = np.convolve(padded_data, np.ones(window_size) / window_size, mode='valid')
    
#     # 移除前后填充的部分，保证输出与输入长度相同
#     return convolved_data

# # force = moving_average_with_padding(force, 25)
# def get_torque(angle,force):
#         lenth = np.sqrt(0.20**2+0.255**2-2*0.20*0.255*np.cos(np.radians(180 - angle)))
#         angle_of_force_radians = np.arcsin(0.20*np.sin(np.radians(180 - angle))/lenth)
#         torque = force * np.sin(angle_of_force_radians)*0.25
#         return torque
# # force =  force * 0.32
# # lenth = np.sqrt(0.20**2+0.255**2-2*0.20*0.255*np.cos(np.radians(180 - angle)))
# # angle_of_force_redius = np.arcsin(0.20*np.sin(np.radians(180 - angle))/lenth)
# # torque = force * np.sin(angle_of_force_redius)*0.255
# # force = force*0.3
# # torque = get_torque(angle,force)
# # torque = moving_average_with_padding(torque, 25)

# angle = moving_average_with_padding(angle,11)
# voltage = moving_average_with_padding(voltage,11)
# 创建一个新的图形对象
sin_wave_10 = np.array(sin_wave_10)
angle_10 = np.array(angle_10)
error_10 = sin_wave_10 - angle_10
rmses = [rmse(pred, act) for pred, act in zip(sin_wave_10, angle_10)]

mean = np.mean(rmses[-10:])  # 计算平均值
std_dev = np.std(rmses[-10:])  # 计算标准差

mean_1 = np.mean(rmses_1[-10:])  # 计算平均值
std_dev_1 = np.std(rmses_1[-10:])  # 计算标准差
# 格式化输出
print(f"FNN1中心值: {mean:.4f}, 正负: ±{std_dev:.4f}")
print(f"FNN中心值: {mean_1:.4f}, 正负: ±{std_dev_1:.4f}")
error = np.concatenate((error_10))
# error_diff = np.diff(error_10)
error_diff = [np.diff(segment)  for segment in error_10]
# delta_error = np.insert(error_diff, 0, 0)  # 在差分误差前面插入一个0
delta_error = [np.insert(segment, 0, 0)  for segment in error_diff]
delta_error = [moving_average_with_padding(segment, 25)*25  for segment in delta_error]
delta_error = np.concatenate((delta_error))
delta_voltage = [np.diff(segment)  for segment in voltage_10]
delta_voltage = [np.insert(segment, 0, 0)  for segment in delta_voltage]
# print(len(delta_error))
# print(error_10)
# delta_error = np.diff(error)
# delta_error = np.insert(delta_error, 0, 0)  # 在差分误差前面插入一个0
# # delta_error = moving_average_with_padding(delta_error, 25) *25
# delta_error = delta_error *25

plt.figure(figsize=(10, 8))

a = 10
# 在第一个子图中绘制 voltage 数组
plt.subplot(4, 1, 1)  # (行数, 列数, 子图编号)
# plt.plot(time, voltage)
plt.scatter(time, error,s=5)
plt.ylabel('Error (Deg)')
plt.axvline(x=a*1, color='gray', linestyle='--')
plt.axvline(x=a*2, color='gray', linestyle='--')
plt.axvline(x=a*3, color='gray', linestyle='--')
plt.axvline(x=a*4, color='gray', linestyle='--')
plt.axvline(x=a*5, color='gray', linestyle='--')
plt.axvline(x=a*6, color='gray', linestyle='--')
plt.axvline(x=a*7, color='gray', linestyle='--')
plt.axvline(x=a*8, color='gray', linestyle='--')
plt.axvline(x=a*9, color='gray', linestyle='--')
plt.grid(True)

plt.subplot(4, 1, 2)  # (行数, 列数, 子图编号)
# plt.plot(time, voltage)
plt.scatter(time,delta_error,s=5)
plt.ylabel('Delta Error (Deg/s)')
plt.axvline(x=a*1, color='gray', linestyle='--')
plt.axvline(x=a*2, color='gray', linestyle='--')
plt.axvline(x=a*3, color='gray', linestyle='--')
plt.axvline(x=a*4, color='gray', linestyle='--')
plt.axvline(x=a*5, color='gray', linestyle='--')
plt.axvline(x=a*6, color='gray', linestyle='--')
plt.axvline(x=a*7, color='gray', linestyle='--')
plt.axvline(x=a*8, color='gray', linestyle='--')
plt.axvline(x=a*9, color='gray', linestyle='--')
plt.grid(True)

plt.subplot(4, 1, 3)  # (行数, 列数, 子图编号)
# plt.plot(time, voltage)
plt.scatter(time, voltage,s=5)
plt.ylabel('Voltage (V)')
plt.axvline(x=a*1, color='gray', linestyle='--')
plt.axvline(x=a*2, color='gray', linestyle='--')
plt.axvline(x=a*3, color='gray', linestyle='--')
plt.axvline(x=a*4, color='gray', linestyle='--')
plt.axvline(x=a*5, color='gray', linestyle='--')
plt.axvline(x=a*6, color='gray', linestyle='--')
plt.axvline(x=a*7, color='gray', linestyle='--')
plt.axvline(x=a*8, color='gray', linestyle='--')
plt.axvline(x=a*9, color='gray', linestyle='--')
plt.grid(True)

# plt.subplot(4, 1, 3)  # (行数, 列数, 子图编号)
# # plt.plot(time, voltage)
# plt.scatter(time, voltage,s=5)
# plt.ylabel('Voltage (V)')
# plt.axvline(x=a*1, color='gray', linestyle='--')
# plt.axvline(x=a*2, color='gray', linestyle='--')
# plt.axvline(x=a*3, color='gray', linestyle='--')
# plt.axvline(x=a*4, color='gray', linestyle='--')
# plt.axvline(x=a*5, color='gray', linestyle='--')
# plt.axvline(x=a*6, color='gray', linestyle='--')
# plt.axvline(x=a*7, color='gray', linestyle='--')
# plt.axvline(x=a*8, color='gray', linestyle='--')
# plt.axvline(x=a*9, color='gray', linestyle='--')
# plt.grid(True)

# 在第二个子图中绘制 angle 数据和叠加了 sin_wave 的数据
plt.subplot(4, 1, 4)  # (行数, 列数, 子图编号)
# plt.plot(time, angle, label='Angle')
plt.scatter(time, angle, s=10, label='Angle')
plt.plot(time, sin_wave, color=(0, 0, 0, 0.5), linestyle='--', label='Desire Angle')
plt.axvline(x=a*1, color='gray', linestyle='--')
plt.axvline(x=a*2, color='gray', linestyle='--')
plt.axvline(x=a*3, color='gray', linestyle='--')
plt.axvline(x=a*4, color='gray', linestyle='--')
plt.axvline(x=a*5, color='gray', linestyle='--')
plt.axvline(x=a*6, color='gray', linestyle='--')
plt.axvline(x=a*7, color='gray', linestyle='--')
plt.axvline(x=a*8, color='gray', linestyle='--')
plt.axvline(x=a*9, color='gray', linestyle='--')
plt.ylabel('Angle (Deg)')
plt.xlabel('Time (Sec)')
plt.grid(True)
plt.legend()

# plt.subplot(4, 1, 3)  # (行数, 列数, 子图编号)
# plt.plot(time,torque)
# plt.ylabel('Torque (N*m)')
# plt.grid(True)

# plt.subplot(4, 1, 4)  # (行数, 列数, 子图编号)
# plt.plot(time, moving_average_with_padding(force, 25))
# plt.xlabel('Time (seconds)')
# plt.ylabel('Force (N)')
# plt.grid(True)

plt.tight_layout()

# plt.figure(figsize=(10, 8))
# plt.plot(time, force)
# plt.ylabel('Force')
# plt.grid(True)

plt.figure(figsize=(10, 8))
plt.plot(range(1,len(rmses)+1),rmses) 
plt.plot(range(1,len(rmses)+1),rmses_1) 
# plt.plot(range(1,len(rmses)+1),rmses_2) 
plt.grid(True)
plt.xticks(range(1,len(rmses)+1))
plt.ylabel('RMSE (Deg)')
plt.xlabel('Cycle')
# # 调整布局，使子图不重叠
# plt.tight_layout()
plt.ylim(0, 7)
plt.show()

# ## 生成一個sin
# period_length = len(target_wave)
# start_index = len(sin_wave) - 3 * period_length

# # 提取倒数第二个波形段
# penultimate_wave = angle[start_index:start_index + int(period_length)]
# penultimate_wave_v = voltage[start_index:start_index + int(period_length)]
# save_dir = './combine4To10SecSineWave/1kg_angle'
# save_path = os.path.join(save_dir, 'sine_1kg_4.npy')
# save_dir_v = './combine4To10SecSineWave/1kg_voltage'
# save_path_v = os.path.join(save_dir_v, 'sine_1kg_4.npy')
# np.save(save_path, penultimate_wave)
# np.save(save_path_v, penultimate_wave_v)