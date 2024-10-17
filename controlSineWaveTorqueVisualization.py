import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy import interpolate
import os
from encoder_function import rescale_array,calculate_rmse
from generate_sine_wave import generate_sine_wave

# 读取两个 NumPy 数组
EXP_DIR   = './sineWaveHistory'
EXP_DIR   = './exp'
data_date = '2024_10_08_2006' # 1kg
data_date_2 = '2024_10_10_1535' # 2kg
data_date = '2024_10_10_1801' # 2kg_1fnn
data_date = '2024_10_11_1349' # 2kg_5rule

voltage_path = f'{EXP_DIR}/{data_date}/1/voltage.npy'
angle_path = f'{EXP_DIR}/{data_date}/1/angle.npy'
force_path = f'{EXP_DIR}/{data_date}/1/force.npy'
torque_path = f'{EXP_DIR}/{data_date}/1/torque.npy'
torque_path_2 = f'{EXP_DIR}/{data_date_2}/1/torque.npy'
h_path = f'{EXP_DIR}/{data_date}/1/h.npy'
voltage = np.load(voltage_path)
angle = np.load(angle_path)
force = np.load(force_path)
torque = np.load(torque_path)
torque_2 = np.load(torque_path_2)
# h = np.load(h_path)
voltage = voltage[1+25*4:]  # 删除第一个元素（如果需要）
angle = angle[1+25*4:]      # 删除第一个元素（如果需要）
force = force[1+25*4:] 
torque = torque[1+25*4:]
torque_2 = torque_2[1+25*4:]
# h = h[1+25*4:]
# angle = rescale_array(angle,30,88)

# 输出数据长度
print("voltage : ",len(voltage))
print("angle : ",len(angle))
print("force : ",len(force))

# 假设采样率为 20 Hz
sampling_rate = 25  # 20 Hz

# 根据实际数据生成时间轴
duration = len(angle) / sampling_rate  # 根据 angle 数据长度计算持续时间
time = np.linspace(0, duration, num=len(angle))

# 生成一个sin_wave，并从第10秒开始循环，前面用30填充
sine_angle = np.asarray([0.005,0.0051,0.0054,0.006,0.0068,0.0078,0.009,0.0104,0.0121,0.014,0.0161,0.0184,0.0209,0.0237,0.0266,0.0298,0.0331,0.0367,0.0405,0.0445,0.0486,0.053,0.0576,0.0623,0.0672,0.0724,0.0777,0.0831,0.0888,0.0946,0.1006,0.1067,0.113,0.1195,0.1261,0.1329,0.1398,0.1468,0.1539,0.1612,0.1687,0.1762,0.1838,0.1916,0.1995,0.2074,0.2155,0.2236,0.2318,0.2401,0.2485,0.257,0.2655,0.274,0.2827,0.2913,0.3,0.3088,0.3175,0.3263,0.3351,0.344,0.3528,0.3616,0.3705,0.3793,0.3881,0.3969,0.4056,0.4143,0.423,0.4317,0.4402,0.4488,0.4573,0.4657,0.474,0.4823,0.4905,0.4986,0.5066,0.5145,0.5223,0.53,0.5376,0.5451,0.5524,0.5597,0.5668,0.5737,0.5805,0.5872,0.5938,0.6001,0.6064,0.6124,0.6183,0.6241,0.6296,0.635,0.6402,0.6452,0.6501,0.6547,0.6592,0.6635,0.6676,0.6714,0.6751,0.6786,0.6818,0.6849,0.6877,0.6904,0.6928,0.695,0.697,0.6988,0.7003,0.7016,0.7027,0.7036,0.7043,0.7047,0.705,0.705,0.7047,0.7043,0.7036,0.7027,0.7016,0.7003,0.6988,0.697,0.695,0.6928,0.6904,0.6877,0.6849,0.6818,0.6786,0.6751,0.6714,0.6676,0.6635,0.6592,0.6547,0.6501,0.6452,0.6402,0.635,0.6296,0.6241,0.6183,0.6124,0.6064,0.6001,0.5938,0.5872,0.5805,0.5737,0.5668,0.5597,0.5524,0.5451,0.5376,0.53,0.5223,0.5145,0.5066,0.4986,0.4905,0.4823,0.474,0.4657,0.4573,0.4488,0.4402,0.4317,0.423,0.4143,0.4056,0.3969,0.3881,0.3793,0.3705,0.3616,0.3528,0.344,0.3351,0.3263,0.3175,0.3088,0.3,0.2913,0.2827,0.274,0.2655,0.257,0.2485,0.2401,0.2318,0.2236,0.2155,0.2074,0.1995,0.1916,0.1838,0.1762,0.1687,0.1612,0.1539,0.1468,0.1398,0.1329,0.1261,0.1195,0.113,0.1067,0.1006,0.0946,0.0888,0.0831,0.0777,0.0724,0.0672,0.0623,0.0576,0.053,0.0486,0.0445,0.0405,0.0367,0.0331,0.0298,0.0266,0.0237,0.0209,0.0184,0.0161,0.014,0.0121,0.0104,0.009,0.0078,0.0068,0.006,0.0054,0.0051,0.005])
sine_angle_torque_2kg = np.asarray([0.02,0.0203,0.0211,0.0225,0.0245,0.027,0.0301,0.0337,0.0379,0.0426,0.0479,0.0537,0.06,0.0669,0.0743,0.0823,0.0908,0.0997,0.1092,0.1192,0.1297,0.1407,0.1522,0.1641,0.1765,0.1894,0.2027,0.2165,0.2307,0.2453,0.2603,0.2758,0.2916,0.3079,0.3245,0.3415,0.3588,0.3765,0.3945,0.4128,0.4315,0.4504,0.4697,0.4892,0.5089,0.5289,0.5492,0.5697,0.5903,0.6112,0.6323,0.6535,0.6749,0.6965,0.7181,0.7399,0.7618,0.7838,0.8058,0.8279,0.8501,0.8722,0.8944,0.9167,0.9388,0.961,0.9831,1.0052,1.0272,1.0492,1.071,1.0927,1.1143,1.1358,1.1571,1.1783,1.1992,1.22,1.2406,1.261,1.2811,1.301,1.3206,1.34,1.3591,1.3779,1.3964,1.4146,1.4324,1.4499,1.4671,1.4839,1.5003,1.5163,1.532,1.5472,1.5621,1.5765,1.5905,1.604,1.6171,1.6298,1.6419,1.6536,1.6649,1.6756,1.6858,1.6956,1.7048,1.7135,1.7217,1.7294,1.7366,1.7432,1.7493,1.7548,1.7598,1.7643,1.7682,1.7715,1.7743,1.7766,1.7782,1.7794,1.7799,1.7799,1.7794,1.7782,1.7766,1.7743,1.7715,1.7682,1.7643,1.7598,1.7548,1.7493,1.7432,1.7366,1.7294,1.7217,1.7135,1.7048,1.6956,1.6858,1.6756,1.6649,1.6536,1.6419,1.6298,1.6171,1.604,1.5905,1.5765,1.5621,1.5472,1.532,1.5163,1.5003,1.4839,1.4671,1.4499,1.4324,1.4146,1.3964,1.3779,1.3591,1.34,1.3206,1.301,1.2811,1.261,1.2406,1.22,1.1992,1.1783,1.1571,1.1358,1.1143,1.0927,1.071,1.0492,1.0272,1.0052,0.9831,0.961,0.9388,0.9167,0.8944,0.8722,0.8501,0.8279,0.8058,0.7838,0.7618,0.7399,0.7181,0.6965,0.6749,0.6535,0.6323,0.6112,0.5903,0.5697,0.5492,0.5289,0.5089,0.4892,0.4697,0.4504,0.4315,0.4128,0.3945,0.3765,0.3588,0.3415,0.3245,0.3079,0.2916,0.2758,0.2603,0.2453,0.2307,0.2165,0.2027,0.1894,0.1765,0.1641,0.1522,0.1407,0.1297,0.1192,0.1092,0.0997,0.0908,0.0823,0.0743,0.0669,0.06,0.0537,0.0479,0.0426,0.0379,0.0337,0.0301,0.027,0.0245,0.0225,0.0211,0.0203,0.02])
wave_1 = generate_sine_wave()
x = np.linspace(0, 1, 250)
f = interpolate.interp1d(x, wave_1)
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

## force 濾波
def moving_average_with_padding(data, window_size):
    # 计算填充大小
    pad_size = window_size // 2
    
    # 使用反射填充模式在数据前后进行填充
    padded_data = np.pad(data, pad_size, mode='reflect')
    
    # 对填充后的数据进行卷积
    convolved_data = np.convolve(padded_data, np.ones(window_size) / window_size, mode='valid')
    
    # 移除前后填充的部分，保证输出与输入长度相同
    return convolved_data

# # force = moving_average_with_padding(force, 25)
# lenth = np.sqrt(0.16**2+0.255**2-2*0.16*0.255*np.cos(np.radians(180 - angle)))
# angle_of_force_redius = np.arcsin(0.16*np.sin(np.radians(180 - angle))/lenth)
# torque = force * np.sin(angle_of_force_redius)*0.255

# torque = moving_average_with_padding(torque, 25)
# angle = moving_average_with_padding(angle, 25)
# voltage = moving_average_with_padding(voltage, 25)
# torque_2 = moving_average_with_padding(torque_2, 25)
# torque = torque*0.6
# sin_wave = sin_wave*0.6

# 创建一个新的图形对象
plt.figure(figsize=(10, 8))

# 在第一个子图中绘制 voltage 数组
plt.subplot(4, 1, 1)  # (行数, 列数, 子图编号)
plt.plot(time,angle)
plt.ylabel('Angle (Deg)')
plt.grid(True)

# 在第二个子图中绘制 angle 数据和叠加了 sin_wave 的数据
plt.subplot(4, 1, 2)  # (行数, 列数, 子图编号)
error = sin_wave - torque
plt.plot(time,error)
plt.ylabel('Delta Torque (N*m)')
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 3)  # (行数, 列数, 子图编号)
plt.plot(time,voltage)
plt.ylabel('Voltage (V)')
plt.grid(True)


plt.subplot(4, 1, 4)  # (行数, 列数, 子图编号)
plt.plot(time, torque, label='Torque')
plt.plot(time, sin_wave, color=(0, 0, 0, 0.5), linestyle='--', label='Desire Torque')
plt.ylabel('Torque (N*m)')
plt.grid(True)
plt.legend(loc='upper left')
# plt.subplot(4, 1, 4)  # (行数, 列数, 子图编号)
# plt.plot(time, moving_average_with_padding(force, 25))
plt.xlabel('Time (Sec)')
# plt.ylabel('Force (N)')
# plt.grid(True)

plt.tight_layout()

plt.figure(figsize=(10, 8))
plt.plot(time, force)
plt.ylabel('Force')
plt.grid(True)


rmse_list = []  # 用于存储每个周期的RMSE值

# 使用 for 循环逐周期计算 RMSE
for i in range(0,len(torque),len(target_wave)):

    torque_period = torque[i:i+len(target_wave)]
    # sin_wave_period = sin_wave[i]
    
    # 计算该周期的 RMSE
    rmse = calculate_rmse(torque_period, target_wave)
    rmse_list.append(rmse)

rmse_list_2 = []
for i in range(0,len(torque_2),len(target_wave)):

    torque_period_2 = torque_2[i:i+len(target_wave)]
    # sin_wave_period = sin_wave[i]
    
    # 计算该周期的 RMSE
    rmse_2 = calculate_rmse(torque_period_2, target_wave)
    rmse_list_2.append(rmse_2)

plt.figure(figsize=(10, 8))
plt.plot(range(1,int(len(torque)/len(target_wave))+1),rmse_list,label = '5Rule-FNN')
# plt.plot(range(1,int(len(torque)/len(target_wave))+1),rmse_list_2,label = '7Rule-FNN')
plt.ylabel('RMSE (N*m)')
plt.xlabel('Cycle')
plt.grid(True)
plt.legend(loc='upper left')
# # 调整布局，使子图不重叠
# plt.tight_layout()
plt.show()

# # ######test
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import butter, filtfilt
# from scipy import interpolate
# import os
# from encoder_function import rescale_array,calculate_rmse
# from generate_sine_wave import generate_sine_wave

# # 读取两个 NumPy 数组
# EXP_DIR   = './sineWaveHistory'
# EXP_DIR   = './exp'
# data_date = '2024_10_08_2006' # 1kg
# data_date_2 = '2024_10_10_1535' # 2kg
# data_date = '2024_10_10_1801' # 2kg_1fnn
# data_date = '2024_10_11_1418' # 2kg_5rule

# voltage_path = f'{EXP_DIR}/{data_date}/1/voltage.npy'
# angle_path = f'{EXP_DIR}/{data_date}/1/angle.npy'
# force_path = f'{EXP_DIR}/{data_date}/1/force.npy'
# torque_path = f'{EXP_DIR}/{data_date}/1/torque.npy'
# torque_path_2 = f'{EXP_DIR}/{data_date_2}/1/torque.npy'
# h_path = f'{EXP_DIR}/{data_date}/1/h.npy'
# voltage = np.load(voltage_path)
# angle = np.load(angle_path)
# force = np.load(force_path)
# torque = np.load(torque_path)
# torque_2 = np.load(torque_path_2)
# # h = np.load(h_path)
# voltage = voltage[1+25*4:]  # 删除第一个元素（如果需要）
# angle = angle[1+25*4:]      # 删除第一个元素（如果需要）
# force = force[1+25*4:] 
# torque = torque[1+25*4:]
# torque_2 = torque_2[1+25*4:]
# # h = h[1+25*4:]
# # angle = rescale_array(angle,30,88)

# # 输出数据长度
# print("voltage : ",len(voltage))
# print("angle : ",len(angle))
# print("force : ",len(force))

# # 假设采样率为 20 Hz
# sampling_rate = 25  # 20 Hz

# # 根据实际数据生成时间轴
# duration = len(angle) / sampling_rate  # 根据 angle 数据长度计算持续时间
# time = np.linspace(0, duration, num=len(angle))

# # 生成一个sin_wave，并从第10秒开始循环，前面用30填充
# sine_angle = np.asarray([0.005,0.0051,0.0054,0.006,0.0068,0.0078,0.009,0.0104,0.0121,0.014,0.0161,0.0184,0.0209,0.0237,0.0266,0.0298,0.0331,0.0367,0.0405,0.0445,0.0486,0.053,0.0576,0.0623,0.0672,0.0724,0.0777,0.0831,0.0888,0.0946,0.1006,0.1067,0.113,0.1195,0.1261,0.1329,0.1398,0.1468,0.1539,0.1612,0.1687,0.1762,0.1838,0.1916,0.1995,0.2074,0.2155,0.2236,0.2318,0.2401,0.2485,0.257,0.2655,0.274,0.2827,0.2913,0.3,0.3088,0.3175,0.3263,0.3351,0.344,0.3528,0.3616,0.3705,0.3793,0.3881,0.3969,0.4056,0.4143,0.423,0.4317,0.4402,0.4488,0.4573,0.4657,0.474,0.4823,0.4905,0.4986,0.5066,0.5145,0.5223,0.53,0.5376,0.5451,0.5524,0.5597,0.5668,0.5737,0.5805,0.5872,0.5938,0.6001,0.6064,0.6124,0.6183,0.6241,0.6296,0.635,0.6402,0.6452,0.6501,0.6547,0.6592,0.6635,0.6676,0.6714,0.6751,0.6786,0.6818,0.6849,0.6877,0.6904,0.6928,0.695,0.697,0.6988,0.7003,0.7016,0.7027,0.7036,0.7043,0.7047,0.705,0.705,0.7047,0.7043,0.7036,0.7027,0.7016,0.7003,0.6988,0.697,0.695,0.6928,0.6904,0.6877,0.6849,0.6818,0.6786,0.6751,0.6714,0.6676,0.6635,0.6592,0.6547,0.6501,0.6452,0.6402,0.635,0.6296,0.6241,0.6183,0.6124,0.6064,0.6001,0.5938,0.5872,0.5805,0.5737,0.5668,0.5597,0.5524,0.5451,0.5376,0.53,0.5223,0.5145,0.5066,0.4986,0.4905,0.4823,0.474,0.4657,0.4573,0.4488,0.4402,0.4317,0.423,0.4143,0.4056,0.3969,0.3881,0.3793,0.3705,0.3616,0.3528,0.344,0.3351,0.3263,0.3175,0.3088,0.3,0.2913,0.2827,0.274,0.2655,0.257,0.2485,0.2401,0.2318,0.2236,0.2155,0.2074,0.1995,0.1916,0.1838,0.1762,0.1687,0.1612,0.1539,0.1468,0.1398,0.1329,0.1261,0.1195,0.113,0.1067,0.1006,0.0946,0.0888,0.0831,0.0777,0.0724,0.0672,0.0623,0.0576,0.053,0.0486,0.0445,0.0405,0.0367,0.0331,0.0298,0.0266,0.0237,0.0209,0.0184,0.0161,0.014,0.0121,0.0104,0.009,0.0078,0.0068,0.006,0.0054,0.0051,0.005])
# sine_angle_torque_2kg = np.asarray([0.02,0.0203,0.0211,0.0225,0.0245,0.027,0.0301,0.0337,0.0379,0.0426,0.0479,0.0537,0.06,0.0669,0.0743,0.0823,0.0908,0.0997,0.1092,0.1192,0.1297,0.1407,0.1522,0.1641,0.1765,0.1894,0.2027,0.2165,0.2307,0.2453,0.2603,0.2758,0.2916,0.3079,0.3245,0.3415,0.3588,0.3765,0.3945,0.4128,0.4315,0.4504,0.4697,0.4892,0.5089,0.5289,0.5492,0.5697,0.5903,0.6112,0.6323,0.6535,0.6749,0.6965,0.7181,0.7399,0.7618,0.7838,0.8058,0.8279,0.8501,0.8722,0.8944,0.9167,0.9388,0.961,0.9831,1.0052,1.0272,1.0492,1.071,1.0927,1.1143,1.1358,1.1571,1.1783,1.1992,1.22,1.2406,1.261,1.2811,1.301,1.3206,1.34,1.3591,1.3779,1.3964,1.4146,1.4324,1.4499,1.4671,1.4839,1.5003,1.5163,1.532,1.5472,1.5621,1.5765,1.5905,1.604,1.6171,1.6298,1.6419,1.6536,1.6649,1.6756,1.6858,1.6956,1.7048,1.7135,1.7217,1.7294,1.7366,1.7432,1.7493,1.7548,1.7598,1.7643,1.7682,1.7715,1.7743,1.7766,1.7782,1.7794,1.7799,1.7799,1.7794,1.7782,1.7766,1.7743,1.7715,1.7682,1.7643,1.7598,1.7548,1.7493,1.7432,1.7366,1.7294,1.7217,1.7135,1.7048,1.6956,1.6858,1.6756,1.6649,1.6536,1.6419,1.6298,1.6171,1.604,1.5905,1.5765,1.5621,1.5472,1.532,1.5163,1.5003,1.4839,1.4671,1.4499,1.4324,1.4146,1.3964,1.3779,1.3591,1.34,1.3206,1.301,1.2811,1.261,1.2406,1.22,1.1992,1.1783,1.1571,1.1358,1.1143,1.0927,1.071,1.0492,1.0272,1.0052,0.9831,0.961,0.9388,0.9167,0.8944,0.8722,0.8501,0.8279,0.8058,0.7838,0.7618,0.7399,0.7181,0.6965,0.6749,0.6535,0.6323,0.6112,0.5903,0.5697,0.5492,0.5289,0.5089,0.4892,0.4697,0.4504,0.4315,0.4128,0.3945,0.3765,0.3588,0.3415,0.3245,0.3079,0.2916,0.2758,0.2603,0.2453,0.2307,0.2165,0.2027,0.1894,0.1765,0.1641,0.1522,0.1407,0.1297,0.1192,0.1092,0.0997,0.0908,0.0823,0.0743,0.0669,0.06,0.0537,0.0479,0.0426,0.0379,0.0337,0.0301,0.027,0.0245,0.0225,0.0211,0.0203,0.02])
# x = np.linspace(0, 1, 250)
# f = interpolate.interp1d(x, sine_angle_torque_2kg)
# x_10 = np.linspace(0, 1, 250)
# x_8 = np.linspace(0, 1, 200)
# x_6 = np.linspace(0, 1, 150)
# x_4 = np.linspace(0, 1, 100)
# period_10 = f(x_10)
# period_8 = f(x_8)
# period_6 = f(x_6)
# period_4 = f(x_4)

# wave_1 = generate_sine_wave()
# wave_2 = generate_sine_wave(amplitude=0.45, num_points=250)
# wave_3 = generate_sine_wave(num_points=180)
# wave_4 = generate_sine_wave(amplitude=0.6, num_points=180)
# period=  np.concatenate((wave_1, wave_2,wave_3,wave_4))
# # 生成与 angle 数据长度相同的 sin_wave
# sin_wave = period


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

# # # force = moving_average_with_padding(force, 25)
# # lenth = np.sqrt(0.16**2+0.255**2-2*0.16*0.255*np.cos(np.radians(180 - angle)))
# # angle_of_force_redius = np.arcsin(0.16*np.sin(np.radians(180 - angle))/lenth)
# # torque = force * np.sin(angle_of_force_redius)*0.255

# torque = moving_average_with_padding(torque, 25)
# angle = moving_average_with_padding(angle, 25)
# voltage = moving_average_with_padding(voltage, 25)
# torque_2 = moving_average_with_padding(torque_2, 25)
# # torque = torque*0.6
# # sin_wave = sin_wave*0.6

# # 创建一个新的图形对象
# plt.figure(figsize=(10, 8))

# # 在第一个子图中绘制 voltage 数组
# plt.subplot(4, 1, 1)  # (行数, 列数, 子图编号)
# plt.plot(time,angle)
# plt.ylabel('Angle (Deg)')
# plt.grid(True)

# # 在第二个子图中绘制 angle 数据和叠加了 sin_wave 的数据
# plt.subplot(4, 1, 2)  # (行数, 列数, 子图编号)
# error = sin_wave - torque
# plt.plot(time,error)
# plt.ylabel('Delta Torque (N*m)')
# plt.grid(True)
# plt.legend()

# plt.subplot(4, 1, 3)  # (行数, 列数, 子图编号)
# plt.plot(time,voltage)
# plt.ylabel('Voltage (V)')
# plt.grid(True)


# plt.subplot(4, 1, 4)  # (行数, 列数, 子图编号)
# plt.plot(time, torque, label='Torque')
# plt.plot(time, sin_wave, color=(0, 0, 0, 0.5), linestyle='--', label='Desire Torque')
# plt.ylabel('Torque (N*m)')
# plt.grid(True)
# plt.legend(loc='upper left')
# # plt.subplot(4, 1, 4)  # (行数, 列数, 子图编号)
# # plt.plot(time, moving_average_with_padding(force, 25))
# plt.xlabel('Time (Sec)')
# # plt.ylabel('Force (N)')
# # plt.grid(True)

# plt.tight_layout()

# # plt.figure(figsize=(10, 8))
# # plt.plot(time, force)
# # plt.ylabel('Force')
# # plt.grid(True)


# # rmse_list = []  # 用于存储每个周期的RMSE值

# # # 使用 for 循环逐周期计算 RMSE
# # for i in range(0,len(torque),len(target_wave)):

# #     torque_period = torque[i:i+len(target_wave)]
# #     # sin_wave_period = sin_wave[i]
    
# #     # 计算该周期的 RMSE
# #     rmse = calculate_rmse(torque_period, target_wave)
# #     rmse_list.append(rmse)

# # rmse_list_2 = []
# # for i in range(0,len(torque_2),len(target_wave)):

# #     torque_period_2 = torque_2[i:i+len(target_wave)]
# #     # sin_wave_period = sin_wave[i]
    
# #     # 计算该周期的 RMSE
# #     rmse_2 = calculate_rmse(torque_period_2, target_wave)
# #     rmse_list_2.append(rmse_2)

# # plt.figure(figsize=(10, 8))
# # plt.plot(range(1,int(len(torque)/len(target_wave))+1),rmse_list,label = '5Rule-FNN')
# # plt.plot(range(1,int(len(torque)/len(target_wave))+1),rmse_list_2,label = '7Rule-FNN')
# # plt.ylabel('RMSE (N*m)')
# # plt.xlabel('Cycle')
# # plt.grid(True)
# # plt.legend(loc='upper left')
# # # # 调整布局，使子图不重叠
# # # plt.tight_layout()
# plt.show()

