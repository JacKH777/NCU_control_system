# import numpy as np
# m = 1
# moment_of_inertia = (1/12*0.33*(0.25**4 - 0.24**4))+ 0.33*0.14**2 + m*0.14**2
# print("I value : ",moment_of_inertia)
# g = 9.81
# d = 0.14
# # theta = np.arcsin(torque /m/g/d) / np.pi*180
# # print(theta)
# # t =  m *g*d*np.sin(0.48/180*np.pi)
# # print(t)
# initial_torque = m * g * d * np.sin(50/180*np.pi)
# print(initial_torque)
# torque = initial_torque + 0.05
# theta = np.arcsin(torque /m/g/d) / np.pi*180
# print(theta)

import serial
from encoder_function import forceGauge,encoder
import time
angle_encoder = encoder()
angle_encoder.get_com_port('COM7')
while True:
    print(angle_encoder.get_angle())
# ser = forceGauge("COM16")
# for i in range(100):

#     print(ser.read_data())


# ser.close() 

# import math

# def calculate_angles(a, b = 0.19, c = 0.27):
#     # 使用余弦定理计算角度
#     angle = math.acos((b**2 + c**2 - a**2) / (2 * b * c))
 
#     # 将弧度转为角度
#     angle_deg = math.degrees(angle)
    
#     return angle_deg

# def find_a_for_90_degree(b = 0.20, c = 0.27):
#     return math.sqrt(b**2 + c**2)

# # 長度
# a = 0.445
# # 每次減少的量（1.5公分）
# decrease_step = 1.5 / 100  # 1.5 公分轉換為米
# # 循環，逐步減少 a，並打印每次減少後的角度
# a_current = a
# while a_current >=find_a_for_90_degree():
#     angle_current = 180 - calculate_angles(a_current)
#     print(f"a = {a_current:.4f} m 對應的角度: {angle_current:.2f}°")
#     a_current -= decrease_step


# import numpy as np
# from encoder_function import encoder, forceGauge
# import serial
# import time, datetime, os, shutil
# from datetime import datetime
# import matplotlib.pyplot as plt

# # # 角度計
# # angle_encoder = encoder()
# # angle_encoder.get_com_port('COM7')
# # # stm32
# # stm32 = serial.Serial('COM12', 115200)
# # force_gauge = forceGauge("COM16")
# # print(f"Successfull Open")

# # 定义文件夹路径
# folder_path = "./combine4To10SecSineWave/1kg_voltage/"
# file_names = [f for f in os.listdir(folder_path) if f.endswith('.npy')]

# # 加载所有 NumPy 文件
# arrays = [np.load(os.path.join(folder_path, file)) for file in file_names]

# # 将这些数组合并，按行（第一个维度）合并
# combined_array = np.concatenate(arrays, axis=0) 

# combined_array = combined_array[::-1]


# folder_path_a = "./combine4To10SecSineWave/1kg_angle/"
# file_names_a = [f for f in os.listdir(folder_path) if f.endswith('.npy')]

# # 加载所有 NumPy 文件
# arrays_a = [np.load(os.path.join(folder_path_a, file)) for file in file_names_a]

# # 将这些数组合并，按行（第一个维度）合并
# combined_array_a = np.concatenate(arrays_a, axis=0) 

# combined_array_a = combined_array_a[::-1]
# # force_his = np.asarray([])

# # for i in range(len(combined_array)):
# #     controller_u = combined_array[i]
# #     controller_u_output = int(controller_u/10*65535)
# #     stm32.write(controller_u_output.to_bytes(2, byteorder='big'))
# #     # angle = angle_encoder.get_angle()
# #     force = force_gauge.read_data()
# #     force_his = np.append(force_his,force)

# # path = "./combine4To10SecSineWave/force_his/"
# # save_path_force = os.path.join(path,'voltage_1kg_noHuman.npy') 
# # np.save(save_path_force, force_his)
# # 创建一个新的图形窗口并绘制合并后的数据
# plt.figure(figsize=(10, 6))
# plt.subplot(2, 1, 1)
# # 绘制合并后的数组
# plt.plot(combined_array)
# # plt.title('Combined Data')
# plt.xlabel('Index')
# plt.ylabel('Voltage (V)')
# plt.grid(True)
# plt.subplot(2, 1, 2)

# plt.plot(combined_array_a)
# # plt.title('Combined Data')
# plt.xlabel('Index')
# plt.ylabel('angle')
# plt.grid(True)
# # 显示图形
# plt.tight_layout()
# plt.show() 
