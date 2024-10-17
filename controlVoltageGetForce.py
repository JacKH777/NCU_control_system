import numpy as np
from encoder_function import encoder, forceGauge
import serial
import time, datetime, os, shutil
from datetime import datetime
import matplotlib.pyplot as plt
from kerasFuzzy import  go_to_desire_angle

# 角度計
angle_encoder = encoder()
angle_encoder.get_com_port('COM7')


stm32 = serial.Serial('COM12', 115200)
force_gauge = forceGauge("COM16")
print(f"Successfull Open")
controller_u = 0.5
for i in range(30,100,10):
    desire_angle = i

    _,controller_u = go_to_desire_angle(angle_encoder, stm32, desire_angle, controller_u)
    print(angle_encoder.get_angle())
    print(controller_u)
    time.sleep(5)
# # 定义文件夹路径
# folder_path = "./combine4To10SecSineWave/1kg_voltage/"
# file_names = [f for f in os.listdir(folder_path) if f.endswith('.npy')]

# # 加载所有 NumPy 文件
# arrays = [np.load(os.path.join(folder_path, file)) for file in file_names]

# # 将这些数组合并，按行（第一个维度）合并
# combined_array = np.concatenate(arrays, axis=0) 

# combined_array = combined_array[::-1]
# force_his = np.asarray([])

# for i in range(len(combined_array)):
#     controller_u = combined_array[i]
#     controller_u_output = int(controller_u/10*65535)
#     stm32.write(controller_u_output.to_bytes(2, byteorder='big'))
#     # angle = angle_encoder.get_angle()
#     force = force_gauge.read_data()
#     force_his = np.append(force_his,force)

# path = "./combine4To10SecSineWave/force_his/"
# save_path_force = os.path.join(path,'voltage_1kg_noHuman.npy') 
# np.save(save_path_force, force_his)
# force_gauge.close()

# plt.figure(figsize=(10, 6))
# plt.subplot(2, 1, 1)
# # 绘制合并后的数组
# plt.plot(combined_array)
# # plt.title('Combined Data')
# plt.xlabel('Index')
# plt.ylabel('Voltage (V)')
# plt.grid(True)
# plt.subplot(2, 1, 2)

# # plt.plot(combined_array_a)
# # # plt.title('Combined Data')
# # plt.xlabel('Index')
# # plt.ylabel('angle')
# # plt.grid(True)
# # # 显示图形
# # plt.tight_layout()
# plt.show() 