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
import time

ser = serial.Serial(
    port='COM16',        # 替换为实际的串口号，例如 'COM16'
    baudrate=2400,       # 设置波特率
    bytesize=serial.EIGHTBITS,  # 数据位
    parity=serial.PARITY_NONE,  # 校验位
    stopbits=serial.STOPBITS_ONE,  # 1位停止位
    timeout=1            # 读取超时设置
)
while True:

    print(ser.read(6))
    

