import matplotlib.pyplot as plt
import numpy as np
from encoder_function import decoder
import serial
import time

# #圖2
# # 假设电压和角度随时间变化的数据
# voltage = np.concatenate([np.linspace(0.5, 3.2, 100), np.linspace(3.2, 0.5, 100)])
# ser_1 = serial.Serial("COM6", 115200)
# right_hand = decoder()
# right_hand.get_com_port('COM4')
# angle_his = np.asarray([])
# i = 0
# n = 0

# while n!=6:
#     controller_u_output = int(voltage[i]/10*65535)
#     ser_1.write(controller_u_output.to_bytes(2, byteorder='big'))
#     actual_angle = right_hand.get_angle()
#     angle_his = np.append(angle_his,actual_angle)
#     i = i+1
#     if i == 200:
#         i =0
#         n = n+1
#     time.sleep(0.1)

# voltage = np.concatenate([voltage,voltage,voltage,voltage,voltage])
# # 创建图像和轴
# fig, ax1 = plt.subplots()

# # 绘制电压变化曲线
# color = 'tab:blue'
# ax1.set_xlabel('Voltage (V)')
# ax1.set_ylabel('Angle (Degree)')
# ax1.plot(voltage,angle_his[200:], color=color)

# # 显示图表
# fig.tight_layout()  # 调整布局以防止重叠
# plt.show()

#方波
# 假设电压和角度随时间变化的数据
ser_1 = serial.Serial("COM6", 115200)
right_hand = decoder()
right_hand.get_com_port('COM4')
angle_his = np.asarray([])
voltage = np.asarray([])
other_data = np.asarray([])
x_data = np.linspace(0, 20, 100)
i = 0
first_true = True
first_down = True

while i != 500:
    if (i>99 and i<200)or(i>299 and i<400):
        controller_u_output = 3.2
        controller_u_output_a = int(controller_u_output/10*65535)
        # other_data = np.append(other_data,86)
    else:
        controller_u_output = 0.5
        controller_u_output_a = int(controller_u_output/10*65535)
        # other_data = np.append(other_data,30)
    ser_1.write(controller_u_output_a.to_bytes(2, byteorder='big'))
    actual_angle = right_hand.get_angle()
    voltage = np.append(voltage,controller_u_output)
    angle_his = np.append(angle_his,actual_angle)
    if actual_angle>90 and first_true == True:
        print(i)
        first_true = False
    if actual_angle<32 and first_down == True and i>200:
        print('down',i)
        first_down = False
    i = i+1
    time.sleep(0.05)

# 創建一個畫布和兩個軸
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True) 
x_data = np.linspace(0, 20, 400)
# # 複製第一個軸並共享 x 軸
# ax2 = ax1.twinx()
ax1.set_xlim(0, 20)
# 繪製電壓變化曲線
color1 = 'tab:blue'
# ax1.set_xlabel('Time (Sec)')
ax1.set_ylabel('Voltage (V)')
ax1.plot(x_data,voltage[99:499], color=color1)

# 繪製角度變化曲線
color2 = 'tab:blue'
ax2.set_xlabel('Time (Sec)')
ax2.set_ylabel('Angle (Degree)')
ax2.plot(x_data,angle_his[99:499], color=color2)
# ax2.plot(x_data, other_data[99:499], color='black', linestyle='--',alpha= 0.8)

# 顯示圖表
plt.show()
np.savetxt('angle_response_cube_piz_1.txt', angle_his, delimiter=',')