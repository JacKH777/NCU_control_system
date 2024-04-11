import matplotlib.pyplot as plt
import numpy as np
from decoder_function import decoder
import serial
import time

# 假设电压和角度随时间变化的数据
voltage = np.concatenate([np.linspace(0, 3, 100), np.linspace(3, 0, 100)])
ser_1 = serial.Serial("COM6", 115200)
right_hand = decoder()
right_hand.get_com_port('COM4')
angle_his = np.asarray([])
i = 0

while i != 200:
    controller_u_output = int(voltage[i]/10*65535)
    ser_1.write(controller_u_output.to_bytes(2, byteorder='big'))
    actual_angle = right_hand.get_angle()
    angle_his = np.append(angle_his,actual_angle)
    i = i+1
    time.sleep(0.1)

# 创建图像和轴
fig, ax1 = plt.subplots()

# 绘制电压变化曲线
color = 'tab:blue'
ax1.set_xlabel('Voltage (V)')
ax1.set_ylabel('Angle (Degree)')
ax1.plot(voltage,angle_his, color=color)

# 显示图表
fig.tight_layout()  # 调整布局以防止重叠
plt.show()