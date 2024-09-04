import serial
import time
import numpy as np
from encoder_function import encoder
import matplotlib.pyplot as plt
ser_1 = serial.Serial("COM9", 115200)

# 角度計
angle_encoder = encoder()
angle_encoder.get_com_port('COM7')

EXP_DIR = './sineWaveHistory'
data_date = '2024_08_22_1738'
voltage_path = f'{EXP_DIR}/{data_date}/1/voltage.npy'
voltage = np.load(voltage_path)
# voltage = voltage[1:]
voltage = voltage[:1250:25]

for i in range(len(voltage)):

    controller_u = voltage[i]
    controller_u_output = controller_u/10*65535
    controller_u_output = int(controller_u_output)
    ser_1.write(controller_u_output.to_bytes(2, byteorder='big'))
    print(controller_u)
    # i = i+1
    time.sleep(3)

# plt.figure(figsize=(12, 9))
# plt.figure(figsize=(12, 9))
# plt.plot(voltage)
# plt.tight_layout()
# plt.show()