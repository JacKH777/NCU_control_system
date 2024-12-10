import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy import interpolate

# 读取两个 NumPy 数组
EXP_DIR   = './sineWaveHistory'
EXP_DIR   = './exp'
# data_date = '2024_10_07_1105'
# data_date = '2024_10_11_1016'
data_date = '2024_10_12_1212'
voltage_path = f'{EXP_DIR}/{data_date}/1/voltage.npy'
angle_path = f'{EXP_DIR}/{data_date}/1/angle.npy'
force_path = f'{EXP_DIR}/{data_date}/1/force.npy'
emg_path = f'{EXP_DIR}/{data_date}/1/1.npy'
voltage = np.load(voltage_path)
angle = np.load(angle_path)
force = np.load(force_path)
emg = np.load(emg_path)

voltage = voltage[1+25*4:]  # 删除第一个元素（如果需要）
angle = angle[1+25*4:]      # 删除第一个元素（如果需要）
def apply_bandpass_filter(data, fs, lowcut=10.0, highcut=100, order=6):
    b, a = butter(order, [lowcut / (0.5 * fs), highcut / (0.5 * fs)], btype='band')
    return filtfilt(b, a, data, axis=0)
emg = apply_bandpass_filter(emg,1000)
emg = emg[::25,0]
emg = emg[25*40:25*40+len(angle)]
# force = force[1+25*4:] 
voltage = voltage - 0.5



# 输出数据长度
print("emg : ",len(emg))
print("voltage : ",len(voltage))
print("angle : ",len(angle))
print("force : ",len(force))

# 假设采样率为 20 Hz
sampling_rate = 25  # 20 Hz

# # 根据实际数据生成时间轴
# duration = len(angle) / sampling_rate  # 根据 angle 数据长度计算持续时间
# time = np.linspace(0, duration, num=len(angle))

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
target_period = period_10
for cycle in range(1,5):
    if cycle == 0:
            desire_angle = target_period[0]
    elif cycle == 1:

        prefix = np.full(80, 30)
        target_period_1 =  np.concatenate((prefix, target_period))

    elif cycle == 2:

        prefix = np.full(425, 30)

        target_period_1 =  np.concatenate((target_period_1 , prefix))
    elif cycle == 3:

        prefix = np.full(145, 30)
        target_period_1 =  np.concatenate((target_period_1,prefix, target_period))

    elif cycle == 4:

        prefix = np.full(165, 30)
        target_period_1 =  np.concatenate((target_period_1,prefix, target_period))

duration = len(target_period_1) / sampling_rate  # 根据 angle 数据长度计算持续时间
time = np.linspace(0, duration, num=len(target_period_1))

# angle[0:250] = 30
# angle[250*3:250*4] = 30
# angle[250*5:250*6] = 30
# angle[250*7:250*9] = 30

# voltage[0:250] = 0.58
# voltage[250*3:250*4] = 0.58
# voltage[250*5:250*6] = 0.58
# voltage[250*7:250*9] = 0.58

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

error = target_period_1 - angle 
error = moving_average_with_padding(error, 25)
delta_error = np.diff(error)
delta_error = np.insert(delta_error, 0, 0)  # 在差分误差前面插入一个0
delta_error = moving_average_with_padding(delta_error, 25) *25
# force = moving_average_with_padding(force, 25)
def get_torque(angle,force):
        lenth = np.sqrt(0.20**2+0.255**2-2*0.20*0.255*np.cos(np.radians(180 - angle)))
        angle_of_force_radians = np.arcsin(0.20*np.sin(np.radians(180 - angle))/lenth)
        torque = force * np.sin(angle_of_force_radians)*0.25
        return torque
# force =  force * 0.32
# lenth = np.sqrt(0.20**2+0.255**2-2*0.20*0.255*np.cos(np.radians(180 - angle)))
# angle_of_force_redius = np.arcsin(0.20*np.sin(np.radians(180 - angle))/lenth)
# torque = force * np.sin(angle_of_force_redius)*0.255
# force = force*0.3
# torque = get_torque(angle,force)
# torque = moving_average_with_padding(torque, 25)
# plt.figure(figsize=(10, 8))
# plt.plot(time, target_period_1)
# plt.plot(time, angle)
# plt.show()

# # 创建一个新的图形对象
# plt.figure(figsize=(10, 8))

plt.subplot(6, 1, 1)  # (行数, 列数, 子图编号)
plt.plot(time, emg)
plt.ylim( -15*10**-5, 15*10**-5)  
# plt.axhspan(-3 * 10**-5, 3 * 10**-5, facecolor='yellow', alpha=0.5)
# plt.ylabel('Emg (V)')
plt.grid(True)


# 在第一个子图中绘制 voltage 数组
plt.subplot(6, 1, 2)  # (行数, 列数, 子图编号)
plt.plot(time, error)
# plt.ylabel('Error (Deg)')
plt.grid(True)

# 在第二个子图中绘制 angle 数据和叠加了 sin_wave 的数据
plt.subplot(6, 1, 3)  # (行数, 列数, 子图编号)
plt.plot(time, delta_error)

# plt.ylabel('Delta Error (Deg/s)')
plt.grid(True)

voltage = moving_average_with_padding(voltage, 25)

# 在第一个子图中绘制 voltage 数组
plt.subplot(6, 1, 4)  # (行数, 列数, 子图编号)
d_voltage = np.diff(voltage,  append=voltage[-1])
plt.plot(time, d_voltage)
# plt.ylabel('Delta Voltage (V)')
plt.grid(True)

# 在第一个子图中绘制 voltage 数组
plt.subplot(6, 1, 5)  # (行数, 列数, 子图编号)
plt.plot(time, voltage)
# plt.ylabel('Voltage (V)')
plt.grid(True)

# 在第二个子图中绘制 angle 数据和叠加了 sin_wave 的数据
plt.subplot(6, 1, 6) # (行数, 列数, 子图编号)
plt.plot(time, angle)
plt.plot(time, target_period_1, color=(0, 0, 0, 0.5), linestyle='--')
# plt.xlabel('Time (Sec)')
# plt.ylabel('Angle (Deg)')
# plt.legend()
plt.grid(True)
plt.show()
