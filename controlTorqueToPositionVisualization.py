import numpy as np
import matplotlib.pyplot as plt
from accelerationFunction import compute_derivatives

# 加载角度数据
EXP_DIR = './exp'
data_date = '2024_08_30_1104'
angle_path = f'{EXP_DIR}/{data_date}/1/angle.npy'
angle = np.load(angle_path)
angle = angle[12*25:]  # 忽略初始数据的一些点

# 假设采样率为 25 Hz
sampling_rate = 25

# 初始化參數
m = 1
g = 9.81 
d_load = 0.14
moment_of_inertia = (1/12*0.33*(0.25**4 - 0.24**4))+ 0.33*0.14**2 + m*0.14**2

# 每隔四秒切一段数据
segment_length = 4 * sampling_rate  # 四秒的数据长度
num_segments = len(angle) // segment_length  # 计算总段数

segments = []

# 循环处理每个四秒段落的数据
for i in range(num_segments):
    if i % 2 == 0:  # 只选择奇数段：1, 3, 5, 7, 9 ...
        segment_start = i * segment_length
        segment_end = segment_start + segment_length
        segment_angle = angle[segment_start:segment_end]
        # 使用第一秒的第一个值填补前25个数据点
        list_pad = np.full((25,), segment_angle[0])
        segment_angle = np.concatenate([list_pad, segment_angle])
        segments.append(segment_angle)  # 将当前段添加到列表中

# 设置滑动窗口参数
window_size = 15  # 窗口大小设为15个数据点
step_size = 1    # 步长设为1个数据点

num_plots = sum(1 for i in range(num_segments) if i % 2 == 0) * 2

# 第一个图形窗口：每段的角度和加速度
plt.figure(figsize=(15, 20))  # 根据子图数量调整图形尺寸

# 处理每个四秒段
subplot_index = 1  # 子图索引从1开始
for i in range(num_segments):
    if i % 2 == 0:  # 只处理奇数段：1, 3, 5, 7, 9 ...
        segment_angle = segments[i // 2]

        # 时间轴（每段开始于0）
        segment_time = np.linspace(0, len(segment_angle) / sampling_rate, num=len(segment_angle))

        # 创建角度子图
        ax_angle = plt.subplot(num_plots, 2, subplot_index)
        ax_angle.plot(segment_time, segment_angle)
        ax_angle.set_ylabel('Angle')

        # 为加速度创建新的子图
        subplot_index += 1
        ax_acceleration = plt.subplot(num_plots, 2, subplot_index)

        # 使用滑动窗口计算加速度
        accelerations = []
        acc_times = []
        for j in range(0, len(segment_angle) - window_size + 1, step_size):
            window = segment_angle[j:j + window_size]
            _, acceleration = compute_derivatives(window, sampling_rate)
            if acceleration is not None:
                accelerations.append(acceleration)
                acc_times.append(segment_time[j + window_size // 2])  # 使用窗口中间的时间点

        # 绘制加速度子图
        ax_acceleration.plot(acc_times, accelerations, 'r-')
        ax_acceleration.set_ylabel('Acceleration')

        # 更新子图索引，为下一个奇数段准备
        subplot_index += 1

        # 最后一个角度子图显示x轴标签
        if subplot_index > num_plots - 1:
            ax_angle.set_xlabel('Time (s)')
            ax_acceleration.set_xlabel('Time (s)')

plt.tight_layout()  # 调整子图布局
# plt.show()

# 第二个图形窗口：重叠绘制所有段的角度
plt.figure(figsize=(12, 8))

# 时间轴（0到5秒）
common_time = np.linspace(0, (segment_length + 25) / sampling_rate, num=(segment_length + 25))

# 循环绘制所有段
for i, segment_angle in enumerate(segments):
    plt.plot(common_time, segment_angle, label=f'Segment {i+1}', alpha=0.5)  # 使用透明度区分

plt.xlabel('Time (s)')
plt.ylabel('Angle')
plt.title('Segments')
plt.legend()

# 第三个图形窗口：重叠绘制所有段的角加速度
plt.figure(figsize=(12, 8))
total_torque = 1.1
comp = np.concatenate([np.full(25, m * g * d_load  *np.sin(50/180*np.pi)), np.full(100, m * g * d_load *np.sin(50/180*np.pi))+0.15])
common_time = np.linspace(0, (segment_length + 25) / sampling_rate, num=(segment_length + 25))
# 循环计算并绘制所有段的角加速度
for i, segment_angle in enumerate(segments):
    accelerations = []
    acc_times = []
    for j in range(0, len(segment_angle) - window_size + 1, step_size):
        window = segment_angle[j:j + window_size]
        _, acceleration = compute_derivatives(window, sampling_rate)
        if acceleration is not None:
            acceleration = acceleration * moment_of_inertia * np.pi/180
            acceleration = acceleration + m * g * d_load  *np.sin(segment_angle[j + 12]/180*np.pi)
            accelerations.append(acceleration)
            acc_times.append(common_time[j + window_size // 2]+0.15)  # 使用窗口中间的时间点

    plt.plot(acc_times, accelerations, label=f'Experiment {i+1}', alpha=0.5)  # 使用透明度区分
plt.plot(common_time, comp, color='black',linestyle='--',alpha=0.5)
plt.xlabel('Time (s)')
plt.ylabel('PMA Torque (N*s)')
plt.title('PMA Torque')
plt.legend()
plt.show()
