# 使用Savitzky-Golay滤波器平滑化三角波，确保底部也平滑化
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

# 使用Savitzky-Golay滤波器平滑化三角波，确保底部也平滑化
amplitude = 38  # 振幅
offset = 23    # 位移
frequency = 3      # 频率
num_points = 600    # 点的数量

# 生成x值
x = np.linspace(0, 2 * np.pi, num_points)

# 生成三角波形
triangle_wave = amplitude * (2 * np.arccos(np.cos(frequency * x)) / np.pi) + offset


window_size = 30

# 创建一个均匀的移动平均窗口
window = np.ones(window_size) / window_size

# 使用 convolve 函数进行移动平均
triangle_wave_smoothed = np.convolve(triangle_wave, window, 'same')
triangle_wave_smoothed = triangle_wave_smoothed[200:400]

# 四舍五入，确保波形的最大值为40000
triangle_wave_smoothed_rounded = np.round(triangle_wave_smoothed)

# 限制波形的最大值为40000
triangle_wave_smoothed_clipped = np.clip(triangle_wave_smoothed_rounded, 0, 40000)

# 将三角波形数组元素转换为正整数
triangle_wave_smoothed_int = triangle_wave_smoothed_clipped.astype(int)

# 将三角波形数组元素以逗号区隔，并以字符串形式表示
triangle_wave_smoothed_str = ','.join(map(str, triangle_wave_smoothed_int))

# 可视化原始与平滑化的三角波形
plt.figure(figsize=(14, 8))
plt.plot(np.arange(0,100,0.5), triangle_wave[200:400], label='Original Triangle Wave', color='blue')
plt.plot(np.arange(0,100,0.5), triangle_wave_smoothed, label='Moving Average Smoothed Triangle Wave', color='red', linestyle='--')
plt.title('Rehabilitation Trajectory')
plt.legend()
plt.xlabel('Percentage of Rehabilitation Action Cycle(%)')
plt.ylabel('Angle(deg)')
plt.show()

# 显示以逗号区隔的正整数三角波形字符串
print(triangle_wave_smoothed_str)