import numpy as np
import matplotlib.pyplot as plt

# 参数设置
sample_rate = 25  # 采样率 25 Hz
t1 = 5  # 前段时间
t2 = 5  # 中段时间
t3 = 5  # 后段时间

# 每段的采样点数
n1 = sample_rate * t1
n2 = sample_rate * t2
n3 = sample_rate * t3

# 梯形波形高度
min_height = 0.6867
max_height = round(1.3 * 9.81 * 0.14, 4)

# 梯形波形生成
front = np.linspace(min_height, max_height, n1)
middle = np.ones(n2) * max_height
back = np.linspace(max_height, min_height, n3)

# 合并波形
trapezoid_wave = np.concatenate([front, middle, back])

# 保留小数点后四位
trapezoid_wave = np.round(trapezoid_wave, 4)

# 将波形转换为逗号间隔的字符串
trapezoid_wave_str = ','.join(map(str, trapezoid_wave))

# 输出逗号间隔的数组字符串
print(trapezoid_wave_str)

# 时间轴
time = np.linspace(0, t1 + t2 + t3, len(trapezoid_wave))

# 绘制波形
plt.figure(figsize=(10, 4))
plt.plot(time, trapezoid_wave)
plt.title('Trapezoidal Waveform')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()