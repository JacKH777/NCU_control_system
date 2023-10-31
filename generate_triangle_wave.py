import numpy as np

# 設定三角波形的參數
amplitude = 32  # 振幅
offset = 25     # 位移，使得波形的最小值為0
frequency = 1      # 頻率（週期數）
num_points = 100    # 點的數量

# 產生0到2π之間的等間隔數值（64個點）
x = np.linspace(0, 2 * np.pi, num_points)

# 使用NumPy函式計算三角波形，並加上位移以確保波形的最小值為0
triangle_wave = amplitude * (2 * np.arccos(np.cos(frequency * x)) / np.pi) + offset

# 四捨五入，確保波形的最大值為40000
triangle_wave = np.round(triangle_wave)

# 限制波形的最大值為40000
triangle_wave = np.clip(triangle_wave, 0, 40000)

# 將三角波形陣列元素轉換為正整數
triangle_wave_int = triangle_wave.astype(int)

# 將三角波形陣列元素以逗號區隔，並以字串形式表示
triangle_wave_str = ','.join(map(str, triangle_wave_int))

# 顯示以逗號區隔的正整數三角波形字串
print(triangle_wave_str)