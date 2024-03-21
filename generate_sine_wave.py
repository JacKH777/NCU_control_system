import numpy as np

# 設定sin波形的參數
amplitude = 32  # 振幅
offset = 62     # 位移，使得波形的最小值為0
frequency = 1   # 頻率（週期數）
num_points = 200    # 點的數量

# 產生0到2π之間的等間隔數值（num_points個點）
x = np.linspace(0, 2 * np.pi, num_points)

# 使用NumPy函式計算sin波形，並加上位移以確保波形的最小值為0
sin_wave = amplitude * np.sin(frequency * x) + offset

# 四捨五入，確保波形的最大值為40000
sin_wave = np.round(sin_wave)

# 限制波形的最大值為40000
sin_wave = np.clip(sin_wave, 0, 40000)

# 將sin波形陣列元素轉換為正整數
sin_wave_int = sin_wave.astype(int)

# 找出最小值的索引
min_index = np.argmin(sin_wave_int)

# 從最小值開始重新排列陣列
sin_wave_rolled = np.roll(sin_wave_int, -min_index)

# 將sin波形陣列元素以逗號區隔，並以字串形式表示
sin_wave_str = ','.join(map(str, sin_wave_rolled))

print(sin_wave_str)  # 顯示以逗號區隔的正整數sin波形字串