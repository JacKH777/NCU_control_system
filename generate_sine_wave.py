import numpy as np

def generate_sine_wave(amplitude=0.82, num_points=250):
    """
    生成调整后的正弦波形。
    
    参数:
    amplitude (float): 正弦波的振幅，决定波峰和波谷的高度。
    num_points (int): 生成的数据点数。
    
    返回:
    numpy.ndarray: 调整后的正弦波形数组。
    """
    offset = amplitude + 0.06  # 根据振幅自动计算偏移，确保最低点始终为0.05
    frequency = 1  # 频率设置为1周期

    # 生成0到2π之间的等间隔数值（num_points个点）
    x = np.linspace(0, 2 * np.pi, num_points)

    # 调整正弦波的相位，使得波形从最小值开始（通过减去π/2）
    sin_wave = amplitude * np.sin(frequency * x - np.pi / 2) + offset

    # 四舍五入，保留小数点后4位
    sin_wave_rounded = np.round(sin_wave, 4)

    return sin_wave_rounded

# # 调用函数并打印结果
# result = generate_sine_wave()
# print(result)