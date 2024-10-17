import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# 读取两个 NumPy 数组
EXP_DIR = './sineWaveHistory'
EXP_DIR = './exp'
# data_date = '2024_09_20_0927' #1kg
data_date = '2024_09_27_1346' #2kg
# data_date = '2024_09_30_1607' #2kg

voltage_path = f'{EXP_DIR}/{data_date}/1/voltage.npy'
angle_path = f'{EXP_DIR}/{data_date}/1/angle.npy'
force_path = f'{EXP_DIR}/{data_date}/1/force.npy'
torque_path = f'{EXP_DIR}/{data_date}/1/torque.npy'
voltage = np.load(voltage_path)
angle = np.load(angle_path)
force = np.load(force_path)
torque = np.load(torque_path)
voltage = voltage[1 + 25 * 4:]  # 删除前面部分
angle = angle[1 + 25 * 4:]      # 删除前面部分
force = force[1 + 25 * 4:]
torque = torque[1 + 25 * 4:]
# torque = torque *9.81
time = np.linspace(0, len(voltage)/50, len(voltage))

# force 滤波
def moving_average_with_padding(data, window_size):
    pad_size = window_size // 2
    padded_data = np.pad(data, pad_size, mode='reflect')
    convolved_data = np.convolve(padded_data, np.ones(window_size) / window_size, mode='valid')
    return convolved_data

torque = moving_average_with_padding(torque, 25)
torque = torque - np.min(torque)
torque = torque * 1.3
angle = angle - 2  # 对角度进行调整

# 多项式拟合（六次多项式）
degree = 6
coefficients = np.polyfit(voltage, torque, degree)

# 生成拟合曲线
fitted_curve = np.polyval(coefficients, voltage)

# 计算斜率的导数（加速度）
derivative_coefficients = np.polyder(coefficients)  # 一次导数（斜率）
slope_curve = np.polyval(derivative_coefficients, voltage)  # 斜率曲线

second_derivative_coefficients = np.polyder(derivative_coefficients)  # 二次导数（加速度）
acceleration_curve = np.polyval(second_derivative_coefficients, voltage)  # 加速度曲线

# 找到加速度的波峰和波谷
peaks, _ = find_peaks(acceleration_curve)  # 找到波峰
valleys, _ = find_peaks(-acceleration_curve[:len(acceleration_curve)//2], height=-np.max(acceleration_curve) * 0.1)
fitted_curve = fitted_curve[:125]

# 确保至少有一个波峰和一个波谷
if len(peaks) > 0 and len(valleys) > 0:
    # 选择第一个波峰和第一个波谷作为分段点
    first_peak = peaks[0]
    first_valley = valleys[0]
    
    # 选择波谷和波峰作为分段点（确保顺序）
    if first_valley < first_peak:
        threshold1_idx = first_valley
        threshold2_idx = first_peak
    else:
        threshold1_idx = first_peak
        threshold2_idx = first_valley

    # # 检查和调整以确保分段的合理性
    # threshold1_idx = max(0, threshold1_idx)
    # threshold2_idx = min(125 - 1, threshold2_idx)

    # 切分数据
    segment1_voltage = voltage[0:threshold1_idx + 1]
    segment1_torque = fitted_curve[0:threshold1_idx + 1]

    segment2_voltage = voltage[threshold1_idx + 1:threshold2_idx + 1]
    segment2_torque = fitted_curve[threshold1_idx + 1:threshold2_idx + 1]

    segment3_voltage = voltage[threshold2_idx + 1:125]
    segment3_torque = fitted_curve[threshold2_idx + 1:]
    # 对每一段进行线性拟合
    # 第一段拟合
    coeffs1 = np.polyfit(segment1_voltage, segment1_torque, 1)
    line1 = np.polyval(coeffs1, segment1_voltage)

    # 第二段拟合
    coeffs2 = np.polyfit(segment2_voltage, segment2_torque, 1)
    line2 = np.polyval(coeffs2, segment2_voltage)

    # 第三段拟合：强制起点接上第二段末端
    coeffs3 = np.polyfit(segment3_voltage, segment3_torque, 1)
    line3_start = np.polyval(coeffs3, segment2_voltage[-1])  # 第二段的末端点
    line3_slope = coeffs3[0]  # 保持拟合的斜率
    line3_intercept = line3_start - line3_slope * segment3_voltage[0]  # 调整截距
    line3 = line3_slope * segment3_voltage + line3_intercept

    # 创建图形
    plt.figure(figsize=(12, 6))

    # # 第一个子图：原始数据和三段线性拟合
    # plt.subplot(3, 1, 1)
    plt.plot(voltage, torque, 'o', alpha=0.2)
    plt.plot(segment1_voltage,fitted_curve[0:threshold1_idx + 1], 'r')
    plt.plot(segment2_voltage, fitted_curve[threshold1_idx + 1:threshold2_idx + 1], 'g')
    plt.plot(segment3_voltage,fitted_curve[threshold2_idx + 1:], 'b')
    plt.plot(voltage[:125], fitted_curve[:125], 'k--',alpha=0.2)

    plt.xlabel('Voltage (V)')
    plt.ylabel('Torque')
    # plt.title('Three-Segment Linear Fit with Continuity at Boundaries')
    plt.grid(True)
    plt.legend()
    print(fitted_curve[threshold1_idx ])
    print(fitted_curve[threshold1_idx + 1])
    print(fitted_curve[threshold2_idx ])


    plt.tight_layout()

    # 输出每段的线性拟合系数
    print(f'第一段线性拟合系数: 斜率 = {coeffs1[0]:.4f}, 截距 = {coeffs1[1]:.4f}')
    print(f'第二段线性拟合系数: 斜率 = {coeffs2[0]:.4f}, 截距 = {coeffs2[1]:.4f}')
    print(f'第三段线性拟合系数: 斜率 = {line3_slope:.4f}, 截距 = {line3_intercept:.4f}')
else:
    print("未找到足够的波峰和波谷进行分段。")

# 找到所有波峰
peaks, _ = find_peaks(acceleration_curve)

# 找到所有波谷
valleys, _ = find_peaks(-acceleration_curve)

# 创建图形
plt.figure(figsize=(12, 6))

# 绘制加速度曲线
plt.plot(voltage, acceleration_curve, 'b-', label='Acceleration of the Fitted Curve')

# 标记波峰
plt.plot(voltage[peaks], acceleration_curve[peaks], 'ro', label='Peaks')  # 红色圆点表示波峰

# 标记波谷
plt.plot(voltage[valleys], acceleration_curve[valleys], 'go', label='Valleys')  # 绿色圆点表示波谷

# 添加图例和标签
plt.xlabel('Voltage (V)')
plt.ylabel('Acceleration')
plt.title('Acceleration Curve with Peaks and Valleys')
plt.grid(True)
plt.legend()

plt.figure(figsize=(12, 6))
plt.plot(voltage,torque)

plt.xlabel('Voltage (V)')
plt.ylabel('Torque (N*m)')
plt.grid(True)
# # 第二个子图：拟合曲线和斜率
plt.figure(figsize=(12, 6))
lenth0 = np.sqrt(0.20**2+0.255**2-2*0.20*0.255*np.cos(np.radians(180 - 29)))
lenth = np.sqrt(0.20**2+0.255**2-2*0.20*0.255*np.cos(np.radians(180 - angle)))
# plt.plot(voltage,angle)
plt.plot((lenth0 - lenth)/lenth0,torque)
plt.xlabel('Voltage (V)')
plt.ylabel('Angle (Deg)')
plt.grid(True)

plt.figure(figsize=(12, 6))

# # 第一个子图：原始数据和三段线性拟合
plt.subplot(3, 1, 1)
plt.plot(time, voltage)

plt.ylabel('Voltage (V)')
# plt.title('Three-Segment Linear Fit with Continuity at Boundaries')
plt.grid(True)

# # 第二个子图：拟合曲线和斜率
plt.subplot(3, 1, 2)
plt.plot(time, torque)

plt.ylabel('Torque (N*m)')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(time, angle)
plt.xlabel('Time (Sec)')
plt.ylabel('Angle (Deg)')
plt.grid(True)

# plt.subplot(4, 1, 4)
# plt.plot(time, force)
# plt.xlabel('T')
# plt.ylabel('Torque')
# plt.grid(True)

# 显示图形
plt.tight_layout()
# plt.show()

# 按 125 个点为一周期分隔数据，并分类
num_points_per_period = 125
num_periods = len(voltage) // num_points_per_period

voltage_increasing = []
torque_increasing = []
voltage_decreasing = []
torque_decreasing = []

for i in range(num_periods):
    start_idx = i * num_points_per_period
    end_idx = start_idx + num_points_per_period
    period_voltage = voltage[start_idx:end_idx]
    period_torque = torque[start_idx:end_idx]

    # 检查电压是递增还是递减
    if period_voltage[0] < period_voltage[-1]:
        voltage_increasing.append(period_voltage)
        torque_increasing.append(period_torque)
    else:
        voltage_decreasing.append(period_voltage)
        torque_decreasing.append(period_torque)

# 将数据转换为 NumPy 数组
voltage_increasing = np.concatenate(voltage_increasing)
torque_increasing = np.concatenate(torque_increasing)
voltage_decreasing = np.concatenate(voltage_decreasing)
torque_decreasing = np.concatenate(torque_decreasing)

# 对递增和递减的部分进行多项式拟合
degree = 6

# 递增部分拟合
coefficients_in = np.polyfit(voltage_increasing, torque_increasing, degree)
fitted_curve_in = np.polyval(coefficients_in, voltage_increasing)

# 递减部分拟合
coefficients_de = np.polyfit(voltage_decreasing, torque_decreasing, degree)
fitted_curve_de = np.polyval(coefficients_de, voltage_decreasing)

def find_segments(voltage, fitted_curve):
    # 计算加速度
    derivative_coefficients = np.polyder(np.polyfit(voltage, fitted_curve, degree))  # 一次导数
    second_derivative_coefficients = np.polyder(derivative_coefficients)  # 二次导数
    secondsecond_derivative_coefficients = np.polyder(second_derivative_coefficients)  # 二次导数
    acceleration_curve = np.polyval(secondsecond_derivative_coefficients, voltage)

    # 找到加速度的波峰和波谷
    peaks, _ = find_peaks(acceleration_curve)
    valleys, _ = find_peaks(-acceleration_curve)

    if len(peaks) > 0 and len(valleys) > 0:
        # 选择第一个波峰和第一个波谷作为分段点

        first_peak = peaks[0]
        first_valley = valleys[0]


        # 确定分段点的顺序
        if first_valley < first_peak:
            threshold1_idx = first_valley
            threshold2_idx = first_peak
        else:
            threshold1_idx = first_peak
            threshold2_idx = first_valley

        # # 检查和调整以确保分段的合理性
        # threshold1_idx = max(0, threshold1_idx)
        # threshold2_idx = min(len(voltage) - 1, threshold2_idx)

        # 切分拟合曲线数据
        segment1_voltage = voltage[0:threshold1_idx + 1]
        segment1_torque = fitted_curve[0:threshold1_idx + 1]

        segment2_voltage = voltage[threshold1_idx + 1:threshold2_idx + 1]
        segment2_torque = fitted_curve[threshold1_idx + 1:threshold2_idx + 1]

        segment3_voltage = voltage[threshold2_idx + 1:125]
        segment3_torque = fitted_curve[threshold2_idx + 1:125]

        # segment1_coefficients = np.polyfit(voltage[0:threshold1_idx + 1],fitted_curve[0:threshold1_idx + 1], 1)
        # segment1_torque = np.polyval(segment1_coefficients, segment1_voltage)
        # segment2_coefficients = np.polyfit(voltage[threshold1_idx + 1:threshold2_idx + 1],fitted_curve[threshold1_idx + 1:threshold2_idx + 1], 1)
        # segment2_torque = np.polyval(segment2_coefficients, segment2_voltage)
        # segment3_coefficients = np.polyfit(voltage[threshold2_idx + 1:125],fitted_curve[threshold2_idx + 1:125], 1)
        # segment3_torque = np.polyval(segment3_coefficients, segment3_voltage)

        return [segment1_voltage, segment1_torque, segment2_voltage, segment2_torque, segment3_voltage, segment3_torque, acceleration_curve]
    else:
        return None

# 找到递增部分的分段
segments_increasing = find_segments(voltage_increasing, fitted_curve_in)


# 找到递减部分的分段
segments_decreasing = find_segments(voltage_decreasing[::-1], fitted_curve_de[::-1])

# 创建图形
plt.figure(figsize=(12, 6))

# 绘制递增和递减的电压-力矩关系
plt.plot(voltage_increasing, torque_increasing, 'o', color='black', alpha=0.1)
plt.plot(voltage_decreasing, torque_decreasing, 'o', color='black', alpha=0.1)


plt.plot(segments_increasing[0],segments_increasing[1], 'r')
plt.plot(segments_increasing[2], segments_increasing[3], 'g')
plt.plot(segments_increasing[4],segments_increasing[5], 'b')

plt.plot(segments_decreasing[0],segments_decreasing[1], 'r')
plt.plot(segments_decreasing[2], segments_decreasing[3], 'g')
plt.plot(segments_decreasing[4],segments_decreasing[5], 'b')
# plt.plot(voltage[:125], fitted_curve[:125], 'k--',alpha=0.2)

plt.xlabel('Voltage (V)')
plt.ylabel('Torque (N * m)')
plt.grid(True)


plt.tight_layout()
# 找到递增部分的分段和加速度

acceleration_increasing = segments_increasing[6] if segments_increasing else None

# 找到递减部分的分段和加速度

acceleration_decreasing = segments_decreasing[6] if segments_decreasing else None
# 创建另一张图：绘制上升和下降的加速度曲线
plt.figure(figsize=(12, 6))
if acceleration_increasing is not None:
    plt.plot(voltage_increasing, acceleration_increasing, 'b-', label='Acceleration (Increasing)')
if acceleration_decreasing is not None:
    plt.plot(voltage_decreasing[::-1], acceleration_decreasing, 'r-', label='Acceleration (Decreasing)')
plt.xlabel('Voltage (V)')
plt.ylabel('Acceleration')
plt.title('Acceleration Curves (Increasing and Decreasing)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
