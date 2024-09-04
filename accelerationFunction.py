import numpy as np
import matplotlib.pyplot as plt

def calculate_velocity(angle_window, sampling_rate = 25):
    """计算长度为 25 的角度数组的平均速度"""
    if len(angle_window) < 2:
        return 0
    delta_angle = angle_window[-1] - angle_window[0]
    velocity = delta_angle / ((len(angle_window) - 1) / sampling_rate)
    return velocity

def calculate_acceleration(velocity_window, sampling_rate = 25):
    """计算长度为 25 的速度数组的平均加速度"""
    delta_velocity = velocity_window[-1] - velocity_window[0]
    acceleration = delta_velocity / ((len(velocity_window) - 1) / sampling_rate)
    return acceleration

def compute_derivatives(angle_segment, sampling_rate = 25):
    if len(angle_segment) < 2:
        raise ValueError("The input data segment must contain at least two data points.")

    # 计算时间间隔（时间增量）
    time_increment = 1 / sampling_rate

    # 计算角速度
    delta_angle = angle_segment[-1] - angle_segment[0]
    delta_time = (len(angle_segment) - 1) * time_increment
    velocity = delta_angle / delta_time

    # 如果数据段足够长，计算角加速度
    acceleration = None
    if len(angle_segment) > 2:
        angle_segment_mid = angle_segment[len(angle_segment) // 2]
        velocity_mid = (angle_segment_mid - angle_segment[0]) / ((len(angle_segment) // 2) * time_increment)
        delta_velocity = velocity - velocity_mid
        acceleration = delta_velocity / delta_time

    return velocity, acceleration

if __name__ == '__main__':
    # Load data
    EXP_DIR = './sineWaveHistory'
    data_date = '2024_08_22_1716'
    angle_path = f'{EXP_DIR}/{data_date}/1/angle.npy'
    angle = np.load(angle_path)
    angle = angle[1 + 25 * 4:]  # Skip the initial samples if necessary

    sampling_rate = 25  # Hz
    window_size = 15

    # Initialize lists for storing computed values
    velocities = []
    accelerations = []

    # Compute time array for plots
    time = np.linspace(0, len(angle) / sampling_rate, num=len(angle))

    # Sliding window calculation
    for i in range(window_size, len(angle)):
        delta_angle = angle[i] - angle[i - window_size]
        delta_time = window_size / sampling_rate
        velocity = delta_angle / delta_time
        velocities.append(velocity)

        if i >= 2 * window_size:
            delta_velocity = velocities[-1] - velocities[-window_size]
            acceleration = delta_velocity / delta_time
            accelerations.append(acceleration)

    # Prepare time arrays for plotting velocity and acceleration
    time_velocity = time[window_size:len(velocities) + window_size]
    time_acceleration = time[2 * window_size:len(accelerations) + 2 * window_size]

    # Plotting
    plt.figure(figsize=(12, 9))
    plt.subplot(3, 1, 1)
    plt.plot(time, angle, label='Angle')
    plt.ylabel('Angle (degrees)')

    plt.subplot(3, 1, 2)
    plt.plot(time_velocity, velocities, label='Velocity')
    plt.ylabel('Velocity (degrees/sec)')

    plt.subplot(3, 1, 3)
    plt.plot(time_acceleration, accelerations, label='Acceleration')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Acceleration (degrees/sec²)')

    plt.tight_layout()
    plt.show()