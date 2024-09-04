import serial
import numpy as np
from scipy.stats import norm

class encoder():
    def __init__(self):
        self.ser = None
        self.first_count = 0
        self.reset_count = 0

    def get_com_port(self,ser_com):
        self.ser = serial.Serial(ser_com, 115200)

    def get_angle(self):
        self.ser.write(b'\x54')
        read_data = self.ser.read(2)
        received_val = int.from_bytes(read_data, byteorder='little')

        # 将整数转换成二进制，并移除最高两位
        binary_val = bin(received_val)[2:].zfill(16)  # 将整数转换为16位的二进制字符串
        truncated_binary = binary_val[2:]  # 移除最高两位
        actual_angle = int(truncated_binary, 2)

        # 校正
        if self.first_count==0:
            self.reset_count = actual_angle
            self.first_count = 1
        if actual_angle < 8192 and self.reset_count > 8192:
            actual_angle = ((actual_angle-self.reset_count+16383)/16383*360)*1.5+29
        elif actual_angle > 8192 and self.reset_count > 8192:
            actual_angle = ((actual_angle-self.reset_count)/16383*360)*1.5+29
        elif actual_angle < 8192 and self.reset_count < 8192:
            actual_angle = ((actual_angle-self.reset_count)/16383*360)*1.5+29
        else:
            actual_angle = ((actual_angle-self.reset_count-16383)/16383*360)*1.5+29
        # if actual_angle > 350:
        #     actual_angle = 28
        return round(actual_angle, 2)
        # return actual_angle

def moving_average_filter(data, window_size=50):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def gaussian_weighted_moving_average(data, window_size=50, sigma=10):
    # 生成高斯权重
    gauss_weights = norm.pdf(np.arange(0, window_size), loc=0, scale=sigma)
    gauss_weights /= gauss_weights.sum()  # 归一化权重
    
    # 只取高斯分布的左半部分
    gauss_weights = gauss_weights[::-1]  # 反转权重数组，使得最新的数据点权重最大
    
    # 应用加权移动平均
    return np.convolve(data, gauss_weights, mode='valid')