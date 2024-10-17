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

    def get_expansion(self,angle):
        l_0 = np.sqrt(0.20**2+0.255**2-2*0.20*0.255*np.cos(np.radians(180 - 29)))
        lenth = np.sqrt(0.20**2+0.255**2-2*0.20*0.255*np.cos(np.radians(180 - angle)))
        h = (l_0 - lenth)/l_0
        return h

class forceGauge:
    def __init__(self, port, baudrate=2400, bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE):
        """
        初始化串口通信类
        :param port: 串口端口号
        :param baudrate: 波特率
        :param bytesize: 数据位大小
        :param parity: 校验位
        :param stopbits: 停止位
        :param timeout: 读取超时
        """
        self.ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            bytesize=bytesize,
            parity=parity,
            stopbits=stopbits,
        )
        if self.ser.is_open:
            print(f"Connected to {self.ser.name}")

    def read_data(self, size=6):
        """
        从串口读取数据
        :param size: 读取数据的字节数
        :return: 读取到的数据
        """
        try:
            self.ser.reset_input_buffer()
            data = self.ser.read(size)
            # 将字节串解码为字符串
            str_data = data.decode('ascii').strip()

            # 将字符串转换为浮点数
            float_data = float(str_data)
            return float_data
        
        except serial.SerialException as e:
            print(f"Error reading data: {e}")
            return None
        
    def get_force(self, size=6):
        try:
            self.ser.reset_input_buffer()
            data = self.ser.read(size)
            # 将字节串解码为字符串
            str_data = data.decode('ascii').strip()

            # 将字符串转换为浮点数
            float_data = float(str_data)
            return float_data*0.13
        
        except serial.SerialException as e:
            print(f"Error reading data: {e}")
            return None

    def get_torque(self,angle,force):
        lenth = np.sqrt(0.20**2+0.255**2-2*0.20*0.255*np.cos(np.radians(180 - angle)))
        angle_of_force_radians = np.arcsin(0.20*np.sin(np.radians(180 - angle))/lenth)
        torque = force * np.sin(angle_of_force_radians)*0.255
        return torque

    def close(self):
        """
        关闭串口
        """
        if self.ser.is_open:
            self.ser.close()
            print("Serial port closed.")


## force 濾波
def moving_average_with_padding(data, window_size = 25):
    # 计算填充大小
    pad_size = window_size // 2
    
    # 使用反射填充模式在数据前后进行填充
    padded_data = np.pad(data, pad_size, mode='reflect')
    
    # 对填充后的数据进行卷积
    convolved_data = np.convolve(padded_data, np.ones(window_size) / window_size, mode='valid')
    
    # 移除前后填充的部分，保证输出与输入长度相同
    return convolved_data

def rescale_array(array, new_min, new_max):
    """
    将一个数组的值线性转换到指定的新范围内。

    参数:
    array (np.ndarray): 原始数组
    new_min (float): 新范围的最小值
    new_max (float): 新范围的最大值

    返回:
    np.ndarray: 转换后的新数组
    """
    # 计算原始数组的最小值和最大值
    min_original = array.min()
    max_original = array.max()

    # 线性转换公式
    new_array = ((array - min_original) / (max_original - min_original)) * (new_max - new_min) + new_min

    return new_array

def calculate_rmse(predictions, targets):
    """
    计算均方根误差 (RMSE)
    
    参数:
    predictions (array-like): 预测值
    targets (array-like): 真实值
    
    返回:
    float: 计算得到的RMSE值
    """
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # 计算均方根误差
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    return rmse