
import time, datetime, os, shutil

import serial
import time
import numpy as np
from scipy import interpolate

from kerasFuzzy import Torque_ANFIS,go_to_desire_angle
from datetime import datetime

from encoder_function import encoder,forceGauge,moving_average_with_padding

# 角度計
angle_encoder = encoder()
angle_encoder.get_com_port('COM7')

# stm32
stm32 = serial.Serial('COM12', 115200)

force_gauge = forceGauge("COM16")
print(f"Successfull Open COM7 and COM9")
triangle_voltage_1kg = np.asarray([0.5,0.5289,0.5578,0.5867,0.6157,0.6446,0.6735,0.7024,0.7313,0.7602,0.7892,0.8181,0.847,0.8759,0.9048,0.9337,0.9627,0.9916,1.0205,1.0494,1.0783,1.1072,1.1361,1.1651,1.194,1.2229,1.2518,1.2807,1.3096,1.3386,1.3675,1.3964,1.4253,1.4542,1.4831,1.512,1.541,1.5699,1.5988,1.6277,1.6566,1.6855,1.7145,1.7434,1.7723,1.8012,1.8301,1.859,1.888,1.9169,1.9458,1.9747,2.0036,2.0325,2.0614,2.0904,2.1193,2.1482,2.1771,2.206,2.2349,2.2639,2.2928,2.3217,2.3506,2.3795,2.4084,2.4373,2.4663,2.4952,2.5241,2.553,2.5819,2.6108,2.6398,2.6687,2.6976,2.7265,2.7554,2.7843,2.8133,2.8422,2.8711,2.9,2.9289,2.9578,2.9867,3.0157,3.0446,3.0735,3.1024,3.1313,3.1602,3.1892,3.2181,3.247,3.2759,3.3048,3.3337,3.3627,3.3916,3.4205,3.4494,3.4783,3.5072,3.5361,3.5651,3.594,3.6229,3.6518,3.6807,3.7096,3.7386,3.7675,3.7964,3.8253,3.8542,3.8831,3.912,3.941,3.9699,3.9988,4.0277,4.0566,4.0855,4.0855,4.0566,4.0277,3.9988,3.9699,3.941,3.912,3.8831,3.8542,3.8253,3.7964,3.7675,3.7386,3.7096,3.6807,3.6518,3.6229,3.594,3.5651,3.5361,3.5072,3.4783,3.4494,3.4205,3.3916,3.3627,3.3337,3.3048,3.2759,3.247,3.2181,3.1892,3.1602,3.1313,3.1024,3.0735,3.0446,3.0157,2.9867,2.9578,2.9289,2.9,2.8711,2.8422,2.8133,2.7843,2.7554,2.7265,2.6976,2.6687,2.6398,2.6108,2.5819,2.553,2.5241,2.4952,2.4663,2.4373,2.4084,2.3795,2.3506,2.3217,2.2928,2.2639,2.2349,2.206,2.1771,2.1482,2.1193,2.0904,2.0614,2.0325,2.0036,1.9747,1.9458,1.9169,1.888,1.859,1.8301,1.8012,1.7723,1.7434,1.7145,1.6855,1.6566,1.6277,1.5988,1.5699,1.541,1.512,1.4831,1.4542,1.4253,1.3964,1.3675,1.3386,1.3096,1.2807,1.2518,1.2229,1.194,1.1651,1.1361,1.1072,1.0783,1.0494,1.0205,0.9916,0.9627,0.9337,0.9048,0.8759,0.847,0.8181,0.7892,0.7602,0.7313,0.7024,0.6735,0.6446,0.6157,0.5867,0.5578,0.5289,0.5 ])
triangle_voltage_2kg = np.asarray([0.5,0.5337,0.5675,0.6012,0.6349,0.6687,0.7024,0.7361,0.7699,0.8036,0.8373,0.8711,0.9048,0.9386,0.9723,1.006,1.0398,1.0735,1.1072,1.141,1.1747,1.2084,1.2422,1.2759,1.3096,1.3434,1.3771,1.4108,1.4446,1.4783,1.512,1.5458,1.5795,1.6133,1.647,1.6807,1.7145,1.7482,1.7819,1.8157,1.8494,1.8831,1.9169,1.9506,1.9843,2.0181,2.0518,2.0855,2.1193,2.153,2.1867,2.2205,2.2542,2.288,2.3217,2.3554,2.3892,2.4229,2.4566,2.4904,2.5241,2.5578,2.5916,2.6253,2.659,2.6928,2.7265,2.7602,2.794,2.8277,2.8614,2.8952,2.9289,2.9627,2.9964,3.0301,3.0639,3.0976,3.1313,3.1651,3.1988,3.2325,3.2663,3.3,3.3337,3.3675,3.4012,3.4349,3.4687,3.5024,3.5361,3.5699,3.6036,3.6373,3.6711,3.7048,3.7386,3.7723,3.806,3.8398,3.8735,3.9072,3.941,3.9747,4.0084,4.0422,4.0759,4.1096,4.1434,4.1771,4.2108,4.2446,4.2783,4.312,4.3458,4.3795,4.4133,4.447,4.4807,4.5145,4.5482,4.5819,4.6157,4.6494,4.6831,4.6831,4.6494,4.6157,4.5819,4.5482,4.5145,4.4807,4.447,4.4133,4.3795,4.3458,4.312,4.2783,4.2446,4.2108,4.1771,4.1434,4.1096,4.0759,4.0422,4.0084,3.9747,3.941,3.9072,3.8735,3.8398,3.806,3.7723,3.7386,3.7048,3.6711,3.6373,3.6036,3.5699,3.5361,3.5024,3.4687,3.4349,3.4012,3.3675,3.3337,3.3,3.2663,3.2325,3.1988,3.1651,3.1313,3.0976,3.0639,3.0301,2.9964,2.9627,2.9289,2.8952,2.8614,2.8277,2.794,2.7602,2.7265,2.6928,2.659,2.6253,2.5916,2.5578,2.5241,2.4904,2.4566,2.4229,2.3892,2.3554,2.3217,2.288,2.2542,2.2205,2.1867,2.153,2.1193,2.0855,2.0518,2.0181,1.9843,1.9506,1.9169,1.8831,1.8494,1.8157,1.7819,1.7482,1.7145,1.6807,1.647,1.6133,1.5795,1.5458,1.512,1.4783,1.4446,1.4108,1.3771,1.3434,1.3096,1.2759,1.2422,1.2084,1.1747,1.141,1.1072,1.0735,1.0398,1.006,0.9723,0.9386,0.9048,0.8711,0.8373,0.8036,0.7699,0.7361,0.7024,0.6687,0.6349,0.6012,0.5675,0.5337,0.5])
x = np.linspace(0, 1, 250)
f = interpolate.interp1d(x,triangle_voltage_1kg )
x_10 = np.linspace(0, 1, 250)
x_8 = np.linspace(0, 1, 200)
x_6 = np.linspace(0, 1, 150)
x_4 = np.linspace(0, 1, 100)
period_10 = f(x_10)
period_8 = f(x_8)
period_6 = f(x_6)
period_4 = f(x_4)

# 角度初始化
angle = angle_encoder.get_angle()
force = force_gauge.get_force()
torque_bias = force_gauge.get_torque(angle,force)

# 儲存
ts = time.time()
data_time = datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H%M')
fileDir = './exp/{}'.format(data_time)

# 创建文件夹
if not os.path.isdir(fileDir):
    os.makedirs(os.path.join(fileDir, '1'))
else:
    shutil.rmtree(fileDir)
    os.makedirs(os.path.join(fileDir, '1'))

voltage_his = np.asarray([])
torque_his = np.asarray([])
force_his = np.asarray([])
angle_his = np.asarray([])


# 週期(sec)
target_period = period_10
period = len(target_period)/25
cycle = 0
total_cycle = (10 / period) + 1

idx = 0

# 執行100秒的週期數，加一開始的
while cycle < total_cycle:

    # start_time = time.time()

    # 4秒後啟動
    if cycle == 0 and idx == 100:
        idx = 0
        cycle += 1
    

    # 每次重製
    if cycle != 0 and idx == len(target_period):
        idx = 0
        print(cycle)
        cycle += 1
    
    # 目標路徑
    if cycle == 0:
        desire_voltage = target_period[0]
    else:
        desire_voltage = target_period[idx]

    # 獲取角度  
    actual_angle = angle_encoder.get_angle()
    force = force_gauge.get_force()
    torque = force_gauge.get_torque(actual_angle,force)
    

    controller_u = desire_voltage

    controller_u = float(controller_u)
    controller_u_output = controller_u/10*65535
    controller_u_output = int(controller_u_output)
    if controller_u_output < 0:
        controller_u_output = 0
    stm32.write(controller_u_output.to_bytes(2, byteorder='big'))

    # 安全限制
    if actual_angle > 100 or force > 110:
        # controller_u_output = 1
        # stm32.write(controller_u_output.to_bytes(2, byteorder='big'))
        _,_ = go_to_desire_angle(angle_encoder, stm32, 30, controller_u)
        break

    idx += 1
    # 儲存紀錄
    torque_his = np.append(torque_his, torque)
    voltage_his = np.append(voltage_his, controller_u)
    force_his = np.append(force_his, force)
    angle_his = np.append(angle_his, actual_angle)


save_path_voltage = os.path.join(fileDir, '1', 'voltage.npy')
save_path_torque = os.path.join(fileDir, '1', 'torque.npy')
save_path_force = os.path.join(fileDir, '1', 'force.npy')
save_path_angle = os.path.join(fileDir, '1', 'angle.npy')

np.save(save_path_voltage, voltage_his)
np.save(save_path_torque, torque_his)
np.save(save_path_force, force_his)
np.save(save_path_angle, angle_his)

_,_ = go_to_desire_angle(angle_encoder, stm32, 30, controller_u)
   