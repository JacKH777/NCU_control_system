
import time, datetime, os, shutil

import serial
import time
import numpy as np
from scipy import interpolate

from kerasFuzzy import Torque_ANFIS_1kg,go_to_desire_angle
from datetime import datetime

from encoder_function import encoder,forceGauge,moving_average_with_padding
import tensorflow as tf
# 角度計
angle_encoder = encoder()
angle_encoder.get_com_port('COM7')

# stm32
stm32 = serial.Serial('COM23', 115200)

force_gauge = forceGauge("COM16")
print(f"Successfull Open COM7 and COM9")
sine_angle_torque = np.asarray([0.05,0.0503,0.0512,0.0527,0.0548,0.0576,0.0609,0.0648,0.0693,0.0744,0.0801,0.0864,0.0932,0.1007,0.1087,0.1172,0.1264,0.1361,0.1463,0.1571,0.1684,0.1803,0.1927,0.2056,0.219,0.2328,0.2472,0.2621,0.2774,0.2932,0.3095,0.3261,0.3432,0.3608,0.3787,0.397,0.4158,0.4348,0.4543,0.4741,0.4942,0.5147,0.5354,0.5565,0.5778,0.5994,0.6213,0.6434,0.6657,0.6883,0.711,0.7339,0.757,0.7803,0.8037,0.8272,0.8508,0.8745,0.8983,0.9222,0.9461,0.97,0.994,1.018,1.0419,1.0659,1.0898,1.1136,1.1374,1.161,1.1846,1.2081,1.2314,1.2546,1.2776,1.3004,1.323,1.3455,1.3677,1.3897,1.4114,1.4329,1.4541,1.475,1.4956,1.5159,1.5359,1.5555,1.5748,1.5937,1.6122,1.6303,1.648,1.6654,1.6823,1.6987,1.7147,1.7303,1.7454,1.76,1.7742,1.7878,1.801,1.8136,1.8257,1.8373,1.8484,1.8589,1.8688,1.8783,1.8871,1.8954,1.9031,1.9103,1.9168,1.9228,1.9282,1.933,1.9373,1.9409,1.9439,1.9463,1.9481,1.9493,1.9499,1.9499,1.9493,1.9481,1.9463,1.9439,1.9409,1.9373,1.933,1.9282,1.9228,1.9168,1.9103,1.9031,1.8954,1.8871,1.8783,1.8688,1.8589,1.8484,1.8373,1.8257,1.8136,1.801,1.7878,1.7742,1.76,1.7454,1.7303,1.7147,1.6987,1.6823,1.6654,1.648,1.6303,1.6122,1.5937,1.5748,1.5555,1.5359,1.5159,1.4956,1.475,1.4541,1.4329,1.4114,1.3897,1.3677,1.3455,1.323,1.3004,1.2776,1.2546,1.2314,1.2081,1.1846,1.161,1.1374,1.1136,1.0898,1.0659,1.0419,1.018,0.994,0.97,0.9461,0.9222,0.8983,0.8745,0.8508,0.8272,0.8037,0.7803,0.757,0.7339,0.711,0.6883,0.6657,0.6434,0.6213,0.5994,0.5778,0.5565,0.5354,0.5147,0.4942,0.4741,0.4543,0.4348,0.4158,0.397,0.3787,0.3608,0.3432,0.3261,0.3095,0.2932,0.2774,0.2621,0.2472,0.2328,0.219,0.2056,0.1927,0.1803,0.1684,0.1571,0.1463,0.1361,0.1264,0.1172,0.1087,0.1007,0.0932,0.0864,0.0801,0.0744,0.0693,0.0648,0.0609,0.0576,0.0548,0.0527,0.0512,0.0503,0.05])
x = np.linspace(0, 1, 250)
f = interpolate.interp1d(x, sine_angle_torque)
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
for i in range(20):
    angle = angle_encoder.get_angle()
    force = force_gauge.get_force()
    torque_bias = (torque_bias+force_gauge.get_torque(angle,force))/2
    print("torque_bias:",torque_bias)
desire_angle = sine_angle_torque[0]
actual_angle = angle
# angle_1sec = np.full(50, 25)

# 控制器初始化
ku = 0.003
controller_u = 0.5
controller_u_output = 0
error = 0
error_dot = 0
last_error = 0

# FNN 初始化
up = Torque_ANFIS_1kg()
down = Torque_ANFIS_1kg()
down.mf_torque.assign(np.asarray([0.5, 1.05, 1.6]))

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
up_model_his = np.asarray([])
down_model_his = np.asarray([])
force_his = np.asarray([])
angle_his = np.asarray([])


# 週期(sec)
target_period = period_10
period = len(target_period)/25
cycle = 0
total_cycle = (100 / period) + 1

idx = 0

# _,controller_u = go_to_desire_angle(angle_encoder, stm32, 50, controller_u)
# time.sleep(5)
# _,controller_u = go_to_desire_angle(angle_encoder, stm32, 50, controller_u)

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
        up_model_his = np.append(up_model_his, up.return_model())
        down_model_his = np.append(down_model_his, down.return_model())
        # up_model_his = np.concatenate((up_model_his, up.return_model()), axis=0)
        # down_model_his = np.concatenate((down_model_his, down.return_model()), axis=0)

    
    # 目標路徑
    if cycle == 0:
        desire_torque = target_period[0]
    else:
        desire_torque = target_period[idx]

        # 獲取角度  
    actual_angle = angle_encoder.get_angle()
    force = force_gauge.get_force()
    torque = force_gauge.get_torque(actual_angle,force) - torque_bias
    torque_his = np.append(torque_his, torque)
    if len(torque_his) >=40:
        f_torque = moving_average_with_padding(torque_his[-40:],40)[-1]
        # f_torque_his = moving_average_with_padding(torque_his[-25:])[-2]
    else:
        f_torque = torque
    
    angle = actual_angle
    
    # FNN 學習
    if cycle > 1:
        if idx < len(target_period) / 2:
            up.train([f_torque],[error], [desire_torque],[f_torque])
        elif idx >= len(target_period) / 2:
            down.train([f_torque],[error], [desire_torque],[f_torque])
    
    # 計算誤差
    error = (desire_torque - f_torque)
    # error_dot = (error -  last_error)
    # last_error = error # 更新過去誤差

    # FNN 控制
    if idx < len(target_period) / 2:
        delta_u= up.predict([f_torque],[error])
    else:
        delta_u= down.predict([f_torque],[error]) 

    delta_u = delta_u * ku
    controller_u = controller_u + delta_u

    controller_u = float(controller_u)
    controller_u_output = controller_u/10*65535
    controller_u_output = int(controller_u_output)
    if controller_u_output < 0:
        controller_u_output = 0
    stm32.write(controller_u_output.to_bytes(2, byteorder='big'))
    # print(error,controller_u,f_torque,desire_torque)
    # print(force)
    # 安全限制
    if actual_angle > 100 or force > 110:
        # controller_u_output = 1
        # stm32.write(controller_u_output.to_bytes(2, byteorder='big'))
        _,_ = go_to_desire_angle(angle_encoder, stm32, 30, controller_u)
        break


    # # 獲取角度  
    # actual_angle = angle_encoder.get_angle()

    # # # 跟過去一秒做平滑
    # # angle_1sec = np.roll(angle_1sec, -1)
    # # angle_1sec[-1] = actual_angle
    # # # angle_1sec_filter = moving_average.update(angle_1sec)
    # # # angle_1sec_filter = moving_average_filter(angle_1sec)
    # # angle_1sec_filter = gaussian_weighted_moving_average(angle_1sec, window_size=50, sigma=10)
    # # angle = angle_1sec_filter[-1]
    # angle = actual_angle
    
    # # FNN 學習
    # if cycle > 1:
    #     if idx < len(target_period) / 2:
    #         up.train([error],[error_dot], [desire_angle],[angle])
    #     elif idx >= len(target_period) / 2:
    #         down.train([error],[error_dot], [desire_angle],[angle])
    

    idx += 1

    # 儲存紀錄
    voltage_his = np.append(voltage_his, controller_u)
    # up_model_his = np.append(up_model_his, up.return_model())
    # down_model_his = np.append(down_model_his, down.return_model())
    force_his = np.append(force_his, force)
    angle_his = np.append(angle_his, angle)


save_path_voltage = os.path.join(fileDir, '1', 'voltage.npy')
save_path_torque = os.path.join(fileDir, '1', 'torque.npy')
save_path_up_model = os.path.join(fileDir, '1', 'up_model.npy')
save_path_down_model = os.path.join(fileDir, '1', 'down_model.npy')
save_path_force = os.path.join(fileDir, '1', 'force.npy')
save_path_angle = os.path.join(fileDir, '1', 'angle.npy')

np.save(save_path_voltage, voltage_his)
np.save(save_path_torque, torque_his)
np.save(save_path_up_model, up_model_his)
np.save(save_path_down_model, down_model_his)
np.save(save_path_force, force_his)
np.save(save_path_angle, angle_his)

_,_ = go_to_desire_angle(angle_encoder, stm32, 30, controller_u)
