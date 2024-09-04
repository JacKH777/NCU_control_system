import numpy as np
import matplotlib.pyplot as plt
from encoder_function import encoder
import serial
from kerasFuzzy import ANFIS,go_to_desire_angle
from accelerationFunction import compute_derivatives


import time, datetime, os, shutil
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# 初始化參數
m = 1
g = 9.81 
d_load = 0.14
moment_of_inertia = (1/12*0.33*(0.25**4 - 0.24**4))+ 0.33*0.14**2 + m*0.14**2
print("I value : {}",moment_of_inertia)

fnn_controller = ANFIS()

# 角度計
angle_encoder = encoder()
angle_encoder.get_com_port('COM7')
# stm32
stm32 = serial.Serial('COM12', 115200)
print(f"Successfull Open COM7 and COM9")

# 控制器初始化
ku = 0.008
controller_u = 0.5
controller_u_output = 0
error = 0
error_dot = 0
last_error = 0
# desire_angle = 30
torque = 0

voltage_his = np.asarray([])
angle_his = np.asarray([])
fnn_controller_model_his = np.asarray([])

idx = 0

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

save_path_voltage = os.path.join(fileDir, '1', 'voltage.npy')
save_path_angle = os.path.join(fileDir, '1', 'angle.npy')
save_path_fnn_controller_model = os.path.join(fileDir, '1', 'fnn_controller_model.npy')
save_path_desire_angle = os.path.join(fileDir, '1', 'desire_angle.npy')
save_path_desire_angle_dot = os.path.join(fileDir, '1', 'desire_angle_dot.npy')
save_path_desire_angle_dotdot = os.path.join(fileDir, '1', 'desire_angle_dotdot.npy')

voltage_his = np.asarray([])
angle_his = np.asarray([])
desire_angle_his = np.asarray([])
fnn_controller_model_his = np.asarray([])

# # 初始化到指定位置
# initial_angle = 50
# if not go_to_desire_angle(fnn_controller, angle_encoder, stm32, initial_angle, controller_u):
#     print("Too Far Or Cant Reach{}",initial_angle)
# time.sleep(3)

cycle = 0

while cycle < 11 :
    error = 0
    error_dot = 0
    last_error = 0
    idx = 0

    # 初始化到指定位置
    initial_angle = 50
    desire_angle = 50

    angle_arr,controller_u = go_to_desire_angle(angle_encoder, stm32, initial_angle, controller_u)
    # desire_angle_his =  np.append(angle_his,np.full(len(angle_arr), initial_angle))
    angle_his =  np.append(angle_his,angle_arr)
    initial_torque = m * g * d_load  *np.sin(initial_angle/180*np.pi)

    desire_torque = initial_torque + 0.15
    desire_angle = np.arcsin(desire_torque/m/g/d_load)/np.pi*180
    # print( desire_angle)
    while idx < 100:
        angle = angle_encoder.get_angle()
        fnn_controller.train([error],[error_dot], [desire_angle],[angle])
        angle_his =  np.append(angle_his,angle)
        # desire_velocity,desire_acceleration = compute_derivatives(desire_angle_his[-1:])
        # total_torque = desire_torque - m * g * d_load  *np.sin(desire_angle/180*np.pi)
        # desire_angle = np.arcsin(total_torque / m / g / d_load) / np.pi*180
        
        # desire_angle_dotdot = total_torque / moment_of_inertia *180/np.pi
        # desire_velocity = desire_velocity + (desire_angle_dotdot*0.04)
        # desire_angle = desire_angle_his[-1] + (desire_velocity*0.04)
        # desire_angle_his = np.append(desire_angle_his,desire_angle)
        # print(total_torque,desire_velocity,desire_angle,desire_velocity*0.04)
        error = desire_angle - angle
        error_dot = (error -  last_error)
        last_error = error # 更新過去誤差

        delta_u= fnn_controller.predict([error],[error_dot]) 
        delta_u = delta_u * ku
        controller_u = controller_u + delta_u

        controller_u = float(controller_u)
        controller_u_output = controller_u/10*65535
        controller_u_output = int(controller_u_output)
        if controller_u > 6:
            controller_u = 0
            break
        stm32.write(controller_u_output.to_bytes(2, byteorder='big'))
        idx +=1

    print(cycle)
    cycle += 1


np.save(save_path_voltage, voltage_his)
np.save(save_path_angle, angle_his)
np.save(save_path_fnn_controller_model, fnn_controller_model_his)

# np.save(save_path_desire_angle, desire_angle_his)
# np.save(save_path_desire_angle_dot, desire_angle_dot_his)
# np.save(save_path_desire_angle_dotdot, desire_angle_dotdot_his)

angle_arr,controller_u = go_to_desire_angle(angle_encoder, stm32, 30, controller_u)
controller_u_output = int(0/10*65535)
stm32.write(controller_u_output.to_bytes(2, byteorder='big'))


