
import time, datetime, os, shutil

import serial
import time
import numpy as np
from scipy import interpolate

from kerasFuzzy import ANFIS,go_to_desire_angle
from datetime import datetime

from encoder_function import encoder,moving_average_with_padding

# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '-1'
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 角度計
angle_encoder = encoder()
angle_encoder.get_com_port('COM7')

# stm32
stm32 = serial.Serial('COM6', 115200)

# force_gauge = forceGauge("COM16")
print(f"Successfull Open COM7 and COM9")
sine_angle = np.asarray([30,30,30,30,30,30,30,30,30,30,30,30,30,30,31,31,31,31,31,31,32,32,32,32,33,33,33,34,34,35,35,35,36,36,37,37,38,38,39,39,40,41,41,42,42,43,44,44,45,45,46,47,48,48,49,50,50,51,52,53,53,54,55,55,56,57,58,58,59,60,60,61,62,62,63,64,65,65,66,67,67,68,69,70,70,71,72,72,73,74,75,75,76,76,77,78,78,79,79,80,81,81,82,82,83,83,84,84,85,85,85,86,86,87,87,87,88,88,88,88,89,89,89,89,89,89,90,90,90,90,90,90,90,90,90,90,90,90,90,90,89,89,89,89,89,88,88,88,88,87,87,87,86,86,86,85,85,84,84,83,83,82,82,81,81,80,80,79,79,78,77,77,76,75,75,74,74,73,72,71,71,70,69,69,68,67,66,66,65,64,63,63,62,61,60,60,59,58,57,57,56,55,54,54,53,52,51,51,50,49,49,48,47,46,46,45,45,44,43,43,42,41,41,40,40,39,39,38,38,37,37,36,36,35,35,34,34,34,33,33,33,32,32,32,32,31,31,31,31,31])
x = np.linspace(0, 1, 250)
f = interpolate.interp1d(x, sine_angle)
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
desire_angle = sine_angle[0]
actual_angle = angle
# angle_1sec = np.full(50, 25)

# 控制器初始化
ku = 0.008
controller_u = 0.5
controller_u_output = 0
error = 0
error_dot = 0
last_error = 0

# 儲存
ts = time.time()
data_time = datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H%M')
fileDir = './exp/{}'.format(data_time)

# FNN 初始化
up = ANFIS()
down = ANFIS()
fileDir_up = os.path.join('./exp/','2024_10_21_1919', '1', 'up_model.npy')
fileDir_down = os.path.join('./exp/','2024_10_21_1919', '1', 'down_model.npy')
# up.load_model(fileDir_up,cycle=13.5)
# down.load_model(fileDir_down,cycle=13.5)
# up.change_learning_rate = 0
# down.change_learning_rate = 0

# 创建文件夹
if not os.path.isdir(fileDir):
    os.makedirs(os.path.join(fileDir, '1'))
else:
    shutil.rmtree(fileDir)
    os.makedirs(os.path.join(fileDir, '1'))

voltage_his = np.asarray([])
angle_his = np.asarray([])
up_model_his = np.asarray([])
down_model_his = np.asarray([])
force_his = np.asarray([])
delta_his = np.asarray([])

# 週期(sec)
target_period = period_10
period = len(target_period)/25
cycle = 0
total_cycle = (30 / period) + 1
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
        if cycle != 0:
            up.load_model(fileDir_up,cycle=25)
            down.load_model(fileDir_down,cycle=25)
            up.change_learning_rate = 0
            down.change_learning_rate = 0
    

    # 每次重製
    if cycle != 0 and idx == len(target_period):
        idx = 0
        # _,controller_u = go_to_desire_angle(angle_encoder, stm32, 30, controller_u)
        print(cycle,total_cycle)
        cycle += 1

    
    # 目標路徑
    if cycle == 0:
        desire_angle = target_period[0]
    else:
        desire_angle = target_period[idx]

        # 獲取角度  
    actual_angle = angle_encoder.get_angle()
    # print(desire_angle,cycle)
    # # 跟過去一秒做平滑
    # angle_1sec = np.roll(angle_1sec, -1)
    # angle_1sec[-1] = actual_angle
    # # angle_1sec_filter = moving_average.update(angle_1sec)
    # # angle_1sec_filter = moving_average_filter(angle_1sec)
    # angle_1sec_filter = gaussian_weighted_moving_average(angle_1sec, window_size=50, sigma=10)
    # angle = angle_1sec_filter[-1]
    angle = actual_angle
    
    # # FNN 學習
    # if cycle > 1:
    #     if idx < len(target_period) / 2:
    #         up.train([error],[error_dot], [desire_angle],[angle])
    #     elif idx >= len(target_period) / 2:
    #         down.train([error],[error_dot], [desire_angle],[angle])
    
    # # 獲取角度  
    # actual_angle = angle.get_angle()

    # # 跟過去一秒做平滑
    # angle_1sec = np.roll(angle_1sec, -1)
    # angle_1sec[-1] = actual_angle
    # angle_1sec_filter = moving_average_filter(angle_1sec)
    # angle = angle_1sec_filter[-1]
    angle_his = np.append(angle_his, angle)
    
    # 計算誤差
    if(len(angle_his)>5):
        angle = moving_average_with_padding(angle_his[-5:])[-1]

    error = desire_angle - angle

    error_dot = (error -  last_error)
    last_error = error # 更新過去誤差
    delta_his = np.append(delta_his,error_dot)
    if(len(delta_his)>25):
        error_dot = moving_average_with_padding(delta_his[-25:])[-1]
    # # FNN 學習
    # if cycle > 0:
    #     if idx < len(target_period) / 2:
    #         up.train([error],[error_dot], [desire_angle],[angle])
    #     elif idx >= len(target_period) / 2:
    #         down.train([error],[error_dot], [desire_angle],[angle])
    
    # FNN 控制
    if idx < len(target_period) / 2:
        delta_u= up.predict([error],[error_dot])
    else:
        delta_u= down.predict([error],[error_dot]) 

    delta_u = delta_u * ku
    controller_u = controller_u + delta_u

    controller_u = float(controller_u)
    controller_u_output = controller_u/10*65535
    controller_u_output = int(controller_u_output)
    if controller_u_output < 0:
        controller_u_output = 0
    stm32.write(controller_u_output.to_bytes(2, byteorder='big'))
    time.sleep(0.02)
    # force = force_gauge.read_data()
    # print(force)
    # 安全限制
    # if actual_angle > 100 or force > 400:
    #     # controller_u_output = 1
    #     # stm32.write(controller_u_output.to_bytes(2, byteorder='big'))
    #     _,_ = go_to_desire_angle(angle_encoder, stm32, 30, controller_u)
    #     break


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
    up_model_his = np.append(up_model_his, up.return_model())
    down_model_his = np.append(down_model_his, down.return_model())
    # force_his = np.append(force_his, force)

    # elapsed_time = time.time() - start_time
    # print(elapsed_time)
    # # 如果运行时间小于指定的总时间，则补充剩余的时间
    # if elapsed_time < 0.05:
    #     remaining_time = 0.05 - elapsed_time
    #     time.sleep(remaining_time)
save_path_voltage = os.path.join(fileDir, '1', 'voltage.npy')
save_path_angle = os.path.join(fileDir, '1', 'angle.npy')
save_path_up_model = os.path.join(fileDir, '1', 'up_model.npy')
save_path_down_model = os.path.join(fileDir, '1', 'down_model.npy')
save_path_force = os.path.join(fileDir, '1', 'force.npy')
np.save(save_path_voltage, voltage_his)
np.save(save_path_angle, angle_his)
np.save(save_path_up_model, up_model_his)
np.save(save_path_down_model, down_model_his)
np.save(save_path_force, force_his)

_,_ = go_to_desire_angle(angle_encoder, stm32, 30, controller_u)



# import time, datetime, os, shutil

# import serial
# import time
# import numpy as np
# from scipy import interpolate

# from kerasFuzzy import ANFIS,go_to_desire_angle
# from datetime import datetime

# from encoder_function import encoder,moving_average_with_padding

# # os.environ['TF_ENABLE_ONEDNN_OPTS'] = '-1'
# # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# # 角度計
# angle_encoder = encoder()
# angle_encoder.get_com_port('COM7')

# # stm32
# stm32 = serial.Serial('COM6', 115200)

# # force_gauge = forceGauge("COM16")
# print(f"Successfull Open COM7 and COM9")
# sine_angle = np.asarray([30,30,30,30,30,30,30,30,30,30,30,30,30,30,31,31,31,31,31,31,32,32,32,32,33,33,33,34,34,35,35,35,36,36,37,37,38,38,39,39,40,41,41,42,42,43,44,44,45,45,46,47,48,48,49,50,50,51,52,53,53,54,55,55,56,57,58,58,59,60,60,61,62,62,63,64,65,65,66,67,67,68,69,70,70,71,72,72,73,74,75,75,76,76,77,78,78,79,79,80,81,81,82,82,83,83,84,84,85,85,85,86,86,87,87,87,88,88,88,88,89,89,89,89,89,89,90,90,90,90,90,90,90,90,90,90,90,90,90,90,89,89,89,89,89,88,88,88,88,87,87,87,86,86,86,85,85,84,84,83,83,82,82,81,81,80,80,79,79,78,77,77,76,75,75,74,74,73,72,71,71,70,69,69,68,67,66,66,65,64,63,63,62,61,60,60,59,58,57,57,56,55,54,54,53,52,51,51,50,49,49,48,47,46,46,45,45,44,43,43,42,41,41,40,40,39,39,38,38,37,37,36,36,35,35,34,34,34,33,33,33,32,32,32,32,31,31,31,31,31])
# x = np.linspace(0, 1, 250)
# f = interpolate.interp1d(x, sine_angle)
# x_10 = np.linspace(0, 1, 250)
# x_8 = np.linspace(0, 1, 200)
# x_6 = np.linspace(0, 1, 150)
# x_4 = np.linspace(0, 1, 100)
# period_10 = f(x_10)
# period_8 = f(x_8)
# period_6 = f(x_6)
# period_4 = f(x_4)

# # 角度初始化
# angle = angle_encoder.get_angle()
# desire_angle = sine_angle[0]
# actual_angle = angle
# # angle_1sec = np.full(50, 25)

# # 控制器初始化
# ku = 0.008
# controller_u = 0.5
# controller_u_output = 0
# error = 0
# error_dot = 0
# last_error = 0

# # FNN 初始化
# up = ANFIS()
# fileDir_up = os.path.join('./exp/','2024_10_21_2158', '1', 'up_model.npy')

# # up.load_model(fileDir_up,cycle=12)

# # up.change_learning_rate = 0



# # 儲存
# ts = time.time()
# data_time = datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H%M')
# fileDir = './exp/{}'.format(data_time)

# # 创建文件夹
# if not os.path.isdir(fileDir):
#     os.makedirs(os.path.join(fileDir, '1'))
# else:
#     shutil.rmtree(fileDir)
#     os.makedirs(os.path.join(fileDir, '1'))

# voltage_his = np.asarray([])
# angle_his = np.asarray([])
# up_model_his = np.asarray([])
# down_model_his = np.asarray([])
# force_his = np.asarray([])
# delta_his = np.asarray([])

# # 週期(sec)
# target_period = period_10
# period = len(target_period)/25
# cycle = 0
# total_cycle = (30 / period) + 1
# idx = 0

# # _,controller_u = go_to_desire_angle(angle_encoder, stm32, 50, controller_u)
# # time.sleep(5)
# # _,controller_u = go_to_desire_angle(angle_encoder, stm32, 50, controller_u)

# # 執行100秒的週期數，加一開始的
# while cycle < total_cycle:
#     # start_time = time.time()

#     # 4秒後啟動
#     if cycle == 0 and idx == 100:
#         idx = 0
#         cycle += 1
#         if cycle != 0:
#             up.load_model(fileDir_up,cycle=17)
#             up.change_learning_rate(0)
    

#     # 每次重製
#     if cycle != 0 and idx == len(target_period):
#         idx = 0
#         # _,controller_u = go_to_desire_angle(angle_encoder, stm32, 30, controller_u)
#         print(cycle,total_cycle)
#         cycle += 1

    
#     # 目標路徑
#     if cycle == 0:
#         desire_angle = target_period[0]
#     else:
#         desire_angle = target_period[idx]

#         # 獲取角度  
#     actual_angle = angle_encoder.get_angle()
#     # print(desire_angle,cycle) 
#     # # 跟過去一秒做平滑
#     # angle_1sec = np.roll(angle_1sec, -1)
#     # angle_1sec[-1] = actual_angle
#     # # angle_1sec_filter = moving_average.update(angle_1sec)
#     # # angle_1sec_filter = moving_average_filter(angle_1sec)
#     # angle_1sec_filter = gaussian_weighted_moving_average(angle_1sec, window_size=50, sigma=10)
#     # angle = angle_1sec_filter[-1]
#     angle = actual_angle
    
#     # # FNN 學習
#     # if cycle > 1:
#     #     if idx < len(target_period) / 2:
#     #         up.train([error],[error_dot], [desire_angle],[angle])
#     #     elif idx >= len(target_period) / 2:
#     #         down.train([error],[error_dot], [desire_angle],[angle])
    
#     # # 獲取角度  
#     # actual_angle = angle.get_angle()

#     # # 跟過去一秒做平滑
#     # angle_1sec = np.roll(angle_1sec, -1)
#     # angle_1sec[-1] = actual_angle
#     # angle_1sec_filter = moving_average_filter(angle_1sec)
#     # angle = angle_1sec_filter[-1]
#     angle_his = np.append(angle_his, angle)
    
#     # 計算誤差
#     if(len(angle_his)>5):
#         angle = moving_average_with_padding(angle_his[-5:])[-1]

#     error = desire_angle - angle

#     error_dot = (error -  last_error)
#     last_error = error # 更新過去誤差
#     delta_his = np.append(delta_his,error_dot)
#     if(len(delta_his)>25):
#         error_dot = moving_average_with_padding(delta_his[-25:])[-1]
#     # # FNN 學習
#     # if cycle > 0 and idx % 2==1:
 
#     #     up.train([error],[error_dot], [desire_angle],[angle])
 
    
#     # FNN 控制

#     delta_u= up.predict([error],[error_dot])


#     delta_u = delta_u * ku
#     controller_u = controller_u + delta_u

#     controller_u = float(controller_u)
#     controller_u_output = controller_u/10*65535
#     controller_u_output = int(controller_u_output)
#     if controller_u_output < 0:
#         controller_u_output = 0
#     stm32.write(controller_u_output.to_bytes(2, byteorder='big'))
#     time.sleep(0.03)
#     # force = force_gauge.read_data()
#     # print(force)
#     # 安全限制
#     # if actual_angle > 100 or force > 400:
#     #     # controller_u_output = 1
#     #     # stm32.write(controller_u_output.to_bytes(2, byteorder='big'))
#     #     _,_ = go_to_desire_angle(angle_encoder, stm32, 30, controller_u)
#     #     break


#     # # 獲取角度  
#     # actual_angle = angle_encoder.get_angle()

#     # # # 跟過去一秒做平滑
#     # # angle_1sec = np.roll(angle_1sec, -1)
#     # # angle_1sec[-1] = actual_angle
#     # # # angle_1sec_filter = moving_average.update(angle_1sec)
#     # # # angle_1sec_filter = moving_average_filter(angle_1sec)
#     # # angle_1sec_filter = gaussian_weighted_moving_average(angle_1sec, window_size=50, sigma=10)
#     # # angle = angle_1sec_filter[-1]
#     # angle = actual_angle
    
#     # # FNN 學習
#     # if cycle > 1:
#     #     if idx < len(target_period) / 2:
#     #         up.train([error],[error_dot], [desire_angle],[angle])
#     #     elif idx >= len(target_period) / 2:
#     #         down.train([error],[error_dot], [desire_angle],[angle])
    

#     idx += 1
#     # 儲存紀錄
#     voltage_his = np.append(voltage_his, controller_u)
#     up_model_his = np.append(up_model_his, up.return_model())

#     # force_his = np.append(force_his, force)

#     # elapsed_time = time.time() - start_time
#     # print(elapsed_time)
#     # # 如果运行时间小于指定的总时间，则补充剩余的时间
#     # if elapsed_time < 0.05:
#     #     remaining_time = 0.05 - elapsed_time
#     #     time.sleep(remaining_time)
# save_path_voltage = os.path.join(fileDir, '1', 'voltage.npy')
# save_path_angle = os.path.join(fileDir, '1', 'angle.npy')
# save_path_up_model = os.path.join(fileDir, '1', 'up_model.npy')

# save_path_force = os.path.join(fileDir, '1', 'force.npy')
# np.save(save_path_voltage, voltage_his)
# np.save(save_path_angle, angle_his)
# np.save(save_path_up_model, up_model_his)

# np.save(save_path_force, force_his)

# _,_ = go_to_desire_angle(angle_encoder, stm32, 30, controller_u)

