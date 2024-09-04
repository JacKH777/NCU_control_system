
import time, datetime, os, shutil

import serial
import time
import numpy as np
from scipy import interpolate

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from kerasFuzzy import ANFIS
from datetime import datetime

from encoder_function import encoder,moving_average_filter,gaussian_weighted_moving_average

# 角度計
angle_encoder = encoder()
angle_encoder.get_com_port('COM7')

# stm32
stm32 = serial.Serial('COM9', 115200)
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
ku = 0.005
controller_u = 0.5
controller_u_output = 0
error = 0
error_dot = 0
last_error = 0

# FNN 初始化
up = ANFIS()
down = ANFIS()

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
angle_his = np.asarray([])
up_model_his = np.asarray([])
down_model_his = np.asarray([])

# 週期(sec)
target_period = period_10
period = len(target_period)/25
cycle = 0
total_cycle = (100 / period) + 1

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
        cycle += 1

    
    # 目標路徑
    if cycle == 0:
        desire_angle = target_period[0]
    else:
        desire_angle = target_period[idx]

        # 獲取角度  
    actual_angle = angle_encoder.get_angle()

    # # 跟過去一秒做平滑
    # angle_1sec = np.roll(angle_1sec, -1)
    # angle_1sec[-1] = actual_angle
    # # angle_1sec_filter = moving_average.update(angle_1sec)
    # # angle_1sec_filter = moving_average_filter(angle_1sec)
    # angle_1sec_filter = gaussian_weighted_moving_average(angle_1sec, window_size=50, sigma=10)
    # angle = angle_1sec_filter[-1]
    angle = actual_angle
    
    # FNN 學習
    if cycle > 1:
        if idx < len(target_period) / 2:
            up.train([error],[error_dot], [desire_angle],[angle])
        elif idx >= len(target_period) / 2:
            down.train([error],[error_dot], [desire_angle],[angle])
    
    # # 獲取角度  
    # actual_angle = angle.get_angle()

    # # 跟過去一秒做平滑
    # angle_1sec = np.roll(angle_1sec, -1)
    # angle_1sec[-1] = actual_angle
    # angle_1sec_filter = moving_average_filter(angle_1sec)
    # angle = angle_1sec_filter[-1]

    
    # 計算誤差
    error = desire_angle - angle
    error_dot = (error -  last_error)
    last_error = error # 更新過去誤差

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

    # 提供給 stm32
    # 安全限制
    if actual_angle > 100:
        controller_u_output = 1
        stm32.write(controller_u_output.to_bytes(2, byteorder='big'))
        break
    stm32.write(controller_u_output.to_bytes(2, byteorder='big'))
    
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
    angle_his = np.append(angle_his, angle)
    up_model_his = np.append(up_model_his, up.return_model())
    down_model_his = np.append(down_model_his, down.return_model())

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
np.save(save_path_voltage, voltage_his)
np.save(save_path_angle, angle_his)
np.save(save_path_up_model, up_model_his)
np.save(save_path_down_model, down_model_his)
controller_u_output = int(0/10*65535)
stm32.write(controller_u_output.to_bytes(2, byteorder='big'))
            # # ##儲存第一次結果
            # # if  first_period == True and  test >=
            # #  4/total_duration -1:
            # #     first_period_detail = np.append(first_period_detail,actual_angle)
            # #     if Idx == len(target_trag)-1:
            # #         first_period = False
            # # if first_period == False and first_period_cycle == True:
            # #     if actual_angle <= first_period_detail[-1]:
            # #         first_period_detail = np.append(first_period_detail,actual_angle)
            # #         first_period_detail = first_period_detail[1:]
            # #     else:
            # #         first_period_cycle = False
            # # ##

            # # 儲存結果 need
            # # if test < 4/total_duration:
            # #     actual_angle = desire_angle
            # #     total_error = 0
            # if test > 4/tot(actual_angle)
            #     queue_receive_deg.put(actual_angle)
            #     queue_desire_deg.put(desire_angle)
            #     queue_voltage.tal_duration:
            #     # int_actual_angle = input(controller_u)
            #     # if test < 4/total_duration -1:
            #     #     queue_first_period.put(actual_angle)
            #     # elif  first_period_cycle == True:
            #     #     queue_first_period.put(first_period_detail[-1])
            #     # else:
            #     #     queue_first_period.put(first_period_detail[first_period_idx])
            #     #     first_period_idx = first_period_idx + 1
            #     #     if first_period_idx == len(target_trag):
            #     #         first_period_idx = 0
            # # 轉成 16 bits 電壓值 need
            # controller_u = float(controller_u)
            # controller_u_output = controller_u/10*65535
            # controller_u_output = int(controller_u_output)
            

            # if simulation == False:

            #     # if controller_u_output>45000:
            #     #     controller_u_output = 45000
            #     if controller_u_output < 0:
            #         controller_u_output = 0

            # if actual_angle > 100:
            #     controller_u_output = 0
            #     self.ser_1.write(controller_u_output.to_bytes(2, byteorder='big'))
            #     break
            
            # self.ser_1.write(controller_u_output.to_bytes(2, byteorder='big'))
            # voltage_his = np.append(controller_u,voltage_his)
            # angle_his = np.append(actual_angle,angle_his)

            # if Idx == len(target_trag)-1:
            #     total_error = (total_error/200)**0.5
            #     print("total_error : ",total_error)
            #     print("cycle_max_error : ",cycle_max_error)
            #     first_half.save_model('sin_10_model_first_3kg')
            #     second_half.save_model('sin_10_model_second_3kg')
            #     # if total_error <= last_total_error:
            #     #     last_mu_error, last_sigma_error,last_mu_delta, last_sigma_delta, last_y = mu_error, sigma_error, mu_delta, sigma_delta, y
            #     #     last_total_error = total_error
            #     # else:
            #     #     early_stop = 1
            #     total_error = 0
            #     cycle_max_error = 0
            #     np.savetxt('voltage_his_learning_10sec_3kg.txt', voltage_his, delimiter=',')
            #     np.savetxt('angle_his_learning_10sec_3kg.txt', angle_his, delimiter=',')

            # else:
            #     total_error = total_error+(error**2)
            #     if cycle_max_error < error:
            #         cycle_max_error = error

            # if simulation == True:
            #     actual_angle = return_simulation_pma_angle(self.df_pma_angle,controller_u_output,actual_angle)

            # # time.sleep(system_time) #delay 0.1 sec
            # elapsed_time = time.time() - start_time

            # # 如果运行时间小于指定的总时间，则补充剩余的时间
            # if elapsed_time < total_duration:
            #     remaining_time = total_duration - elapsed_time
            #     time.sleep(remaining_time)
