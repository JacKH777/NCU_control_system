import time
import threading
import numpy as np
import os, shutil
from datetime import datetime
from kerasFuzzy import Torque_ANFIS_multi_pos_5Rule, go_to_desire_angle,Torque_ANFIS_2kg_multi
from encoder_function import encoder, forceGauge, moving_average_with_padding,calculate_rmse
import serial
from scipy import interpolate


# 初始化全局参数
sin_wave = np.asarray([30,30,30,30,30,30,30,30,30,30,30,30,30,30,31,31,31,31,31,31,32,32,32,32,33,33,33,34,34,35,35,35,36,36,37,37,38,38,39,39,40,41,41,42,42,43,44,44,45,45,46,47,48,48,49,50,50,51,52,53,53,54,55,55,56,57,58,58,59,60,60,61,62,62,63,64,65,65,66,67,67,68,69,70,70,71,72,72,73,74,75,75,76,76,77,78,78,79,79,80,81,81,82,82,83,83,84,84,85,85,85,86,86,87,87,87,88,88,88,88,89,89,89,89,89,89,90,90,90,90,90,90,90,90,90,90,90,90,90,90,89,89,89,89,89,88,88,88,88,87,87,87,86,86,86,85,85,84,84,83,83,82,82,81,81,80,80,79,79,78,77,77,76,75,75,74,74,73,72,71,71,70,69,69,68,67,66,66,65,64,63,63,62,61,60,60,59,58,57,57,56,55,54,54,53,52,51,51,50,49,49,48,47,46,46,45,45,44,43,43,42,41,41,40,40,39,39,38,38,37,37,36,36,35,35,34,34,34,33,33,33,32,32,32,32,31,31,31,31,31])
x = np.linspace(0, 1, 250)
f = interpolate.interp1d(x, sin_wave)
x_10 = np.linspace(0, 1, 250)
x_8 = np.linspace(0, 1, 200)
x_6 = np.linspace(0, 1, 150)
x_4 = np.linspace(0, 1, 100)
period_10 = f(x_10)
period_8 = f(x_8)
period_6 = f(x_6)
period_4 = f(x_4)

# Event 控制信号
train_event_up = threading.Event()
train_event_down = threading.Event()
stop_event = threading.Event()

# 角度计和传感器初始化
angle_encoder = encoder()
angle_encoder.get_com_port('COM7')
# force_gauge = forceGauge("COM16")
stm32 = serial.Serial('COM23', 115200)
print(f"Successfull Open COM7 and COM9")
angle = angle_encoder.get_angle()
# force = force_gauge.get_force()
# torque_bias = force_gauge.get_torque(angle,force)
# for i in range(20):
#     angle = angle_encoder.get_angle()
#     force = force_gauge.get_force()
#     torque_bias = (torque_bias+force_gauge.get_torque(angle,force))/2
#     print("torque_bias:",torque_bias)
desire_angle = sin_wave[0]
actual_angle = angle


# FNN 初始化
up = Torque_ANFIS_multi_pos_5Rule()
down = Torque_ANFIS_multi_pos_5Rule()

# 文件夹创建
ts = time.time()
data_time = datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H%M')
fileDir = './exp/{}'.format(data_time)
if not os.path.isdir(fileDir):
    os.makedirs(os.path.join(fileDir, '1'))
else:
    shutil.rmtree(fileDir)
    os.makedirs(os.path.join(fileDir, '1'))

voltage_his = np.asarray([])
up_model_his = np.asarray([])
down_model_his = np.asarray([])
angle_his = np.asarray([])
delta_error_his = np.asarray([])


# 主程序函数：用于数据采集与计算控制
def control_loop():
    global h_train,train_torque,train_desire_torque,voltage_his,angle_his, up_model_his, down_model_his
    target_period = period_10
    period = len(target_period)/25
    cycle = 0
    total_cycle = (100 / period) + 1

    idx = 0

    ku = 0.01
    controller_u = 0.5
    controller_u_output = 0
    error = 0
    
    while cycle < total_cycle:

        if cycle == 0 and idx == 100:
            idx = 0
            cycle += 1
        # 更新角度和力矩
        if cycle != 0 and idx == len(target_period):
            _,controller_u = go_to_desire_angle(encoder, stm32, 30, controller_u)
            idx = 0
            print(cycle)
            # if cycle == 1:
            #     up_rmse_his = calculate_rmse(moving_average_with_padding(torque_his[-len(target_period)+1:-int(len(target_period)-len(target_period)/2)+1]),target_period[:int(len(target_period)/2)])
            #     down_rmse_his = calculate_rmse(moving_average_with_padding(torque_his[-int(len(target_period)/2):]),target_period[int(len(target_period)/2):])
            # else:
            #     up_rmse = calculate_rmse(moving_average_with_padding(torque_his[-len(target_period)+1:-int(len(target_period)-len(target_period)/2)+1]),target_period[:int(len(target_period)/2)])
            #     down_rmse = calculate_rmse(moving_average_with_padding(torque_his[-int(len(target_period)/2):]),target_period[int(len(target_period)/2):])
            #     if up_rmse - up_rmse_his <= 0:
            #         value = up.return_learning_rate() * 0.9
            #         up.change_learning_rate(value)
            #         up_rmse_his = up_rmse
            #     else:
            #         value = up.return_learning_rate() * 1.1
            #         up.change_learning_rate(value)
            #         up_rmse_his = up_rmse
            #     if down_rmse - down_rmse_his <= 0:
            #         value = down.return_learning_rate() * 0.9
            #         down.change_learning_rate(value)
            #         down_rmse_his = down_rmse
            #     else:
            #         value = down.return_learning_rate() * 1.1
            #         down.change_learning_rate(value)
            #         down_rmse_his = down_rmse
            cycle += 1

        # 目標路徑
        if cycle == 0:
            desire_torque = target_period[0]
        else:
            desire_torque = target_period[idx]
        
        # 更新滤波后的力矩
        actual_angle = angle_encoder.get_angle()
        # h = angle_encoder.get_expansion(actual_angle)
        # force = force_gauge.get_force()
        # torque = force_gauge.get_torque(actual_angle,force) - torque_bias
        # torque_his = np.append(torque_his, torque)
        angle_his = np.append(angle_his, actual_angle)
        error = desire_torque - f_angle
        error_delta = error - last_error
        delta_error_his = np.append(delta_error_his, error_delta)
        last_error = error
        if len(angle_his) >=25:
            # f_torque = moving_average_with_padding(torque_his[-5:])[-1]
            f_angle = moving_average_with_padding(angle_his[-5:])[-1]
            f_delta_error = moving_average_with_padding(delta_error_his[-5:])[-1] * 25
            # f_torque_his = moving_average_with_padding(torque_his[-25:])[-2]
        else:
            # f_torque = torque
            f_angle = actual_angle
        
        # 计算误差
        # error = desire_torque - f_torque
        # print(error)
        # error = desire_torque - f_angle
        # error_delta = error - last_error
        # delta_error_his = np.append(delta_error_his, error_delta)
        # last_error = error

        # 控制器计算
        if idx < len(target_period) / 2:
            delta_u = up.predict([error], [f_delta_error])
        else:
            delta_u = down.predict([error], [f_delta_error])
        
        delta_u *= ku
        controller_u += float(delta_u)
        # voltage_his = np.append(voltage_his, controller_u)
        # f_controller_u = moving_average_with_padding(voltage_his[-25:])[-1]
        controller_u_output = int((controller_u / 10) * 65535)
        controller_u_output = max(0, controller_u_output)
        
        # 发送控制信号到 STM32
        stm32.write(controller_u_output.to_bytes(2, byteorder='big'))
        if actual_angle > 100:

            _,_ = go_to_desire_angle(angle_encoder, stm32, 30, controller_u)
            break

        # 判断是否触发训练
        if cycle > 1 and idx < len(target_period) / 2 and not train_event_up.is_set():
            # print("ok_up")
            up.update_parameter()
            up_model_his = np.append(up_model_his, up.return_model())
            # train_torque = torque_his[-int(len(target_period) / 2):] 
            # train_desire_torque = target_period[-int(len(target_period) / 2):]
            train_torque = error
            train_desire_torque = target_period[idx]
            h_train = f_delta_error
            train_event_up.set()  # 触发训练

        elif cycle > 1 and idx > len(target_period) / 2 and not train_event_down.is_set():
            down.update_parameter()
            down_model_his = np.append(down_model_his, down.return_model())
            # print("ok_down")
            # train_torque = torque_his[-int(len(target_period) / 2):]
            # train_desire_torque = target_period[:-int(len(target_period) / 2)]
            train_torque = error
            train_desire_torque = target_period[idx]
            h_train = f_delta_error
            train_event_down.set() 
        
        # 更新周期和索引
        idx += 1
        voltage_his = np.append(voltage_his, controller_u)
    # up_model_his = np.append(up_model_his, up.return_model())
    # down_model_his = np.append(down_model_his, down.return_model())
        # force_his = np.append(force_his, force)
        # h_his = np.append(h_his, h)
    save_path_voltage = os.path.join(fileDir, '1', 'voltage.npy')
    # save_path_torque = os.path.join(fileDir, '1', 'torque.npy')
    save_path_up_model = os.path.join(fileDir, '1', 'up_model.npy')
    save_path_down_model = os.path.join(fileDir, '1', 'down_model.npy')
    # save_path_force = os.path.join(fileDir, '1', 'force.npy')
    save_path_angle = os.path.join(fileDir, '1', 'angle.npy')
    # save_path_h = os.path.join(fileDir, '1', 'h.npy')
    save_path_delta_error = os.path.join(fileDir, '1', 'delta_error.npy')
    np.save(save_path_voltage, voltage_his)
    # np.save(save_path_torque, torque_his)
    np.save(save_path_up_model, up_model_his)
    np.save(save_path_down_model, down_model_his)
    # np.save(save_path_force, force_his)
    np.save(save_path_angle, angle_his)
    # np.save(save_path_h, h_his)
    np.save(save_path_delta_error, delta_error_his)

# FNN 模型训练函数
def fnn_train_up():
    global train_torque,train_desire_torque,h_train
    while not stop_event.is_set():  # 检查是否有停止信号
        train_event_up.wait()  # 等待控制信号
        if stop_event.is_set():
            break  # 如果收到停止信号，退出循环
        local_torque = train_torque
        local_desire_torque = train_desire_torque
        h_local = h_train
        # error = local_desire_torque - local_torque
        up.train([local_torque], [h_local], [local_desire_torque], [local_torque])
        # f_torque = moving_average_with_padding(local_torque)
        # # error = local_desire_torque - f_torque
        # # f_torque_1 =  [[value] for value in f_torque[:-1]]
        # # f_torque_2 =  [[value] for value in f_torque[1:]]
        # # local_desire_torque = [[value] for value in local_desire_torque[:-1]]
        # # error = [[value] for value in error[:-1]]
        # # 训练模型 up
        # for i in range(len(local_torque)-1):
        #     f_torque_1 = f_torque[i]
        #     error = local_desire_torque[i] - f_torque[i]
        #     f_torque_2 = f_torque[i]
        #     up.train([f_torque_1], [error], [local_desire_torque], [f_torque_2])
        train_event_up.clear()  # 训练完成后清除信号

def fnn_train_down():
    global train_torque,train_desire_torque,h_train
    while not stop_event.is_set():  # 检查是否有停止信号
        train_event_down.wait()  # 等待控制信号
        if stop_event.is_set():
            break  # 如果收到停止信号，退出循环
        local_torque = train_torque
        local_desire_torque = train_desire_torque
        h_local = h_train
        # error = local_desire_torque - local_torque
        down.train([local_torque], [h_local], [local_desire_torque], [local_torque])
        # f_torque = moving_average_with_padding(local_torque)
        # # error = local_desire_torque - f_torque
        # # f_torque_1 =  [[value] for value in f_torque[:-1]]
        # # f_torque_2 =  [[value] for value in f_torque[1:]]
        # # local_desire_torque = [[value] for value in local_desire_torque[:-1]]
        # # error = [[value] for value in error[:-1]]
        # # 训练模型 up
        # for i in range(len(local_torque)-1):
        #     f_torque_1 = f_torque[i]
        #     error = local_desire_torque[i] - f_torque[i]
        #     f_torque_2 = f_torque[i]
        #     down.train([f_torque_1], [error], [local_desire_torque], [f_torque_2])
        train_event_down.clear()  # 训练完成后清除信号

# 启动线程
control_thread = threading.Thread(target=control_loop)
train_up_thread = threading.Thread(target=fnn_train_up)
train_down_thread = threading.Thread(target=fnn_train_down)

# 开始线程执行
control_thread.start()
train_up_thread.start()
train_down_thread.start()

# 主线程等待control_thread结束
control_thread.join()

# 结束训练线程
stop_event.set()  # 发送停止信号
train_event_up.set()  # 防止线程阻塞在 wait 状态
train_event_down.set()  # 防止线程阻塞在 wait 状态

# 等待训练线程结束
train_up_thread.join()
train_down_thread.join()
# 保存记录