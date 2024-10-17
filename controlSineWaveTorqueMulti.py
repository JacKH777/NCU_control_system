# import time
# import threading
# import numpy as np
# import os, shutil
# from datetime import datetime
# from kerasFuzzy import Torque_ANFIS_multi, go_to_desire_angle,Torque_ANFIS_2kg_multi
# from encoder_function import encoder, forceGauge, moving_average_with_padding,calculate_rmse
# import serial
# from scipy import interpolate


# # 初始化全局参数
# # sine_angle_torque_1kg = np.asarray([0.005,0.0051,0.0054,0.006,0.0068,0.0078,0.009,0.0104,0.0121,0.014,0.0161,0.0184,0.0209,0.0237,0.0266,0.0298,0.0331,0.0367,0.0405,0.0445,0.0486,0.053,0.0576,0.0623,0.0672,0.0724,0.0777,0.0831,0.0888,0.0946,0.1006,0.1067,0.113,0.1195,0.1261,0.1329,0.1398,0.1468,0.1539,0.1612,0.1687,0.1762,0.1838,0.1916,0.1995,0.2074,0.2155,0.2236,0.2318,0.2401,0.2485,0.257,0.2655,0.274,0.2827,0.2913,0.3,0.3088,0.3175,0.3263,0.3351,0.344,0.3528,0.3616,0.3705,0.3793,0.3881,0.3969,0.4056,0.4143,0.423,0.4317,0.4402,0.4488,0.4573,0.4657,0.474,0.4823,0.4905,0.4986,0.5066,0.5145,0.5223,0.53,0.5376,0.5451,0.5524,0.5597,0.5668,0.5737,0.5805,0.5872,0.5938,0.6001,0.6064,0.6124,0.6183,0.6241,0.6296,0.635,0.6402,0.6452,0.6501,0.6547,0.6592,0.6635,0.6676,0.6714,0.6751,0.6786,0.6818,0.6849,0.6877,0.6904,0.6928,0.695,0.697,0.6988,0.7003,0.7016,0.7027,0.7036,0.7043,0.7047,0.705,0.705,0.7047,0.7043,0.7036,0.7027,0.7016,0.7003,0.6988,0.697,0.695,0.6928,0.6904,0.6877,0.6849,0.6818,0.6786,0.6751,0.6714,0.6676,0.6635,0.6592,0.6547,0.6501,0.6452,0.6402,0.635,0.6296,0.6241,0.6183,0.6124,0.6064,0.6001,0.5938,0.5872,0.5805,0.5737,0.5668,0.5597,0.5524,0.5451,0.5376,0.53,0.5223,0.5145,0.5066,0.4986,0.4905,0.4823,0.474,0.4657,0.4573,0.4488,0.4402,0.4317,0.423,0.4143,0.4056,0.3969,0.3881,0.3793,0.3705,0.3616,0.3528,0.344,0.3351,0.3263,0.3175,0.3088,0.3,0.2913,0.2827,0.274,0.2655,0.257,0.2485,0.2401,0.2318,0.2236,0.2155,0.2074,0.1995,0.1916,0.1838,0.1762,0.1687,0.1612,0.1539,0.1468,0.1398,0.1329,0.1261,0.1195,0.113,0.1067,0.1006,0.0946,0.0888,0.0831,0.0777,0.0724,0.0672,0.0623,0.0576,0.053,0.0486,0.0445,0.0405,0.0367,0.0331,0.0298,0.0266,0.0237,0.0209,0.0184,0.0161,0.014,0.0121,0.0104,0.009,0.0078,0.0068,0.006,0.0054,0.0051,0.005])
# sine_angle_torque_2kg = np.asarray([0.02,0.0203,0.0211,0.0225,0.0245,0.027,0.0301,0.0337,0.0379,0.0426,0.0479,0.0537,0.06,0.0669,0.0743,0.0823,0.0908,0.0997,0.1092,0.1192,0.1297,0.1407,0.1522,0.1641,0.1765,0.1894,0.2027,0.2165,0.2307,0.2453,0.2603,0.2758,0.2916,0.3079,0.3245,0.3415,0.3588,0.3765,0.3945,0.4128,0.4315,0.4504,0.4697,0.4892,0.5089,0.5289,0.5492,0.5697,0.5903,0.6112,0.6323,0.6535,0.6749,0.6965,0.7181,0.7399,0.7618,0.7838,0.8058,0.8279,0.8501,0.8722,0.8944,0.9167,0.9388,0.961,0.9831,1.0052,1.0272,1.0492,1.071,1.0927,1.1143,1.1358,1.1571,1.1783,1.1992,1.22,1.2406,1.261,1.2811,1.301,1.3206,1.34,1.3591,1.3779,1.3964,1.4146,1.4324,1.4499,1.4671,1.4839,1.5003,1.5163,1.532,1.5472,1.5621,1.5765,1.5905,1.604,1.6171,1.6298,1.6419,1.6536,1.6649,1.6756,1.6858,1.6956,1.7048,1.7135,1.7217,1.7294,1.7366,1.7432,1.7493,1.7548,1.7598,1.7643,1.7682,1.7715,1.7743,1.7766,1.7782,1.7794,1.7799,1.7799,1.7794,1.7782,1.7766,1.7743,1.7715,1.7682,1.7643,1.7598,1.7548,1.7493,1.7432,1.7366,1.7294,1.7217,1.7135,1.7048,1.6956,1.6858,1.6756,1.6649,1.6536,1.6419,1.6298,1.6171,1.604,1.5905,1.5765,1.5621,1.5472,1.532,1.5163,1.5003,1.4839,1.4671,1.4499,1.4324,1.4146,1.3964,1.3779,1.3591,1.34,1.3206,1.301,1.2811,1.261,1.2406,1.22,1.1992,1.1783,1.1571,1.1358,1.1143,1.0927,1.071,1.0492,1.0272,1.0052,0.9831,0.961,0.9388,0.9167,0.8944,0.8722,0.8501,0.8279,0.8058,0.7838,0.7618,0.7399,0.7181,0.6965,0.6749,0.6535,0.6323,0.6112,0.5903,0.5697,0.5492,0.5289,0.5089,0.4892,0.4697,0.4504,0.4315,0.4128,0.3945,0.3765,0.3588,0.3415,0.3245,0.3079,0.2916,0.2758,0.2603,0.2453,0.2307,0.2165,0.2027,0.1894,0.1765,0.1641,0.1522,0.1407,0.1297,0.1192,0.1092,0.0997,0.0908,0.0823,0.0743,0.0669,0.06,0.0537,0.0479,0.0426,0.0379,0.0337,0.0301,0.027,0.0245,0.0225,0.0211,0.0203,0.02])
# x = np.linspace(0, 1, 250)
# f = interpolate.interp1d(x, sine_angle_torque_2kg)
# x_10 = np.linspace(0, 1, 250)
# x_8 = np.linspace(0, 1, 200)
# x_6 = np.linspace(0, 1, 150)
# x_4 = np.linspace(0, 1, 100)
# period_10 = f(x_10)
# period_8 = f(x_8)
# period_6 = f(x_6)
# period_4 = f(x_4)

# # Event 控制信号
# train_event_up = threading.Event()
# train_event_down = threading.Event()
# stop_event = threading.Event()

# # 角度计和传感器初始化
# angle_encoder = encoder()
# angle_encoder.get_com_port('COM7')
# force_gauge = forceGauge("COM16")
# stm32 = serial.Serial('COM23', 115200)
# print(f"Successfull Open COM7 and COM9")
# angle = angle_encoder.get_angle()
# force = force_gauge.get_force()
# torque_bias = force_gauge.get_torque(angle,force)
# for i in range(20):
#     angle = angle_encoder.get_angle()
#     force = force_gauge.get_force()
#     torque_bias = (torque_bias+force_gauge.get_torque(angle,force))/2
#     print("torque_bias:",torque_bias)
# desire_angle = sine_angle_torque_2kg[0]
# actual_angle = angle


# # FNN 初始化
# up =Torque_ANFIS_multi()
# down = Torque_ANFIS_multi()
# # up.change_learning_rate(0.5)


# # up.change_learning_rate(0.1)
# # down.change_learning_rate(0.1)


# # down.mf_torque.assign(np.asarray([0.2, 0.39, 0.58]))
# # down.mf_torque_predict.assign(np.asarray([0.2, 0.39, 0.58]))

# # up = Torque_ANFIS_2kg_multi()
# # down = Torque_ANFIS_2kg_multi()
# # down.mf_torque.assign(np.asarray([0.43,0.935, 1.44]))
# # down.mf_torque_predict.assign(np.asarray([0.43,0.935, 1.44]))

# # down.mf_torque_predict.assign(np.asarray([0.5, 1.05, 1.6]))
# # down.mf_torque.assign(np.asarray([0.5, 1.05, 1.6]))

# # 文件夹创建
# ts = time.time()
# data_time = datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H%M')
# fileDir = './exp/{}'.format(data_time)
# if not os.path.isdir(fileDir):
#     os.makedirs(os.path.join(fileDir, '1'))
# else:
#     shutil.rmtree(fileDir)
#     os.makedirs(os.path.join(fileDir, '1'))

# voltage_his = np.asarray([])
# torque_his = np.asarray([])
# up_model_his = np.asarray([])
# down_model_his = np.asarray([])
# force_his = np.asarray([])
# angle_his = np.asarray([])
# h_his = np.asarray([])

# # 主程序函数：用于数据采集与计算控制
# def control_loop():
#     global h_train,train_torque,train_desire_torque,torque_his, voltage_his, force_his, angle_his, up_model_his, down_model_his
#     target_period = period_10
#     period = len(target_period)/25
#     cycle = 0
#     total_cycle = (100 / period) + 1

#     idx = 0

#     ku = 0.01
#     controller_u = 0.5
#     controller_u_output = 0
#     error = 0
    
#     while cycle < total_cycle:

#         if cycle == 0 and idx == 100:
#             idx = 0
#             cycle += 1
#         # 更新角度和力矩
#         if cycle != 0 and idx == len(target_period):
#             idx = 0
#             print(cycle)
#             if cycle == 1:
#                 up_rmse_his = calculate_rmse(moving_average_with_padding(torque_his[-len(target_period)+1:-int(len(target_period)-len(target_period)/2)+1]),target_period[:int(len(target_period)/2)])
#                 down_rmse_his = calculate_rmse(moving_average_with_padding(torque_his[-int(len(target_period)/2):]),target_period[int(len(target_period)/2):])
#             else:
#                 up_rmse = calculate_rmse(moving_average_with_padding(torque_his[-len(target_period)+1:-int(len(target_period)-len(target_period)/2)+1]),target_period[:int(len(target_period)/2)])
#                 down_rmse = calculate_rmse(moving_average_with_padding(torque_his[-int(len(target_period)/2):]),target_period[int(len(target_period)/2):])
#                 if up_rmse - up_rmse_his <= 0:
#                     value = up.return_learning_rate() * 0.9
#                     up.change_learning_rate(value)
#                     up_rmse_his = up_rmse
#                 else:
#                     value = up.return_learning_rate() * 1.1
#                     up.change_learning_rate(value)
#                     up_rmse_his = up_rmse
#                 if down_rmse - down_rmse_his <= 0:
#                     value = down.return_learning_rate() * 0.9
#                     down.change_learning_rate(value)
#                     down_rmse_his = down_rmse
#                 else:
#                     value = down.return_learning_rate() * 1.1
#                     down.change_learning_rate(value)
#                     down_rmse_his = down_rmse
#             cycle += 1

#         # up_model_his = np.append(up_model_his, up.return_model())
#         # down_model_his = np.append(down_model_his, down.return_model())

#         # 目標路徑
#         if cycle == 0:
#             desire_torque = target_period[0]
#         else:
#             desire_torque = target_period[idx]
        
#         # 更新滤波后的力矩
#         actual_angle = angle_encoder.get_angle()
#         # h = angle_encoder.get_expansion(actual_angle)
#         force = force_gauge.get_force()
#         torque = force_gauge.get_torque(actual_angle,force) - torque_bias
#         torque_his = np.append(torque_his, torque)
#         angle_his = np.append(angle_his, actual_angle)
#         if len(torque_his) >=25:
#             f_torque = moving_average_with_padding(torque_his[-5:])[-1]
#             f_angle = moving_average_with_padding(angle_his[-5:])[-1]
#             # f_torque_his = moving_average_with_padding(torque_his[-25:])[-2]
#         else:
#             f_torque = torque
#             f_angle = actual_angle
        
#         # 计算误差
#         error = desire_torque - f_torque
#         # print(error)
        
#         # 控制器计算
#         if idx < len(target_period) / 2:
#             delta_u = up.predict([f_angle], [error])
#         else:
#             delta_u = down.predict([f_angle], [error])
        
#         delta_u *= ku
#         controller_u += float(delta_u)
#         # voltage_his = np.append(voltage_his, controller_u)
#         # f_controller_u = moving_average_with_padding(voltage_his[-25:])[-1]
#         controller_u_output = int((controller_u / 10) * 65535)
#         controller_u_output = max(0, controller_u_output)
        
#         # 发送控制信号到 STM32
#         stm32.write(controller_u_output.to_bytes(2, byteorder='big'))
#         if actual_angle > 100 or force > 110:

#             _,_ = go_to_desire_angle(angle_encoder, stm32, 30, controller_u)
#             break

#         # 判断是否触发训练
#         if cycle > 1 and idx < len(target_period) / 2 and not train_event_up.is_set():
#             # print("ok_up")
#             up.update_parameter()
#             up_model_his = np.append(up_model_his, up.return_model())
#             # train_torque = torque_his[-int(len(target_period) / 2):] 
#             # train_desire_torque = target_period[-int(len(target_period) / 2):]
#             train_torque = f_torque
#             train_desire_torque = target_period[idx]
#             h_train = f_angle
#             train_event_up.set()  # 触发训练

#         elif cycle > 1 and idx > len(target_period) / 2 and not train_event_down.is_set():
#             down.update_parameter()
#             down_model_his = np.append(down_model_his, down.return_model())
#             # print("ok_down")
#             # train_torque = torque_his[-int(len(target_period) / 2):]
#             # train_desire_torque = target_period[:-int(len(target_period) / 2)]
#             train_torque = f_torque
#             train_desire_torque = target_period[idx]
#             h_train = f_angle
#             train_event_down.set() 
        
#         # 更新周期和索引
#         idx += 1
#         voltage_his = np.append(voltage_his, controller_u)
#     # up_model_his = np.append(up_model_his, up.return_model())
#     # down_model_his = np.append(down_model_his, down.return_model())
#         force_his = np.append(force_his, force)
#         # h_his = np.append(h_his, h)
#     save_path_voltage = os.path.join(fileDir, '1', 'voltage.npy')
#     save_path_torque = os.path.join(fileDir, '1', 'torque.npy')
#     save_path_up_model = os.path.join(fileDir, '1', 'up_model.npy')
#     save_path_down_model = os.path.join(fileDir, '1', 'down_model.npy')
#     save_path_force = os.path.join(fileDir, '1', 'force.npy')
#     save_path_angle = os.path.join(fileDir, '1', 'angle.npy')
#     save_path_h = os.path.join(fileDir, '1', 'h.npy')

#     np.save(save_path_voltage, voltage_his)
#     np.save(save_path_torque, torque_his)
#     np.save(save_path_up_model, up_model_his)
#     np.save(save_path_down_model, down_model_his)
#     np.save(save_path_force, force_his)
#     np.save(save_path_angle, angle_his)
#     np.save(save_path_h, h_his)


# # FNN 模型训练函数
# def fnn_train_up():
#     global train_torque,train_desire_torque,h_train
#     while not stop_event.is_set():  # 检查是否有停止信号
#         train_event_up.wait()  # 等待控制信号
#         if stop_event.is_set():
#             break  # 如果收到停止信号，退出循环
#         local_torque = train_torque
#         local_desire_torque = train_desire_torque
#         h_local = h_train
#         error = local_desire_torque - local_torque
#         up.train([h_local], [error], [local_desire_torque], [local_torque])
#         # f_torque = moving_average_with_padding(local_torque)
#         # # error = local_desire_torque - f_torque
#         # # f_torque_1 =  [[value] for value in f_torque[:-1]]
#         # # f_torque_2 =  [[value] for value in f_torque[1:]]
#         # # local_desire_torque = [[value] for value in local_desire_torque[:-1]]
#         # # error = [[value] for value in error[:-1]]
#         # # 训练模型 up
#         # for i in range(len(local_torque)-1):
#         #     f_torque_1 = f_torque[i]
#         #     error = local_desire_torque[i] - f_torque[i]
#         #     f_torque_2 = f_torque[i]
#         #     up.train([f_torque_1], [error], [local_desire_torque], [f_torque_2])
#         train_event_up.clear()  # 训练完成后清除信号

# def fnn_train_down():
#     global train_torque,train_desire_torque,h_train
#     while not stop_event.is_set():  # 检查是否有停止信号
#         train_event_down.wait()  # 等待控制信号
#         if stop_event.is_set():
#             break  # 如果收到停止信号，退出循环
#         local_torque = train_torque
#         local_desire_torque = train_desire_torque
#         h_local = h_train
#         error = local_desire_torque - local_torque
#         down.train([h_local], [error], [local_desire_torque], [local_torque])
#         # f_torque = moving_average_with_padding(local_torque)
#         # # error = local_desire_torque - f_torque
#         # # f_torque_1 =  [[value] for value in f_torque[:-1]]
#         # # f_torque_2 =  [[value] for value in f_torque[1:]]
#         # # local_desire_torque = [[value] for value in local_desire_torque[:-1]]
#         # # error = [[value] for value in error[:-1]]
#         # # 训练模型 up
#         # for i in range(len(local_torque)-1):
#         #     f_torque_1 = f_torque[i]
#         #     error = local_desire_torque[i] - f_torque[i]
#         #     f_torque_2 = f_torque[i]
#         #     down.train([f_torque_1], [error], [local_desire_torque], [f_torque_2])
#         train_event_down.clear()  # 训练完成后清除信号

# # 启动线程
# control_thread = threading.Thread(target=control_loop)
# train_up_thread = threading.Thread(target=fnn_train_up)
# train_down_thread = threading.Thread(target=fnn_train_down)

# # 开始线程执行
# control_thread.start()
# train_up_thread.start()
# train_down_thread.start()

# # 主线程等待control_thread结束
# control_thread.join()

# # 结束训练线程
# stop_event.set()  # 发送停止信号
# train_event_up.set()  # 防止线程阻塞在 wait 状态
# train_event_down.set()  # 防止线程阻塞在 wait 状态

# # 等待训练线程结束
# train_up_thread.join()
# train_down_thread.join()
# # 保存记录

########################## 1fnn
# import time
# import threading
# import numpy as np
# import os, shutil
# from datetime import datetime
# from kerasFuzzy import Torque_ANFIS_multi, go_to_desire_angle,Torque_ANFIS_2kg_multi
# from encoder_function import encoder, forceGauge, moving_average_with_padding,calculate_rmse
# import serial
# from scipy import interpolate


# # 初始化全局参数
# # sine_angle_torque_1kg = np.asarray([0.005,0.0051,0.0054,0.006,0.0068,0.0078,0.009,0.0104,0.0121,0.014,0.0161,0.0184,0.0209,0.0237,0.0266,0.0298,0.0331,0.0367,0.0405,0.0445,0.0486,0.053,0.0576,0.0623,0.0672,0.0724,0.0777,0.0831,0.0888,0.0946,0.1006,0.1067,0.113,0.1195,0.1261,0.1329,0.1398,0.1468,0.1539,0.1612,0.1687,0.1762,0.1838,0.1916,0.1995,0.2074,0.2155,0.2236,0.2318,0.2401,0.2485,0.257,0.2655,0.274,0.2827,0.2913,0.3,0.3088,0.3175,0.3263,0.3351,0.344,0.3528,0.3616,0.3705,0.3793,0.3881,0.3969,0.4056,0.4143,0.423,0.4317,0.4402,0.4488,0.4573,0.4657,0.474,0.4823,0.4905,0.4986,0.5066,0.5145,0.5223,0.53,0.5376,0.5451,0.5524,0.5597,0.5668,0.5737,0.5805,0.5872,0.5938,0.6001,0.6064,0.6124,0.6183,0.6241,0.6296,0.635,0.6402,0.6452,0.6501,0.6547,0.6592,0.6635,0.6676,0.6714,0.6751,0.6786,0.6818,0.6849,0.6877,0.6904,0.6928,0.695,0.697,0.6988,0.7003,0.7016,0.7027,0.7036,0.7043,0.7047,0.705,0.705,0.7047,0.7043,0.7036,0.7027,0.7016,0.7003,0.6988,0.697,0.695,0.6928,0.6904,0.6877,0.6849,0.6818,0.6786,0.6751,0.6714,0.6676,0.6635,0.6592,0.6547,0.6501,0.6452,0.6402,0.635,0.6296,0.6241,0.6183,0.6124,0.6064,0.6001,0.5938,0.5872,0.5805,0.5737,0.5668,0.5597,0.5524,0.5451,0.5376,0.53,0.5223,0.5145,0.5066,0.4986,0.4905,0.4823,0.474,0.4657,0.4573,0.4488,0.4402,0.4317,0.423,0.4143,0.4056,0.3969,0.3881,0.3793,0.3705,0.3616,0.3528,0.344,0.3351,0.3263,0.3175,0.3088,0.3,0.2913,0.2827,0.274,0.2655,0.257,0.2485,0.2401,0.2318,0.2236,0.2155,0.2074,0.1995,0.1916,0.1838,0.1762,0.1687,0.1612,0.1539,0.1468,0.1398,0.1329,0.1261,0.1195,0.113,0.1067,0.1006,0.0946,0.0888,0.0831,0.0777,0.0724,0.0672,0.0623,0.0576,0.053,0.0486,0.0445,0.0405,0.0367,0.0331,0.0298,0.0266,0.0237,0.0209,0.0184,0.0161,0.014,0.0121,0.0104,0.009,0.0078,0.0068,0.006,0.0054,0.0051,0.005])
# sine_angle_torque_2kg = np.asarray([0.02,0.0203,0.0211,0.0225,0.0245,0.027,0.0301,0.0337,0.0379,0.0426,0.0479,0.0537,0.06,0.0669,0.0743,0.0823,0.0908,0.0997,0.1092,0.1192,0.1297,0.1407,0.1522,0.1641,0.1765,0.1894,0.2027,0.2165,0.2307,0.2453,0.2603,0.2758,0.2916,0.3079,0.3245,0.3415,0.3588,0.3765,0.3945,0.4128,0.4315,0.4504,0.4697,0.4892,0.5089,0.5289,0.5492,0.5697,0.5903,0.6112,0.6323,0.6535,0.6749,0.6965,0.7181,0.7399,0.7618,0.7838,0.8058,0.8279,0.8501,0.8722,0.8944,0.9167,0.9388,0.961,0.9831,1.0052,1.0272,1.0492,1.071,1.0927,1.1143,1.1358,1.1571,1.1783,1.1992,1.22,1.2406,1.261,1.2811,1.301,1.3206,1.34,1.3591,1.3779,1.3964,1.4146,1.4324,1.4499,1.4671,1.4839,1.5003,1.5163,1.532,1.5472,1.5621,1.5765,1.5905,1.604,1.6171,1.6298,1.6419,1.6536,1.6649,1.6756,1.6858,1.6956,1.7048,1.7135,1.7217,1.7294,1.7366,1.7432,1.7493,1.7548,1.7598,1.7643,1.7682,1.7715,1.7743,1.7766,1.7782,1.7794,1.7799,1.7799,1.7794,1.7782,1.7766,1.7743,1.7715,1.7682,1.7643,1.7598,1.7548,1.7493,1.7432,1.7366,1.7294,1.7217,1.7135,1.7048,1.6956,1.6858,1.6756,1.6649,1.6536,1.6419,1.6298,1.6171,1.604,1.5905,1.5765,1.5621,1.5472,1.532,1.5163,1.5003,1.4839,1.4671,1.4499,1.4324,1.4146,1.3964,1.3779,1.3591,1.34,1.3206,1.301,1.2811,1.261,1.2406,1.22,1.1992,1.1783,1.1571,1.1358,1.1143,1.0927,1.071,1.0492,1.0272,1.0052,0.9831,0.961,0.9388,0.9167,0.8944,0.8722,0.8501,0.8279,0.8058,0.7838,0.7618,0.7399,0.7181,0.6965,0.6749,0.6535,0.6323,0.6112,0.5903,0.5697,0.5492,0.5289,0.5089,0.4892,0.4697,0.4504,0.4315,0.4128,0.3945,0.3765,0.3588,0.3415,0.3245,0.3079,0.2916,0.2758,0.2603,0.2453,0.2307,0.2165,0.2027,0.1894,0.1765,0.1641,0.1522,0.1407,0.1297,0.1192,0.1092,0.0997,0.0908,0.0823,0.0743,0.0669,0.06,0.0537,0.0479,0.0426,0.0379,0.0337,0.0301,0.027,0.0245,0.0225,0.0211,0.0203,0.02])
# x = np.linspace(0, 1, 250)
# f = interpolate.interp1d(x, sine_angle_torque_2kg)
# x_10 = np.linspace(0, 1, 250)
# x_8 = np.linspace(0, 1, 200)
# x_6 = np.linspace(0, 1, 150)
# x_4 = np.linspace(0, 1, 100)
# period_10 = f(x_10)
# period_8 = f(x_8)
# period_6 = f(x_6)
# period_4 = f(x_4)

# # Event 控制信号
# train_event_up = threading.Event()
# train_event_down = threading.Event()
# stop_event = threading.Event()

# # 角度计和传感器初始化
# angle_encoder = encoder()
# angle_encoder.get_com_port('COM7')
# force_gauge = forceGauge("COM16")
# stm32 = serial.Serial('COM23', 115200)
# print(f"Successfull Open COM7 and COM9")
# angle = angle_encoder.get_angle()
# force = force_gauge.get_force()
# torque_bias = force_gauge.get_torque(angle,force)
# for i in range(20):
#     angle = angle_encoder.get_angle()
#     force = force_gauge.get_force()
#     torque_bias = (torque_bias+force_gauge.get_torque(angle,force))/2
#     print("torque_bias:",torque_bias)
# desire_angle = sine_angle_torque_2kg[0]
# actual_angle = angle


# # FNN 初始化
# up =Torque_ANFIS_multi()

# up.change_learning_rate(0.04)

# # 文件夹创建
# ts = time.time()
# data_time = datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H%M')
# fileDir = './exp/{}'.format(data_time)
# if not os.path.isdir(fileDir):
#     os.makedirs(os.path.join(fileDir, '1'))
# else:
#     shutil.rmtree(fileDir)
#     os.makedirs(os.path.join(fileDir, '1'))

# voltage_his = np.asarray([])
# torque_his = np.asarray([])
# up_model_his = np.asarray([])
# down_model_his = np.asarray([])
# force_his = np.asarray([])
# angle_his = np.asarray([])
# h_his = np.asarray([])

# # 主程序函数：用于数据采集与计算控制
# def control_loop():
#     global h_train,train_torque,train_desire_torque,torque_his, voltage_his, force_his, angle_his, up_model_his, down_model_his
#     target_period = period_10
#     period = len(target_period)/25
#     cycle = 0
#     total_cycle = (100 / period) + 1

#     idx = 0

#     ku = 0.01
#     controller_u = 0.5
#     controller_u_output = 0
#     error = 0
    
#     while cycle < total_cycle:

#         if cycle == 0 and idx == 100:
#             idx = 0
#             cycle += 1
#         # 更新角度和力矩
#         if cycle != 0 and idx == len(target_period):
#             idx = 0
#             print(cycle)
#             if cycle == 1:
#                 up_rmse_his = calculate_rmse(moving_average_with_padding(torque_his[-len(target_period):]),target_period[:])
               
#             else:
#                 up_rmse = calculate_rmse(moving_average_with_padding(torque_his[-len(target_period):]),target_period[:])
#                 if up_rmse - up_rmse_his <= 0:
#                     value = up.return_learning_rate() * 0.9
#                     up.change_learning_rate(value)
#                     up_rmse_his = up_rmse
#                 else:
#                     value = up.return_learning_rate() * 1.1
#                     up.change_learning_rate(value)
#                     up_rmse_his = up_rmse
#             cycle += 1

#         # up_model_his = np.append(up_model_his, up.return_model())
#         # down_model_his = np.append(down_model_his, down.return_model())

#         # 目標路徑
#         if cycle == 0:
#             desire_torque = target_period[0]
#         else:
#             desire_torque = target_period[idx]
        
#         # 更新滤波后的力矩
#         actual_angle = angle_encoder.get_angle()
#         # h = angle_encoder.get_expansion(actual_angle)
#         force = force_gauge.get_force()
#         torque = force_gauge.get_torque(actual_angle,force) - torque_bias
#         torque_his = np.append(torque_his, torque)
#         angle_his = np.append(angle_his, actual_angle)
#         if len(torque_his) >=25:
#             f_torque = moving_average_with_padding(torque_his[-5:])[-1]
#             f_angle = moving_average_with_padding(angle_his[-5:])[-1]
#             # f_torque_his = moving_average_with_padding(torque_his[-25:])[-2]
#         else:
#             f_torque = torque
#             f_angle = actual_angle
        
#         # 计算误差
#         error = desire_torque - f_torque
#         # print(error)
        
#         # 控制器计算
#         delta_u = up.predict([f_angle], [error])
        
#         delta_u *= ku
#         controller_u += float(delta_u)
#         # voltage_his = np.append(voltage_his, controller_u)
#         # f_controller_u = moving_average_with_padding(voltage_his[-25:])[-1]
#         controller_u_output = int((controller_u / 10) * 65535)
#         controller_u_output = max(0, controller_u_output)
        
#         # 发送控制信号到 STM32
#         stm32.write(controller_u_output.to_bytes(2, byteorder='big'))
#         if actual_angle > 100 or force > 110:

#             _,_ = go_to_desire_angle(angle_encoder, stm32, 30, controller_u)
#             break

#         # 判断是否触发训练
#         if cycle > 1 and not train_event_up.is_set():
#             # print("ok_up")
#             up.update_parameter()
#             up_model_his = np.append(up_model_his, up.return_model())
#             # train_torque = torque_his[-int(len(target_period) / 2):] 
#             # train_desire_torque = target_period[-int(len(target_period) / 2):]
#             train_torque = f_torque
#             train_desire_torque = target_period[idx]
#             h_train = f_angle
#             train_event_up.set()  # 触发训练

        
#         # 更新周期和索引
#         idx += 1
#         voltage_his = np.append(voltage_his, controller_u)
#     # up_model_his = np.append(up_model_his, up.return_model())
#     # down_model_his = np.append(down_model_his, down.return_model())
#         force_his = np.append(force_his, force)
#         # h_his = np.append(h_his, h)
#     save_path_voltage = os.path.join(fileDir, '1', 'voltage.npy')
#     save_path_torque = os.path.join(fileDir, '1', 'torque.npy')
#     save_path_up_model = os.path.join(fileDir, '1', 'up_model.npy')
#     save_path_down_model = os.path.join(fileDir, '1', 'down_model.npy')
#     save_path_force = os.path.join(fileDir, '1', 'force.npy')
#     save_path_angle = os.path.join(fileDir, '1', 'angle.npy')
#     save_path_h = os.path.join(fileDir, '1', 'h.npy')

#     np.save(save_path_voltage, voltage_his)
#     np.save(save_path_torque, torque_his)
#     np.save(save_path_up_model, up_model_his)
#     np.save(save_path_down_model, down_model_his)
#     np.save(save_path_force, force_his)
#     np.save(save_path_angle, angle_his)
#     np.save(save_path_h, h_his)


# # FNN 模型训练函数
# def fnn_train_up():
#     global train_torque,train_desire_torque,h_train
#     while not stop_event.is_set():  # 检查是否有停止信号
#         train_event_up.wait()  # 等待控制信号
#         if stop_event.is_set():
#             break  # 如果收到停止信号，退出循环
#         local_torque = train_torque
#         local_desire_torque = train_desire_torque
#         h_local = h_train
#         error = local_desire_torque - local_torque
#         up.train([h_local], [error], [local_desire_torque], [local_torque])
#         # f_torque = moving_average_with_padding(local_torque)
#         # # error = local_desire_torque - f_torque
#         # # f_torque_1 =  [[value] for value in f_torque[:-1]]
#         # # f_torque_2 =  [[value] for value in f_torque[1:]]
#         # # local_desire_torque = [[value] for value in local_desire_torque[:-1]]
#         # # error = [[value] for value in error[:-1]]
#         # # 训练模型 up
#         # for i in range(len(local_torque)-1):
#         #     f_torque_1 = f_torque[i]
#         #     error = local_desire_torque[i] - f_torque[i]
#         #     f_torque_2 = f_torque[i]
#         #     up.train([f_torque_1], [error], [local_desire_torque], [f_torque_2])
#         train_event_up.clear()  # 训练完成后清除信号

# # 启动线程
# control_thread = threading.Thread(target=control_loop)
# train_up_thread = threading.Thread(target=fnn_train_up)

# # 开始线程执行
# control_thread.start()
# train_up_thread.start()

# # 主线程等待control_thread结束
# control_thread.join()

# # 结束训练线程
# stop_event.set()  # 发送停止信号
# train_event_up.set()  # 防止线程阻塞在 wait 状态

# # 等待训练线程结束
# train_up_thread.join()
# # 保存记录
################1FNN

import time
import threading
import numpy as np
import os, shutil
from datetime import datetime
from kerasFuzzy import Torque_ANFIS_multi_5fuzzy, go_to_desire_angle,Torque_ANFIS_2kg_multi
from encoder_function import encoder, forceGauge, moving_average_with_padding,calculate_rmse
import serial
from scipy import interpolate
from generate_sine_wave import generate_sine_wave


# 初始化全局参数
# sine_angle_torque_1kg = np.asarray([0.005,0.0051,0.0054,0.006,0.0068,0.0078,0.009,0.0104,0.0121,0.014,0.0161,0.0184,0.0209,0.0237,0.0266,0.0298,0.0331,0.0367,0.0405,0.0445,0.0486,0.053,0.0576,0.0623,0.0672,0.0724,0.0777,0.0831,0.0888,0.0946,0.1006,0.1067,0.113,0.1195,0.1261,0.1329,0.1398,0.1468,0.1539,0.1612,0.1687,0.1762,0.1838,0.1916,0.1995,0.2074,0.2155,0.2236,0.2318,0.2401,0.2485,0.257,0.2655,0.274,0.2827,0.2913,0.3,0.3088,0.3175,0.3263,0.3351,0.344,0.3528,0.3616,0.3705,0.3793,0.3881,0.3969,0.4056,0.4143,0.423,0.4317,0.4402,0.4488,0.4573,0.4657,0.474,0.4823,0.4905,0.4986,0.5066,0.5145,0.5223,0.53,0.5376,0.5451,0.5524,0.5597,0.5668,0.5737,0.5805,0.5872,0.5938,0.6001,0.6064,0.6124,0.6183,0.6241,0.6296,0.635,0.6402,0.6452,0.6501,0.6547,0.6592,0.6635,0.6676,0.6714,0.6751,0.6786,0.6818,0.6849,0.6877,0.6904,0.6928,0.695,0.697,0.6988,0.7003,0.7016,0.7027,0.7036,0.7043,0.7047,0.705,0.705,0.7047,0.7043,0.7036,0.7027,0.7016,0.7003,0.6988,0.697,0.695,0.6928,0.6904,0.6877,0.6849,0.6818,0.6786,0.6751,0.6714,0.6676,0.6635,0.6592,0.6547,0.6501,0.6452,0.6402,0.635,0.6296,0.6241,0.6183,0.6124,0.6064,0.6001,0.5938,0.5872,0.5805,0.5737,0.5668,0.5597,0.5524,0.5451,0.5376,0.53,0.5223,0.5145,0.5066,0.4986,0.4905,0.4823,0.474,0.4657,0.4573,0.4488,0.4402,0.4317,0.423,0.4143,0.4056,0.3969,0.3881,0.3793,0.3705,0.3616,0.3528,0.344,0.3351,0.3263,0.3175,0.3088,0.3,0.2913,0.2827,0.274,0.2655,0.257,0.2485,0.2401,0.2318,0.2236,0.2155,0.2074,0.1995,0.1916,0.1838,0.1762,0.1687,0.1612,0.1539,0.1468,0.1398,0.1329,0.1261,0.1195,0.113,0.1067,0.1006,0.0946,0.0888,0.0831,0.0777,0.0724,0.0672,0.0623,0.0576,0.053,0.0486,0.0445,0.0405,0.0367,0.0331,0.0298,0.0266,0.0237,0.0209,0.0184,0.0161,0.014,0.0121,0.0104,0.009,0.0078,0.0068,0.006,0.0054,0.0051,0.005])
sine_angle_torque_2kg = np.asarray([0.02,0.0203,0.0211,0.0225,0.0245,0.027,0.0301,0.0337,0.0379,0.0426,0.0479,0.0537,0.06,0.0669,0.0743,0.0823,0.0908,0.0997,0.1092,0.1192,0.1297,0.1407,0.1522,0.1641,0.1765,0.1894,0.2027,0.2165,0.2307,0.2453,0.2603,0.2758,0.2916,0.3079,0.3245,0.3415,0.3588,0.3765,0.3945,0.4128,0.4315,0.4504,0.4697,0.4892,0.5089,0.5289,0.5492,0.5697,0.5903,0.6112,0.6323,0.6535,0.6749,0.6965,0.7181,0.7399,0.7618,0.7838,0.8058,0.8279,0.8501,0.8722,0.8944,0.9167,0.9388,0.961,0.9831,1.0052,1.0272,1.0492,1.071,1.0927,1.1143,1.1358,1.1571,1.1783,1.1992,1.22,1.2406,1.261,1.2811,1.301,1.3206,1.34,1.3591,1.3779,1.3964,1.4146,1.4324,1.4499,1.4671,1.4839,1.5003,1.5163,1.532,1.5472,1.5621,1.5765,1.5905,1.604,1.6171,1.6298,1.6419,1.6536,1.6649,1.6756,1.6858,1.6956,1.7048,1.7135,1.7217,1.7294,1.7366,1.7432,1.7493,1.7548,1.7598,1.7643,1.7682,1.7715,1.7743,1.7766,1.7782,1.7794,1.7799,1.7799,1.7794,1.7782,1.7766,1.7743,1.7715,1.7682,1.7643,1.7598,1.7548,1.7493,1.7432,1.7366,1.7294,1.7217,1.7135,1.7048,1.6956,1.6858,1.6756,1.6649,1.6536,1.6419,1.6298,1.6171,1.604,1.5905,1.5765,1.5621,1.5472,1.532,1.5163,1.5003,1.4839,1.4671,1.4499,1.4324,1.4146,1.3964,1.3779,1.3591,1.34,1.3206,1.301,1.2811,1.261,1.2406,1.22,1.1992,1.1783,1.1571,1.1358,1.1143,1.0927,1.071,1.0492,1.0272,1.0052,0.9831,0.961,0.9388,0.9167,0.8944,0.8722,0.8501,0.8279,0.8058,0.7838,0.7618,0.7399,0.7181,0.6965,0.6749,0.6535,0.6323,0.6112,0.5903,0.5697,0.5492,0.5289,0.5089,0.4892,0.4697,0.4504,0.4315,0.4128,0.3945,0.3765,0.3588,0.3415,0.3245,0.3079,0.2916,0.2758,0.2603,0.2453,0.2307,0.2165,0.2027,0.1894,0.1765,0.1641,0.1522,0.1407,0.1297,0.1192,0.1092,0.0997,0.0908,0.0823,0.0743,0.0669,0.06,0.0537,0.0479,0.0426,0.0379,0.0337,0.0301,0.027,0.0245,0.0225,0.0211,0.0203,0.02])
wave_1 = generate_sine_wave()
x = np.linspace(0, 1, 250)
f = interpolate.interp1d(x, wave_1)
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
force_gauge = forceGauge("COM16")
stm32 = serial.Serial('COM23', 115200)
print(f"Successfull Open COM7 and COM9")
angle = angle_encoder.get_angle()
force = force_gauge.get_force()
torque_bias = force_gauge.get_torque(angle,force)
for i in range(20):
    angle = angle_encoder.get_angle()
    force = force_gauge.get_force()
    torque_bias = (torque_bias+force_gauge.get_torque(angle,force))/2
    print("torque_bias:",torque_bias)
desire_angle = sine_angle_torque_2kg[0]
actual_angle = angle


# FNN 初始化
up = Torque_ANFIS_multi_5fuzzy()
down = Torque_ANFIS_multi_5fuzzy()
# up.change_learning_rate(0.5)


# up.change_learning_rate(0.1)
# down.change_learning_rate(0.1)


# down.mf_torque.assign(np.asarray([0.2, 0.39, 0.58]))
# down.mf_torque_predict.assign(np.asarray([0.2, 0.39, 0.58]))

# up = Torque_ANFIS_2kg_multi()
# down = Torque_ANFIS_2kg_multi()
# down.mf_torque.assign(np.asarray([0.43,0.935, 1.44]))
# down.mf_torque_predict.assign(np.asarray([0.43,0.935, 1.44]))

# down.mf_torque_predict.assign(np.asarray([0.5, 1.05, 1.6]))
# down.mf_torque.assign(np.asarray([0.5, 1.05, 1.6]))

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
torque_his = np.asarray([])
up_model_his = np.asarray([])
down_model_his = np.asarray([])
force_his = np.asarray([])
angle_his = np.asarray([])
h_his = np.asarray([])

# 主程序函数：用于数据采集与计算控制
def control_loop():
    global h_train,train_torque,train_desire_torque,torque_his, voltage_his, force_his, angle_his, up_model_his, down_model_his
    target_period = period_10
    period = len(target_period)/25
    cycle = 0
    total_cycle = (50 / period) + 1

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
            idx = 0
            print(cycle)
            if cycle == 1:
                up_rmse_his = calculate_rmse(moving_average_with_padding(torque_his[-len(target_period)+1:-int(len(target_period)-len(target_period)/2)+1]),target_period[:int(len(target_period)/2)])
                down_rmse_his = calculate_rmse(moving_average_with_padding(torque_his[-int(len(target_period)/2):]),target_period[int(len(target_period)/2):])
            else:
                up_rmse = calculate_rmse(moving_average_with_padding(torque_his[-len(target_period)+1:-int(len(target_period)-len(target_period)/2)+1]),target_period[:int(len(target_period)/2)])
                down_rmse = calculate_rmse(moving_average_with_padding(torque_his[-int(len(target_period)/2):]),target_period[int(len(target_period)/2):])
                if up_rmse - up_rmse_his <= 0:
                    value = up.return_learning_rate() * 0.9
                    up.change_learning_rate(value)
                    up_rmse_his = up_rmse
                else:
                    value = up.return_learning_rate() * 1.1
                    up.change_learning_rate(value)
                    up_rmse_his = up_rmse
                if down_rmse - down_rmse_his <= 0:
                    value = down.return_learning_rate() * 0.9
                    down.change_learning_rate(value)
                    down_rmse_his = down_rmse
                else:
                    value = down.return_learning_rate() * 1.1
                    down.change_learning_rate(value)
                    down_rmse_his = down_rmse
            cycle += 1

        # up_model_his = np.append(up_model_his, up.return_model())
        # down_model_his = np.append(down_model_his, down.return_model())

        # 目標路徑
        if cycle == 0:
            desire_torque = target_period[0]
        else:
            desire_torque = target_period[idx]
        
        # 更新滤波后的力矩
        actual_angle = angle_encoder.get_angle()
        # h = angle_encoder.get_expansion(actual_angle)
        force = force_gauge.get_force()
        torque = force_gauge.get_torque(actual_angle,force) - torque_bias
        torque_his = np.append(torque_his, torque)
        angle_his = np.append(angle_his, actual_angle)
        if len(torque_his) >=25:
            f_torque = moving_average_with_padding(torque_his[-5:])[-1]
            f_angle = moving_average_with_padding(angle_his[-5:])[-1]
            # f_torque_his = moving_average_with_padding(torque_his[-25:])[-2]
        else:
            f_torque = torque
            f_angle = actual_angle
        
        # 计算误差
        error = desire_torque - f_torque
        # print(error)
        
        # 控制器计算
        if idx < len(target_period) / 2:
            delta_u = up.predict([f_angle], [error])
        else:
            delta_u = down.predict([f_angle], [error])
        
        delta_u *= ku
        controller_u += float(delta_u)
        # voltage_his = np.append(voltage_his, controller_u)
        # f_controller_u = moving_average_with_padding(voltage_his[-25:])[-1]
        test =  np.concatenate((voltage_his[-5:],np.array([controller_u])))
        f_controller_u = moving_average_with_padding(test)[-1]
        # voltage_his = np.append(voltage_his, controller_u)
        # f_controller_u = moving_average_with_padding(voltage_his[-25:])[-1]
        controller_u_output = int((f_controller_u / 10) * 65535)
        # controller_u_output = int((controller_u / 10) * 65535)
        controller_u_output = max(0, controller_u_output)
        
        # 发送控制信号到 STM32
        stm32.write(controller_u_output.to_bytes(2, byteorder='big'))
        if actual_angle > 100 or force > 110:

            _,_ = go_to_desire_angle(angle_encoder, stm32, 30, controller_u)
            break

        # 判断是否触发训练
        if cycle > 1 and idx < len(target_period) / 2 and not train_event_up.is_set():
            # print("ok_up")
            up.update_parameter()
            up_model_his = np.append(up_model_his, up.return_model())
            # train_torque = torque_his[-int(len(target_period) / 2):] 
            # train_desire_torque = target_period[-int(len(target_period) / 2):]
            train_torque = f_torque
            train_desire_torque = target_period[idx]
            h_train = f_angle
            train_event_up.set()  # 触发训练

        elif cycle > 1 and idx > len(target_period) / 2 and not train_event_down.is_set():
            down.update_parameter()
            down_model_his = np.append(down_model_his, down.return_model())
            # print("ok_down")
            # train_torque = torque_his[-int(len(target_period) / 2):]
            # train_desire_torque = target_period[:-int(len(target_period) / 2)]
            train_torque = f_torque
            train_desire_torque = target_period[idx]
            h_train = f_angle
            train_event_down.set() 
        
        # 更新周期和索引
        idx += 1
        voltage_his = np.append(voltage_his, f_controller_u)
    # up_model_his = np.append(up_model_his, up.return_model())
    # down_model_his = np.append(down_model_his, down.return_model())
        force_his = np.append(force_his, force)
        # h_his = np.append(h_his, h)
    save_path_voltage = os.path.join(fileDir, '1', 'voltage.npy')
    save_path_torque = os.path.join(fileDir, '1', 'torque.npy')
    save_path_up_model = os.path.join(fileDir, '1', 'up_model.npy')
    save_path_down_model = os.path.join(fileDir, '1', 'down_model.npy')
    save_path_force = os.path.join(fileDir, '1', 'force.npy')
    save_path_angle = os.path.join(fileDir, '1', 'angle.npy')
    save_path_h = os.path.join(fileDir, '1', 'h.npy')

    np.save(save_path_voltage, voltage_his)
    np.save(save_path_torque, torque_his)
    np.save(save_path_up_model, up_model_his)
    np.save(save_path_down_model, down_model_his)
    np.save(save_path_force, force_his)
    np.save(save_path_angle, angle_his)
    np.save(save_path_h, h_his)


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
        error = local_desire_torque - local_torque
        up.train([h_local], [error], [local_desire_torque], [local_torque])
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
        error = local_desire_torque - local_torque
        down.train([h_local], [error], [local_desire_torque], [local_torque])
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
############ 5rule +ran波

# import time
# import threading
# import numpy as np
# import os, shutil
# from datetime import datetime
# from kerasFuzzy import Torque_ANFIS_multi_5fuzzy, go_to_desire_angle,Torque_ANFIS_2kg_multi
# from encoder_function import encoder, forceGauge, moving_average_with_padding,calculate_rmse
# import serial
# from scipy import interpolate
# from generate_sine_wave import generate_sine_wave


# # 初始化全局参数
# # sine_angle_torque_1kg = np.asarray([0.005,0.0051,0.0054,0.006,0.0068,0.0078,0.009,0.0104,0.0121,0.014,0.0161,0.0184,0.0209,0.0237,0.0266,0.0298,0.0331,0.0367,0.0405,0.0445,0.0486,0.053,0.0576,0.0623,0.0672,0.0724,0.0777,0.0831,0.0888,0.0946,0.1006,0.1067,0.113,0.1195,0.1261,0.1329,0.1398,0.1468,0.1539,0.1612,0.1687,0.1762,0.1838,0.1916,0.1995,0.2074,0.2155,0.2236,0.2318,0.2401,0.2485,0.257,0.2655,0.274,0.2827,0.2913,0.3,0.3088,0.3175,0.3263,0.3351,0.344,0.3528,0.3616,0.3705,0.3793,0.3881,0.3969,0.4056,0.4143,0.423,0.4317,0.4402,0.4488,0.4573,0.4657,0.474,0.4823,0.4905,0.4986,0.5066,0.5145,0.5223,0.53,0.5376,0.5451,0.5524,0.5597,0.5668,0.5737,0.5805,0.5872,0.5938,0.6001,0.6064,0.6124,0.6183,0.6241,0.6296,0.635,0.6402,0.6452,0.6501,0.6547,0.6592,0.6635,0.6676,0.6714,0.6751,0.6786,0.6818,0.6849,0.6877,0.6904,0.6928,0.695,0.697,0.6988,0.7003,0.7016,0.7027,0.7036,0.7043,0.7047,0.705,0.705,0.7047,0.7043,0.7036,0.7027,0.7016,0.7003,0.6988,0.697,0.695,0.6928,0.6904,0.6877,0.6849,0.6818,0.6786,0.6751,0.6714,0.6676,0.6635,0.6592,0.6547,0.6501,0.6452,0.6402,0.635,0.6296,0.6241,0.6183,0.6124,0.6064,0.6001,0.5938,0.5872,0.5805,0.5737,0.5668,0.5597,0.5524,0.5451,0.5376,0.53,0.5223,0.5145,0.5066,0.4986,0.4905,0.4823,0.474,0.4657,0.4573,0.4488,0.4402,0.4317,0.423,0.4143,0.4056,0.3969,0.3881,0.3793,0.3705,0.3616,0.3528,0.344,0.3351,0.3263,0.3175,0.3088,0.3,0.2913,0.2827,0.274,0.2655,0.257,0.2485,0.2401,0.2318,0.2236,0.2155,0.2074,0.1995,0.1916,0.1838,0.1762,0.1687,0.1612,0.1539,0.1468,0.1398,0.1329,0.1261,0.1195,0.113,0.1067,0.1006,0.0946,0.0888,0.0831,0.0777,0.0724,0.0672,0.0623,0.0576,0.053,0.0486,0.0445,0.0405,0.0367,0.0331,0.0298,0.0266,0.0237,0.0209,0.0184,0.0161,0.014,0.0121,0.0104,0.009,0.0078,0.0068,0.006,0.0054,0.0051,0.005])
# sine_angle_torque_2kg = np.asarray([0.02,0.0203,0.0211,0.0225,0.0245,0.027,0.0301,0.0337,0.0379,0.0426,0.0479,0.0537,0.06,0.0669,0.0743,0.0823,0.0908,0.0997,0.1092,0.1192,0.1297,0.1407,0.1522,0.1641,0.1765,0.1894,0.2027,0.2165,0.2307,0.2453,0.2603,0.2758,0.2916,0.3079,0.3245,0.3415,0.3588,0.3765,0.3945,0.4128,0.4315,0.4504,0.4697,0.4892,0.5089,0.5289,0.5492,0.5697,0.5903,0.6112,0.6323,0.6535,0.6749,0.6965,0.7181,0.7399,0.7618,0.7838,0.8058,0.8279,0.8501,0.8722,0.8944,0.9167,0.9388,0.961,0.9831,1.0052,1.0272,1.0492,1.071,1.0927,1.1143,1.1358,1.1571,1.1783,1.1992,1.22,1.2406,1.261,1.2811,1.301,1.3206,1.34,1.3591,1.3779,1.3964,1.4146,1.4324,1.4499,1.4671,1.4839,1.5003,1.5163,1.532,1.5472,1.5621,1.5765,1.5905,1.604,1.6171,1.6298,1.6419,1.6536,1.6649,1.6756,1.6858,1.6956,1.7048,1.7135,1.7217,1.7294,1.7366,1.7432,1.7493,1.7548,1.7598,1.7643,1.7682,1.7715,1.7743,1.7766,1.7782,1.7794,1.7799,1.7799,1.7794,1.7782,1.7766,1.7743,1.7715,1.7682,1.7643,1.7598,1.7548,1.7493,1.7432,1.7366,1.7294,1.7217,1.7135,1.7048,1.6956,1.6858,1.6756,1.6649,1.6536,1.6419,1.6298,1.6171,1.604,1.5905,1.5765,1.5621,1.5472,1.532,1.5163,1.5003,1.4839,1.4671,1.4499,1.4324,1.4146,1.3964,1.3779,1.3591,1.34,1.3206,1.301,1.2811,1.261,1.2406,1.22,1.1992,1.1783,1.1571,1.1358,1.1143,1.0927,1.071,1.0492,1.0272,1.0052,0.9831,0.961,0.9388,0.9167,0.8944,0.8722,0.8501,0.8279,0.8058,0.7838,0.7618,0.7399,0.7181,0.6965,0.6749,0.6535,0.6323,0.6112,0.5903,0.5697,0.5492,0.5289,0.5089,0.4892,0.4697,0.4504,0.4315,0.4128,0.3945,0.3765,0.3588,0.3415,0.3245,0.3079,0.2916,0.2758,0.2603,0.2453,0.2307,0.2165,0.2027,0.1894,0.1765,0.1641,0.1522,0.1407,0.1297,0.1192,0.1092,0.0997,0.0908,0.0823,0.0743,0.0669,0.06,0.0537,0.0479,0.0426,0.0379,0.0337,0.0301,0.027,0.0245,0.0225,0.0211,0.0203,0.02])
# wave_1 = generate_sine_wave()
# wave_2 = generate_sine_wave(amplitude=0.45, num_points=250)
# wave_3 = generate_sine_wave(num_points=180)
# wave_4 = generate_sine_wave(amplitude=0.6, num_points=180)
# period=  np.concatenate((wave_1, wave_2,wave_3,wave_4))

# # x = np.linspace(0, 1, 250)
# # f = interpolate.interp1d(x, sine_angle_torque_2kg)
# # x_10 = np.linspace(0, 1, 250)
# # x_8 = np.linspace(0, 1, 200)
# # x_6 = np.linspace(0, 1, 150)
# # x_4 = np.linspace(0, 1, 100)
# # period_10 = f(x_10)
# # period_8 = f(x_8)
# # period_6 = f(x_6)
# # period_4 = f(x_4)

# # Event 控制信号
# train_event_up = threading.Event()
# train_event_down = threading.Event()
# stop_event = threading.Event()

# # 角度计和传感器初始化
# angle_encoder = encoder()
# angle_encoder.get_com_port('COM7')
# force_gauge = forceGauge("COM16")
# stm32 = serial.Serial('COM23', 115200)
# print(f"Successfull Open COM7 and COM9")
# angle = angle_encoder.get_angle()
# force = force_gauge.get_force()
# torque_bias = force_gauge.get_torque(angle,force)
# for i in range(20):
#     angle = angle_encoder.get_angle()
#     force = force_gauge.get_force()
#     torque_bias = (torque_bias+force_gauge.get_torque(angle,force))/2
#     print("torque_bias:",torque_bias)
# desire_angle = sine_angle_torque_2kg[0]
# actual_angle = angle


# # FNN 初始化
# up = Torque_ANFIS_multi_5fuzzy()
# down = Torque_ANFIS_multi_5fuzzy()

# file_path = './exp/2024_10_11_1349/1/up_model.npy' #2kg 5rule
# loaded_data = np.load(file_path)
# file_path_2 = './exp/2024_10_11_1349/1/down_model.npy' #2kg 5rule
# loaded_data_2 = np.load(file_path_2)
# up.load_model(loaded_data[-42:])
# down.load_model(loaded_data_2[-42:])
# # up.change_learning_rate(0.5)


# # up.change_learning_rate(0.1)
# # down.change_learning_rate(0.1)


# # down.mf_torque.assign(np.asarray([0.2, 0.39, 0.58]))
# # down.mf_torque_predict.assign(np.asarray([0.2, 0.39, 0.58]))

# # up = Torque_ANFIS_2kg_multi()
# # down = Torque_ANFIS_2kg_multi()
# # down.mf_torque.assign(np.asarray([0.43,0.935, 1.44]))
# # down.mf_torque_predict.assign(np.asarray([0.43,0.935, 1.44]))

# # down.mf_torque_predict.assign(np.asarray([0.5, 1.05, 1.6]))
# # down.mf_torque.assign(np.asarray([0.5, 1.05, 1.6]))

# # 文件夹创建
# ts = time.time()
# data_time = datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H%M')
# fileDir = './exp/{}'.format(data_time)
# if not os.path.isdir(fileDir):
#     os.makedirs(os.path.join(fileDir, '1'))
# else:
#     shutil.rmtree(fileDir)
#     os.makedirs(os.path.join(fileDir, '1'))

# voltage_his = np.asarray([])
# torque_his = np.asarray([])
# up_model_his = np.asarray([])
# down_model_his = np.asarray([])
# force_his = np.asarray([])
# angle_his = np.asarray([])
# h_his = np.asarray([])

# # 主程序函数：用于数据采集与计算控制
# def control_loop():
#     global period,h_train,train_torque,train_desire_torque,torque_his, voltage_his, force_his, angle_his, up_model_his, down_model_his
#     target_period = period
#     period = len(target_period)/25
#     cycle = 0
#     total_cycle = 2

#     idx = 0

#     ku = 0.01
#     controller_u = 0.5
#     controller_u_output = 0
#     error = 0
    
#     while cycle < total_cycle:

#         if cycle == 0 and idx == 100:
#             idx = 0
#             cycle += 1
#         # 更新角度和力矩
#         if cycle != 0 and idx == len(target_period):
#             idx = 0
#             print(cycle)
#             cycle += 1

#         # up_model_his = np.append(up_model_his, up.return_model())
#         # down_model_his = np.append(down_model_his, down.return_model())

#         # 目標路徑
#         if cycle == 0:
#             desire_torque = target_period[0]
#         else:
#             desire_torque = target_period[idx]
        
#         # 更新滤波后的力矩
#         actual_angle = angle_encoder.get_angle()
#         # h = angle_encoder.get_expansion(actual_angle)
#         force = force_gauge.get_force()
#         torque = force_gauge.get_torque(actual_angle,force) - torque_bias
#         torque_his = np.append(torque_his, torque)
#         angle_his = np.append(angle_his, actual_angle)
#         if len(torque_his) >=25:
#             f_torque = moving_average_with_padding(torque_his[-5:])[-1]
#             f_angle = moving_average_with_padding(angle_his[-5:])[-1]
#             # f_torque_his = moving_average_with_padding(torque_his[-25:])[-2]
#         else:
#             f_torque = torque
#             f_angle = actual_angle
        
#         # 计算误差
#         error = desire_torque - f_torque
#         # print(error)
        
#         # 控制器计算
#         if idx < len(target_period) / 2:
#             delta_u = up.predict([f_angle], [error])
#         else:
#             delta_u = down.predict([f_angle], [error])
        
#         delta_u *= ku
#         controller_u += float(delta_u)
#         test =  np.concatenate((voltage_his[-5:],np.array([controller_u])))
#         f_controller_u = moving_average_with_padding(test)[-1]
#         # voltage_his = np.append(voltage_his, controller_u)
#         # f_controller_u = moving_average_with_padding(voltage_his[-25:])[-1]
#         controller_u_output = int((f_controller_u / 10) * 65535)
#         controller_u_output = max(0, controller_u_output)
        
#         # 发送控制信号到 STM32
#         stm32.write(controller_u_output.to_bytes(2, byteorder='big'))
#         if actual_angle > 100 or force > 110:

#             _,_ = go_to_desire_angle(angle_encoder, stm32, 30, controller_u)
#             break

#         # 判断是否触发训练
#         if cycle > 1 and idx < len(target_period) / 2 and not train_event_up.is_set():
#             # print("ok_up")
#             # up.update_parameter()
#             up_model_his = np.append(up_model_his, up.return_model())
#             # train_torque = torque_his[-int(len(target_period) / 2):] 
#             # train_desire_torque = target_period[-int(len(target_period) / 2):]
#             train_torque = f_torque
#             train_desire_torque = target_period[idx]
#             h_train = f_angle
#             train_event_up.set()  # 触发训练

#         elif cycle > 1 and idx > len(target_period) / 2 and not train_event_down.is_set():
#             # down.update_parameter()
#             down_model_his = np.append(down_model_his, down.return_model())
#             # print("ok_down")
#             # train_torque = torque_his[-int(len(target_period) / 2):]
#             # train_desire_torque = target_period[:-int(len(target_period) / 2)]
#             train_torque = f_torque
#             train_desire_torque = target_period[idx]
#             h_train = f_angle
#             train_event_down.set() 
        
#         # 更新周期和索引
#         idx += 1
#         voltage_his = np.append(voltage_his, f_controller_u)
#     # up_model_his = np.append(up_model_his, up.return_model())
#     # down_model_his = np.append(down_model_his, down.return_model())
#         force_his = np.append(force_his, force)
#         # h_his = np.append(h_his, h)
#     save_path_voltage = os.path.join(fileDir, '1', 'voltage.npy')
#     save_path_torque = os.path.join(fileDir, '1', 'torque.npy')
#     save_path_up_model = os.path.join(fileDir, '1', 'up_model.npy')
#     save_path_down_model = os.path.join(fileDir, '1', 'down_model.npy')
#     save_path_force = os.path.join(fileDir, '1', 'force.npy')
#     save_path_angle = os.path.join(fileDir, '1', 'angle.npy')
#     save_path_h = os.path.join(fileDir, '1', 'h.npy')

#     np.save(save_path_voltage, voltage_his)
#     np.save(save_path_torque, torque_his)
#     np.save(save_path_up_model, up_model_his)
#     np.save(save_path_down_model, down_model_his)
#     np.save(save_path_force, force_his)
#     np.save(save_path_angle, angle_his)
#     np.save(save_path_h, h_his)


# # FNN 模型训练函数
# def fnn_train_up():
#     global train_torque,train_desire_torque,h_train
#     while not stop_event.is_set():  # 检查是否有停止信号
#         train_event_up.wait()  # 等待控制信号
#         if stop_event.is_set():
#             break  # 如果收到停止信号，退出循环
#         local_torque = train_torque
#         local_desire_torque = train_desire_torque
#         h_local = h_train
#         error = local_desire_torque - local_torque
#         up.train([h_local], [error], [local_desire_torque], [local_torque])
#         # f_torque = moving_average_with_padding(local_torque)
#         # # error = local_desire_torque - f_torque
#         # # f_torque_1 =  [[value] for value in f_torque[:-1]]
#         # # f_torque_2 =  [[value] for value in f_torque[1:]]
#         # # local_desire_torque = [[value] for value in local_desire_torque[:-1]]
#         # # error = [[value] for value in error[:-1]]
#         # # 训练模型 up
#         # for i in range(len(local_torque)-1):
#         #     f_torque_1 = f_torque[i]
#         #     error = local_desire_torque[i] - f_torque[i]
#         #     f_torque_2 = f_torque[i]
#         #     up.train([f_torque_1], [error], [local_desire_torque], [f_torque_2])
#         train_event_up.clear()  # 训练完成后清除信号

# def fnn_train_down():
#     global train_torque,train_desire_torque,h_train
#     while not stop_event.is_set():  # 检查是否有停止信号
#         train_event_down.wait()  # 等待控制信号
#         if stop_event.is_set():
#             break  # 如果收到停止信号，退出循环
#         local_torque = train_torque
#         local_desire_torque = train_desire_torque
#         h_local = h_train
#         error = local_desire_torque - local_torque
#         down.train([h_local], [error], [local_desire_torque], [local_torque])
#         # f_torque = moving_average_with_padding(local_torque)
#         # # error = local_desire_torque - f_torque
#         # # f_torque_1 =  [[value] for value in f_torque[:-1]]
#         # # f_torque_2 =  [[value] for value in f_torque[1:]]
#         # # local_desire_torque = [[value] for value in local_desire_torque[:-1]]
#         # # error = [[value] for value in error[:-1]]
#         # # 训练模型 up
#         # for i in range(len(local_torque)-1):
#         #     f_torque_1 = f_torque[i]
#         #     error = local_desire_torque[i] - f_torque[i]
#         #     f_torque_2 = f_torque[i]
#         #     down.train([f_torque_1], [error], [local_desire_torque], [f_torque_2])
#         train_event_down.clear()  # 训练完成后清除信号

# # 启动线程
# control_thread = threading.Thread(target=control_loop)
# # train_up_thread = threading.Thread(target=fnn_train_up)
# # train_down_thread = threading.Thread(target=fnn_train_down)

# # 开始线程执行
# control_thread.start()
# # train_up_thread.start()
# # train_down_thread.start()

# # 主线程等待control_thread结束
# control_thread.join()

# # 结束训练线程
# stop_event.set()  # 发送停止信号
# train_event_up.set()  # 防止线程阻塞在 wait 状态
# train_event_down.set()  # 防止线程阻塞在 wait 状态

# # # 等待训练线程结束
# # train_up_thread.join()
# # train_down_thread.join()
# # # 保存记录
