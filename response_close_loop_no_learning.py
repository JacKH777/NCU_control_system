import numpy as np
from decoder_function import decoder
import serial
import time

from keras_fuzzy import ANFIS

ser_1 = serial.Serial("COM6", 115200)
right_hand = decoder()
right_hand.get_com_port('COM4')
test = 0
total_duration = 0.05 
fis = ANFIS()
fis.load_model('model.txt')
ku = 0.01
voltage_his = np.asarray([])
degree_his = np.asarray([])
last_error = 0
controller_u = 0

while test < 300:
    # 记录开始时间
    start_time = time.time()

    test = test + 1

    ####################### 目標路徑 #######################
    if test > 100:
        desire_angle = 90
    else :
        desire_angle = 30
    #######################################################
    
    ######## need
    actual_angle = right_hand.get_angle()
    if actual_angle > 100:
        controller_u_output = 0
        ser_1.write(controller_u_output.to_bytes(2, byteorder='big'))
        break

    error = desire_angle - actual_angle
    delta = (error -  last_error)
    

    # new_u= fis.train([error],[delta], [desire_angle],[actual_angle])
    new_u= fis.predict([error],[delta])
    new_u = new_u * ku
    controller_u = controller_u + new_u
    # print(controller_u)
    last_error = error
    # print(error,delta)
    # print(c,w)
    ##########

    # 轉成 16 bits 電壓值 need
    controller_u = float(controller_u)
    controller_u_output = controller_u/10*65535
    controller_u_output = int(controller_u_output)

    # 儲存結果 need
    voltage_his = np.append(controller_u,voltage_his)
    degree_his = np.append(actual_angle,degree_his)

    if controller_u_output < 0:
        controller_u_output = 0
    
    ser_1.write(controller_u_output.to_bytes(2, byteorder='big'))

    elapsed_time = time.time() - start_time

    # 如果运行时间小于指定的总时间，则补充剩余的时间
    if elapsed_time < total_duration:
        remaining_time = total_duration - elapsed_time
        time.sleep(remaining_time)

np.savetxt('voltage_response_learning.txt', voltage_his, delimiter=',')
np.savetxt('degree_response_learning.txt', degree_his, delimiter=',')
controller_u_output = 0
ser_1.write(controller_u_output.to_bytes(2, byteorder='big'))