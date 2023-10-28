from control_system_detail import control_system,Control,return_simulation_pma_angle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

excel_file = pd.ExcelFile('PMA_angle.xlsx')
df_pma_angle = excel_file.parse('Sheet1', usecols="B:C", header=None,nrows=200)
triangle_angle = [20,21,23,24,26,27,28,30,31,33,34,36,37,38,40,41,43,44,45,47,48,50,51,53,54,55,57,58,60,61,62,64,65,67,68,69,71,72,74,75,77,78,79,81,82,84,85,86,88,89,89,88,86,85,84,82,81,79,78,77,75,74,72,71,69,68,67,65,64,62,61,60,58,57,55,54,53,51,50,48,47,45,44,43,41,40,38,37,36,34,33,31,30,28,27,26,24,23,21,20]
simulation = True
mode = 1

# parameter_history = []
label_smc_lambda = np.array([])
label_k_l1 = np.array([])
label_k_l2 = np.array([])
parameter_history_overshoot = np.array([])
parameter_history_risetime = np.array([])
parameter_history_settingtime = np.array([])

C = Control()

# smc_lambda = 0.2    # 0.2 越快到滑膜面
# k_l1 = 0.02          # 0.5
# k_l2 = 0.01          # 0.1 趨近速度增加，抖振增

Idx = 0
test = 0
check_rise_time = False
check_setting_time = False
rise_time = 0

setting_time = [0,0]


desire_angle = 20
actual_angle = 20


learning_array = [0] * 100
first_period = True
controller_u = 0
last_max_angle = 0
last_max_angle_time = 0
        
first_count = 0
reset_count = 0

desire_angle_history = []
actual_angle_history = []

i = 0

def find_overshoot(actual_angle,last_max_angle,test,last_max_angle_time):
    if  last_max_angle < actual_angle:
        last_max_angle = actual_angle
        last_max_angle_time = test
    return last_max_angle,last_max_angle_time

def reset_history(Idx, test, check_rise_time, check_setting_time, rise_time, setting_time, desire_angle,
                    actual_angle, learning_array, first_period, controller_u, last_max_angle,
                    last_max_angle_time, first_count, reset_count, desire_angle_history, actual_angle_history):
    Idx = 0
    test = 0
    check_rise_time = False
    check_setting_time = False
    rise_time = 0

    setting_time = [0,0]

    desire_angle = 20
    actual_angle = 20

    learning_array = [0] * 100
    first_period = True
    controller_u = 0
    last_max_angle = 0
    last_max_angle_time = 0
        
    first_count = 0
    reset_count = 0

    desire_angle_history = []
    actual_angle_history = []
    return Idx, test, check_rise_time, check_setting_time, rise_time, setting_time, desire_angle, actual_angle, learning_array, first_period, controller_u, last_max_angle,last_max_angle_time, first_count, reset_count, desire_angle_history, actual_angle_history

for m in np.arange(0,1,0.05):
    for n in np.arange(0,1,0.05):
        for l in np.arange(0,1,0.05):
            smc_lambda = m    # 0.2 越快到滑膜面
            k_l1 = n          # 0.5
            k_l2 = l          # 0.1 趨近速度增加，抖振增

            while True:
                parameter = np.array([0,0,0])
                Idx = Idx + 1
                if Idx == 100:
                    Idx = 0 
                test = test + 1

            ####################### 目標路徑 #######################
                desire_angle = 90
            #######################################################

                # 控制系統
                controller_u, learning_array, first_period, C = control_system(controller_u,desire_angle,actual_angle,learning_array,Idx,first_period, C, smc_lambda, k_l1, k_l2)

                if simulation == True:
                    actual_angle = return_simulation_pma_angle(df_pma_angle,int(controller_u/10*65535),actual_angle)
                
                desire_angle_history.append(desire_angle)
                actual_angle_history.append(actual_angle)

                last_max_angle,last_max_angle_time = find_overshoot(actual_angle,last_max_angle,test,last_max_angle_time)
                if last_max_angle > 130 :
                    last_max_angle = -1
                if actual_angle > 81 and check_rise_time == False:
                    rise_time = test
                    check_rise_time = True
                # 安定時間
                if check_setting_time == False and actual_angle < 92.7 and actual_angle > 87.3:
                    setting_time[0] = test
                    setting_time[1] = actual_angle
                    check_setting_time = True
                if check_setting_time == True and (actual_angle > 92.7 or actual_angle < 87.3):
                    check_setting_time = False
                if test > 130 and check_setting_time == False:
                    setting_time[1] = -100
                    label_smc_lambda = np.append(m,label_smc_lambda)
                    label_k_l1 = np.append(n,label_k_l1)
                    label_k_l2 = np.append(l,label_k_l2)
                    parameter_history_overshoot = np.append(last_max_angle,parameter_history_overshoot)
                    parameter_history_risetime = np.append(rise_time-1,parameter_history_risetime)
                    parameter_history_settingtime = np.append(setting_time[0]-1,parameter_history_settingtime)
                    # parameter_history[0][0][i] = last_max_angle
                    # parameter_history[0][1][i] = rise_time-1
                    # parameter_history[0][2][i] = setting_time[0]-1
                    # i = i+1
                    break
                elif test > 130 and check_setting_time == True:
                    label_smc_lambda = np.append(m,label_smc_lambda)
                    label_k_l1 = np.append(n,label_k_l1)
                    label_k_l2 = np.append(l,label_k_l2)
                    parameter_history_overshoot = np.append(last_max_angle,parameter_history_overshoot)
                    parameter_history_risetime = np.append(rise_time-1,parameter_history_risetime)
                    parameter_history_settingtime = np.append(setting_time[0]-1,parameter_history_settingtime)
                    # parameter_history[0][0][i] = last_max_angle
                    # parameter_history[0][1][i] = rise_time-1
                    # parameter_history[0][2][i] = setting_time[0]-1
                    # i = i+1
                    break
            Idx, test, check_rise_time, check_setting_time, rise_time, setting_time, desire_angle, actual_angle, learning_array, first_period, controller_u, last_max_angle,last_max_angle_time, first_count, reset_count, desire_angle_history, actual_angle_history = reset_history(Idx, test, check_rise_time, check_setting_time, rise_time, setting_time, desire_angle,
                    actual_angle, learning_array, first_period, controller_u, last_max_angle,
                    last_max_angle_time, first_count, reset_count, desire_angle_history, actual_angle_history)
            

fig1 = plt.figure()
ax = fig1.add_subplot(111, projection='3d')
scatter1 = ax.scatter(label_smc_lambda, label_k_l1, label_k_l2, c=parameter_history_overshoot, marker='o')

# 设置坐标轴标签
ax.set_xlabel('smc_lambda')
ax.set_ylabel('k_l1')
ax.set_zlabel('k_l2')

colorbar1 = plt.colorbar(scatter1, ax=ax, label='overshoot')

fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
scatter2 = ax2.scatter(label_smc_lambda, label_k_l1, label_k_l2, c=parameter_history_risetime, marker='o')

# 设置坐标轴标签
ax2.set_xlabel('smc_lambda')
ax2.set_ylabel('k_l1')
ax2.set_zlabel('k_l2')

colorbar2 = plt.colorbar(scatter2, ax=ax2, label='risetime')

fig3 = plt.figure()
ax3 = fig3.add_subplot(111, projection='3d')
scatter3 = ax3.scatter(label_smc_lambda, label_k_l1, label_k_l2, c=parameter_history_settingtime, marker='o')


# 设置坐标轴标签
ax3.set_xlabel('smc_lambda')
ax3.set_ylabel('k_l1')
ax3.set_zlabel('k_l2')

colorbar3 = plt.colorbar(scatter3, ax=ax3, label='settingtime')

plt.show()


# Idx, test, check_rise_time, check_setting_time, rise_time, setting_time, desire_angle, actual_angle, learning_array, first_period, controller_u, last_max_angle,last_max_angle_time, first_count, reset_count, desire_angle_history, actual_angle_history = reset_history(Idx, test, check_rise_time, check_setting_time, rise_time, setting_time, desire_angle,
#                     actual_angle, learning_array, first_period, controller_u, last_max_angle,
#                     last_max_angle_time, first_count, reset_count, desire_angle_history, actual_angle_history)

# time_axis = range(len(desire_angle_history))

# time_axis = [0] + list(time_axis)
# desire_angle_history = [20] + list(desire_angle_history)
# actual_angle_history = [20] + list(actual_angle_history)

# plt.plot(time_axis, desire_angle_history, label='desire angle', color='blue')
# plt.plot(time_axis, actual_angle_history, label='actual angle', color='green')

# plt.scatter(last_max_angle_time-1, last_max_angle, color='red', marker='o', s=10)
# plt.scatter(rise_time-1, actual_angle_history[rise_time-1], color='red', marker='o', s=10)
# plt.scatter(setting_time[0]-1, setting_time[1], color='red', marker='o', s=10)

# print( actual_angle_history[rise_time-1])
# plt.xlabel('Time')
# plt.ylabel('Angle')
# plt.legend()
# plt.show()
