from control_system_detail import control_system,Control,return_simulation_pma_angle
import pandas as pd
import matplotlib.pyplot as plt

excel_file = pd.ExcelFile('PMA_angle.xlsx')
df_pma_angle = excel_file.parse('Sheet1', usecols="B:C", header=None,nrows=200)
triangle_angle = [20,21,23,24,26,27,28,30,31,33,34,36,37,38,40,41,43,44,45,47,48,50,51,53,54,55,57,58,60,61,62,64,65,67,68,69,71,72,74,75,77,78,79,81,82,84,85,86,88,89,89,88,86,85,84,82,81,79,78,77,75,74,72,71,69,68,67,65,64,62,61,60,58,57,55,54,53,51,50,48,47,45,44,43,41,40,38,37,36,34,33,31,30,28,27,26,24,23,21,20]
simulation = True
mode = 1

C = Control()

Idx = 0
test = 0
check_rise_time = False
check_setting_time = False
rise_time = 0

setting_time = [0,0]

desire_angle = triangle_angle[0]
actual_angle = triangle_angle[0]

learning_array = [0] * 100
first_period = True
controller_u = 0
last_max_angle = 0
last_max_angle_time = 0
        
first_count = 0
reset_count = 0

desire_angle_history = []
actual_angle_history = []

def find_overshoot(actual_angle,last_max_angle,test,last_max_angle_time):
    if  last_max_angle < actual_angle:
        last_max_angle = actual_angle
        last_max_angle_time = test
    return last_max_angle,last_max_angle_time

while True:
    Idx = Idx + 1
    if Idx == 100:
        Idx = 0 
    test = test + 1

####################### 目標路徑 #######################
    # desire_angle = triangle_angle[Idx]
    # if mode == 1:
    #     if desire_angle > 55 or test > 60:
    #         desire_angle = 90
    #     else :
    #         desire_angle = 20
    desire_angle = 90
#######################################################

    # 控制系統
    controller_u, learning_array, first_period, C = control_system(controller_u,desire_angle,actual_angle,learning_array,Idx,first_period, C)

    if simulation == True:
        actual_angle = return_simulation_pma_angle(df_pma_angle,int(controller_u/10*65535),actual_angle)
    
    desire_angle_history.append(desire_angle)
    actual_angle_history.append(actual_angle)

    last_max_angle,last_max_angle_time = find_overshoot(actual_angle,last_max_angle,test,last_max_angle_time)
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
        setting_time[1] = -1
        break
    elif test > 130 and check_setting_time == True:
        break

time_axis = range(len(desire_angle_history))

time_axis = [0] + list(time_axis)
desire_angle_history = [20] + list(desire_angle_history)
actual_angle_history = [20] + list(actual_angle_history)

plt.plot(time_axis, desire_angle_history, label='desire angle', color='blue')
plt.plot(time_axis, actual_angle_history, label='actual angle', color='green')

plt.scatter(last_max_angle_time-1, last_max_angle, color='red', marker='o', s=10)
plt.scatter(rise_time-1, actual_angle_history[rise_time], color='red', marker='o', s=10)
plt.scatter(setting_time[0]-1, setting_time[1], color='red', marker='o', s=10)

print(setting_time[1])
plt.xlabel('Time')
plt.ylabel('Angle')
plt.legend()
plt.show()
