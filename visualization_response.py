import numpy as np
import matplotlib.pyplot as plt

# # 讀取數據
# # 確保將 'filename.txt' 替換為您的文件名
# # 假設每行有兩個以空格分隔的數字
# degree_his = np.loadtxt('degree_response_learning.txt')
# voltage_his = np.loadtxt('voltage_response_learning.txt')
# degree_his_no = np.loadtxt('degree_response_no_learning.txt')
# voltage_his_no = np.loadtxt('voltage_response_no_learning.txt')

# # 數據點的 x 坐標和 y 坐標
# degree_his = degree_his[::-1]
# voltage_his = voltage_his[::-1]
# degree_his_no = degree_his_no[::-1]
# voltage_his_no = voltage_his_no[::-1]
# degree_desire = np.full(200, 90)

# print('overshoot:',degree_his.max())
# print('rise time:',((np.argmax(degree_his>85))-100)*0.05)
# in_range = (degree_his >= 86) & (degree_his <= 94)
# in_range_inverse = in_range[::-1]

# for i in range(len(in_range_inverse)):
#     if in_range_inverse[i] == False and i !=0:
#         true_index = len(degree_his) - i -1
#         print('setting time:',((true_index)-100)*0.05)
#         break

# fig, ax2 = plt.subplots( 1, sharex=True)  # sharex=True 表示兩個圖共享x軸
# x_data = np.linspace(0, 10, 200)

# # 在第一個軸上繪製第一組數據
# ax2.plot(x_data,degree_his[100:], color='tab:red', label='learning')  # 'r-' 表示紅色的線條
# ax2.plot(x_data,degree_his_no[100:], color='tab:blue', label='without learning')  # 'r-' 表示紅色的線條
# # ax2.set_title('degree')  # 第一個圖的標題
# ax2.set_ylabel('Angle (Degree)')  # Y軸標籤
# ax2.set_xlabel('Time (Sec)')  # 兩個圖表共享X軸，只需在最下面的圖表設置X軸標籤
# ax2.plot(x_data,degree_desire , color='k',linewidth=1)
# ax2.legend(loc='lower right')
# ax2.set_xlim(0, 10)

# # 自動調整子圖參數，以確保子圖不會重疊
# plt.tight_layout()
# # 顯示圖表
# plt.show()

#pic1
voltage_his=np.array([3.2]*100+[0.5]*100+[3.2]*100+[0.5]*100)
degree_his = np.loadtxt('angle_response_cube_piz_1.txt')
degree_his = degree_his+6.5
degree_desire = np.array([90]*100+[32]*100+[90]*100+[32]*100)
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)  # sharex=True 表示兩個圖共享x軸
color1 = 'tab:blue'
x_data = np.linspace(0, 20, 400)
ax1.plot(x_data,voltage_his, color='k')  # 'b-' 表示藍色的線條
# ax1.set_title('Voltage (V)')
ax1.set_ylabel('Voltage (V)')
# 在第一個軸上繪製第一組數據
ax2.plot(x_data,degree_his[100:], color=color1)  # 'r-' 表示紅色的線條
ax2.set_ylabel('Angle (Degree)')  # Y軸標籤
ax2.set_xlabel('Time (Sec)')  # 兩個圖表共享X軸，只需在最下面的圖表設置X軸標籤
ax2.plot(x_data,degree_desire , color='k',linewidth=1,alpha = 0.7)
ax2.legend()  # 添加圖例

ax1.set_xlim(0, 20)
ax1.set_ylim(-0.1, 4)
# 自動調整子圖參數，以確保子圖不會重疊
plt.tight_layout()
# 顯示圖表
plt.show()
