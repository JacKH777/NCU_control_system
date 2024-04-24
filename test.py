import matplotlib.pyplot as plt
import numpy as np
# 創建一個畫布和兩個軸
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True) 

# # 複製第一個軸並共享 x 軸
# ax2 = ax1.twinx()
x_data = np.linspace(0, 20, 400)
# 繪製電壓變化曲線
color1 = 'tab:blue'
# ax1.set_xlabel('Time (sec)')

ax1.set_ylabel('Voltage (V)')
ax1.plot(x_data,np.zeros(400), color=color1)
ax1.set_xlim(0, 20)
# 繪製角度變化曲線
color2 = 'tab:blue'
ax2.set_xlabel('Time (sec)')
ax2.set_ylabel('Angle (Degree)')
ax2.plot(x_data,np.zeros(400), color=color2)

# 顯示圖表
plt.show()