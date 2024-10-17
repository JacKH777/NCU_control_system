import numpy as np
import matplotlib.pyplot as plt

# 读取 NumPy 文件
file_path = './combine4To10SecSineWave/force_his/voltage_1kg_noHuman.npy'  # 替换为你的实际文件路径
data = np.load(file_path)

# 绘制数据
plt.plot(data)
plt.title('Plot of Loaded NumPy Array')
plt.xlabel('Index')
plt.ylabel('Value')
plt.grid(True)

# 显示图形
plt.show()
