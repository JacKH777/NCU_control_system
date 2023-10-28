import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 创建示例数据，包括 x、y、z 和 m 值
x = np.random.rand(100)
y = np.random.rand(100)
z = np.random.rand(100)
m = np.random.rand(100)  # 假设这是你的 m 值数据

# 创建一个3D图形窗口
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制三维散点图，并使用 m 值作为颜色映射
scatter = ax.scatter(x, y, z, c=m, cmap='viridis', marker='o')

# 设置坐标轴标签
ax.set_xlabel('X轴')
ax.set_ylabel('Y轴')
ax.set_zlabel('Z轴')

# 添加颜色条
colorbar = plt.colorbar(scatter, ax=ax, label='m 值')

# 显示图形
plt.show()