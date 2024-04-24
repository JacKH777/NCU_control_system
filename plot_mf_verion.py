# import numpy as np
# import matplotlib.pyplot as plt

# # 定义隶属函数的参数
# mu_error = np.array([-5, -3, -1, 0, 1, 3, 5])
# sigma_error = np.array([1.5, 1, 1, 1, 1, 1, 1.5])

# # 创建一个范围内的error值
# error_range = np.linspace(-10, 10, 300)

# # 定义自定义Sigmoid函数
# def custom_sigmoid(x, mu,sigma_error):
#     return 1 / (1 + np.exp(-sigma_error * (x - mu)))

# # 定义高斯函数
# def gaussian(x, mu, sigma):
#     return np.exp(-np.square(x - mu) / np.square(sigma))

# # 计算第一个和最后一个隶属函数的值
# error_first = custom_sigmoid(error_range, mu_error[0],-sigma_error[0])
# error_last = custom_sigmoid(error_range, mu_error[-1],sigma_error[-1])

# # 计算中间的五个隶属函数的值
# rul_errors = np.array([gaussian(error_range, mu, sigma) for mu, sigma in zip(mu_error[1:6], sigma_error[1:6])])

# # 绘制隶属函数
# plt.figure(figsize=(10, 6))
# plt.plot(error_range, error_first, label='First Rule')
# for i, rul_error in enumerate(rul_errors, start=2):
#     plt.plot(error_range, rul_error, label=f'Rule {i}')
# plt.plot(error_range, error_last, label='Last Rule')
# plt.title('Membership Functions')
# plt.xlabel('e / é')
# plt.ylabel('Membership Value')
# plt.legend()
# plt.grid(True)
# plt.show()

import numpy as np
import matplotlib.pyplot as plt

# 定义隶属函数的参数
mu_error = np.array([-5, -3, -1, 0, 1, 3, 5])
sigma_error = np.array([1.5, 1, 1, 1, 1, 1, 1.5])

# 创建一个范围内的error值
error_range = np.linspace(-10, 10, 300)

# 定义自定义Sigmoid函数
def custom_sigmoid(x, mu,sigma_error):
    return 1 / (1 + np.exp(-sigma_error * (x - mu)))

# 定义高斯函数
def gaussian(x, mu, sigma):
    return np.exp(-np.square(x - mu) / np.square(sigma))

# 计算第一个和最后一个隶属函数的值
error_first = custom_sigmoid(error_range, mu_error[0],-sigma_error[0])
error_last = custom_sigmoid(error_range, mu_error[-1],sigma_error[-1])

# 计算中间的五个隶属函数的值
rul_errors = np.array([gaussian(error_range, mu, sigma) for mu, sigma in zip(mu_error[1:6], sigma_error[1:6])])

# 绘制隶属函数
plt.figure(figsize=(10, 6))
plt.plot(error_range, error_first, label='First Rule')
for i, rul_error in enumerate(rul_errors, start=1):
    plt.plot(error_range, rul_error, label=f'Rule {i}')
plt.plot(error_range, error_last, label='Last Rule')
plt.title('Membership Functions')
plt.xlabel('e / é')
plt.ylabel('Membership Value')
plt.legend()
plt.grid(True)
plt.xlim(-10, 10)
plt.show()
