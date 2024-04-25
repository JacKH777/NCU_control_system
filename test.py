import numpy as np
# from io import StringIO

# file_path = 'model.txt'  # 假设的文件路径

# # file_like = StringIO(file_path)
# # # 讀取每一行並轉換為numpy數組
# # array1 = np.loadtxt(file_like, delimiter=',', max_rows=1)
# # array2 = np.loadtxt(file_like, delimiter=',', max_rows=1)
# # # 如果需要繼續讀取其他行，可以繼續使用np.loadtxt()

# # array1, array2
# # 读取文件
# with open(file_path, 'r') as file:
#     lines = file.readlines()

# # 解析四个数组
# arrays = [line.strip().split() for line in lines]

# # # 将数组元素转换为整数或浮点数
# # for i, arr in enumerate(arrays):
# #     arrays[i] = [float(num) if '.' in num else int(num) for num in arr]

# # 返回解析后的数组列表
# print(arrays)
all_data_loaded = np.loadtxt('model.txt', delimiter=',')
    
# 分列存储到各个变量中，假设我们知道有六列
mu_error_loaded = all_data_loaded[:, 0]
sigma_error_loaded = all_data_loaded[:, 1]
mu_delta_loaded = all_data_loaded[:, 2]
sigma_delta_loaded = all_data_loaded[:, 3]
y_loaded = all_data_loaded[:, 4]
y_sigma_loaded = all_data_loaded[:, 5]

print(mu_error_loaded)