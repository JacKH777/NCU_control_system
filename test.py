import numpy as np

# 假設 result 是之前計算出的陣列
arr1 = np.array([1, 3, 5, 7])
arr2 = np.array([2, 2, 6, 8])
smaller = np.less(arr1, arr2)  # 或者使用 arr1 < arr2
result = np.where(smaller, 1, 0)

# 獲得相反的陣列
opposite_result = 1 - result

print("Result:", result)
print("Opposite Result:", opposite_result)