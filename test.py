import numpy as np

# Create a sample numpy array
arr = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.02, 0.3])

output_gauss_width = np.asarray([3, 3, 3, 3, 3, 3, 3])
output_gauss_member = np.asarray([0.      ,   0.      ,   0.    ,     0.24935221, 1.     ,    0.24935221,0.        ])
print(np.sum(output_gauss_width * output_gauss_member))