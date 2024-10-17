
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from kerasFuzzy import Torque_ANFIS_1kg_multi

# Create an instance of your Torque_ANFIS_1kg class
model = Torque_ANFIS_1kg_multi()
file_path = './exp/2024_09_30_1156/1/down_model.npy'
# Load the latest saved model data
loaded_data = np.load(file_path)
# print(loaded_data)
loaded_data = loaded_data[-42:]

# Load the model parameters into the class instance
model.load_model(loaded_data)

# Define the input ranges for torque and delta_torque
torque_range = np.linspace(0, 2.5, 50)  # Modify range as needed
delta_torque_range = np.linspace(-1, 1, 50)  # Modify range as needed

# Create a meshgrid for plotting in 3D
torque_grid, delta_torque_grid = np.meshgrid(torque_range, delta_torque_range)

# Initialize an array to hold the predictions
output_grid = np.zeros(torque_grid.shape)

# Populate the output grid by making predictions for each pair of (torque, delta_torque)
for i in range(torque_grid.shape[0]):
    for j in range(torque_grid.shape[1]):
        # Use the predict method from the Torque_ANFIS_1kg class
        torque = torque_grid[i, j]
        delta_torque = delta_torque_grid[i, j]
        output_grid[i, j] = model.predict([torque], [delta_torque])

# Plotting in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Surface plot
ax.plot_surface(torque_grid, delta_torque_grid, output_grid, cmap='viridis')

# Add labels
ax.set_xlabel('Torque')
ax.set_ylabel('Delta Torque')
ax.set_zlabel('Output')

# Show the plot
plt.show()
