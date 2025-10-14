import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def rankk_model(x, acc_max, delta, alpha, betha):
    """Curve equation with fixed b value."""
    return acc_max * (1 - (1 - delta) ** ((x + betha)**alpha ))

# def rankk_model(x, acc_max, delta, alpha, betha):
#     """Curve equation with fixed b value."""
#     return acc_max * (1 - (1 - delta) ** (np.log(x + betha+1) ))

# Base directory setup
base_path = "/home/kgc221/rankk/output"
dataset = "CIFAR10"
method = "random_nf"  # Single method
num_variants = 5
model = "resnet18"

num_of_samples = 200
b = 100  # Set a fixed value for b

# List to store y-values from different runs
y_values_list = []

for i in range(1, num_variants + 1):
    y_file_path = os.path.join(base_path, dataset, model, f"{method}_simclr_b20_{i}", "plot_episode_yvalues.npy")
        
    if os.path.exists(y_file_path):
        y_data = np.load(y_file_path)[:num_of_samples]  # Load only the first 20 samples
        y_values_list.append(y_data)

# Ensure that at least one file was loaded
if not y_values_list:
    raise FileNotFoundError(f"No y-values files found for method {method}")

# Compute average across multiple runs
y_avg_array = np.mean(y_values_list, axis=0)  # This is now a NumPy array
print(y_avg_array)
b=100
# Generate x-values to match the number of samples
x_data = (np.arange(1,  len(y_avg_array) + 1) )
print(x_data)

print(x_data)

# Fit the model
plt.figure(figsize=(10, 6))

# Ensure x_data and y_avg are the same length
x_fit = x_data

# Initial guesses for parameters (excluding b)
initial_guesses = [max(y_avg_array), 0.5, 1, 0]  # [acc_max, delta, alpha, betha]

# Set parameter bounds (excluding b)
bounds = (
    [0, 0, 0, -1000],  # Lower bounds
    [100, 1, 10, 1000]  # Upper bounds
)

# Fit the model (b is fixed)
popt, pcov = curve_fit(rankk_model, x_data, y_avg_array, p0=initial_guesses, bounds=bounds)

# Extract fitted parameters (excluding b)
acc_max, delta, alpha, betha = popt

print(f"Method: {method}")
print(f"b (fixed): {b}")
print(f"acc_max: {acc_max:.4f}, delta: {delta:}, alpha: {alpha:}, betha: {betha:}")

# # Generate fitted curve
# fitted_y = rankk_model(x_fit, acc_max, delta, alpha, betha)

# Plot the results
# plt.plot(x_fit, y_avg_array, 'o', label=f'{method} Data', markersize=6)
# plt.plot(x_fit, fitted_y, '-', label=f'{method} Fit', linewidth=2)

# plt.title("Averaged Data and Fitted Curve")
# plt.xlabel("X Values (n)")
# plt.ylabel("Y Values (Accuracy)")
# plt.legend()
# plt.grid()
# plt.show()



# # Define the curve equation
# def rankk_model(x, b, acc_max, delta, alpha, betha):
#     return acc_max * (1 - (1 - delta) ** ((x * b + betha) ** alpha))

# # Load y_data from .npy file
# path_1 = "/home/kgc221/rankk/output/CIFAR10/resnet18/random_nf_5/plot_episode_yvalues.npy"
# y_data = np.load(path_1)

# y_data = y_data[:]

# # Generate x_values based on the length of y_data
# x_data = np.arange(1, len(y_data) + 1)

# # Initial guesses for parameters [acc_max, delta, alpha, betha]
# initial_guesses = [max(y_data), 0.5, 1, 0]  # Adjusted based on typical starting assumptions

# # Set bounds for parameters
# bounds = (
#     [0, 0, 0, -100],  # Lower bounds: acc_max >= 0, delta >= 0, alpha >= 0, betha >= -10
#     [100, 1, 5, 100],  # Upper bounds: acc_max <= 100, delta < 1, alpha <= 5, betha <= 10
# )

# # Fit the model to the data
# popt, pcov = curve_fit(rankk_model, x_data, y_data, p0=initial_guesses, bounds=bounds)

# # Extract fitted parameters
# acc_max, delta, alpha, betha = popt

# # Display the results
# print("Estimated Parameters:")
# print("num of points:", len(x_data))
# print(f"acc_max: {acc_max:.4f}")
# print(f"delta: {delta:.4f}")
# print(f"alpha: {alpha:.4f}")
# print(f"betha: {betha:.4f}")

# # Generate the fitted curve
# fitted_y = rankk_model(x_data, acc_max, delta, alpha, betha)

# # Plot the original data points and the fitted curve
# plt.figure(figsize=(10, 6))
# plt.plot(x_data, y_data, 'o', label='Data Points', markersize=8)  # Original data points
# plt.plot(x_data, fitted_y, '-', label='Fitted Curve', linewidth=2)  # Fitted curve
# plt.title("Data Points and Fitted Curve")
# plt.xlabel("X Values (n)")
# plt.ylabel("Y Values (Accuracy)")
# plt.legend()
# plt.grid()
# plt.show()
