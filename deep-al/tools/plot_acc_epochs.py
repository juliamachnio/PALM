import numpy as np
import matplotlib.pyplot as plt
import os 



def load_npy_files(file_list):
    """Loads multiple .npy files into a list of numpy arrays."""
    if not file_list:  
        return None
    
    data = [np.load(f)[0:100] for f in file_list]
    return np.array(data)  # Shape: (num_files, num_points)

def rankk(iteration_num, budget, acc_max, delta, alpha, betha):
    """Applies the rankk transformation to accuracy values."""

    iteration_num = iteration_num+1
    y = acc_max*(1-(1-delta)**(iteration_num*budget+betha)**alpha)

    return y


def plot_multiple_methods(x_file, y_dict, save_path=None):
    """
    Plots multiple averaging methods from different sets of .npy y-files.

    Args:
        x_file (str): Path to the .npy file for x-axis values.
        y_dict (dict): Dictionary with method names as keys and lists of y-file paths as values.
        save_path (str, optional): Path to save the plot image.
    """
    # Load x-axis values
    budget_size = 100
    x_vals = np.load(x_file)
    x_vals += 1
    # x_vals = np.insert(x_vals, 0, 0)
    x_vals *= budget_size
    print(x_vals)


    # Define colors for different methods
    colors = {
        "typiclust_rp": "b",
        "margin_ff": "g",
        "bald": "r",
        "dbal": "c",
        "entropy": "m",
        "margin_nf": "y",
        "uncertainty": "k",
        "random_ff": "orange",
        "random_nf": "m",
        "typiclust_rp_nf": "r",
    }

    plt.figure(figsize=(8, 5))

    # iterations = 100
    # x_vals = x_vals[:iterations]

    for method, y_files in y_dict.items():
        y_data = load_npy_files(y_files)  # Shape: (num_y_files, num_points)
        x_vals = x_vals[:y_data.shape[1]]
       
        

        if y_data is None:  # Skip if no files available for the method
            continue

        # Compute mean and std across y_files

        y_mean = np.mean(y_data, axis=0)
        y_std = np.std(y_data, axis=0)
        acc_max = np.max(y_mean, axis=0)
        # y_mean = np.insert(y_mean, 0, 0)
        print(y_mean)
        # y_std = np.insert(y_std, 0, 0)
        print(acc_max)
        if method == "margin_ff":
            y_rankk = np.array([rankk(i, 100, acc_max, 0.45,0.39, -0.8) for i, y in enumerate(y_mean)])
            
        elif method == "random_ff_simclr":
            y_rankk = np.array([rankk(i, 100, acc_max, 0.33,0.48, -0.28) for i, y in enumerate(y_mean)])
            
        elif method == "margin_nf":
            y_rankk = np.array([rankk(i, 90, 100, 0.225,0.4691, 11.4) for i, y in enumerate(y_mean)])
            
        elif method == "random_nf":
            y_rankk = np.array([rankk(i, 100, 90, 0.225,0.4691, 1.4) for i, y in enumerate(y_mean)])
          

        print(y_data.shape[1])
        # Plot mean with shaded std
        plt.plot(x_vals, y_mean, label=method, color=colors.get(method, "gray"), linewidth=2)
        plt.fill_between(x_vals, y_mean - y_std, y_mean + y_std, color=colors.get(method, "gray"), alpha=0.3)

        # Plot rankk-transformed curve

        plt.plot(x_vals, y_rankk, label=f"{method} (rankk)", linestyle="dashed", color=colors.get(method, "gray"))
        # print(f"Method: {method}")
        # print(f"y_mean[:5]: {y_mean[:10]}")
        # print(f"y_std[:5]: {y_std[:5]}")

    # Formatting
    plt.xlabel("Cumulative Budget")
    # plt.xlabel("Iteration")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Test Accuracy vs. Training Cumulative Budget")
    plt.legend()
    plt.grid(True)

    # Save or show plot
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


base_path = "/home/kgc221/rankk/output"
dataset = "CIFAR10"
methods = ["random_nf", "random_ff_simclr"]
# methods = ["random_nf", "margin_nf", "random_ff", "margin_ff",]
num_variants = 5
model = 'resnet18'

x_file = os.path.join(base_path, dataset, 'resnet18',"random_nf_1", "plot_episode_xvalues.npy")

y_dict = {}

for method in methods:
    key = f"{method}"  # Dictionary key

    y_files = []  # Initialize empty list to store file paths

    for i in range(1, 6):
        print(i)
        print(f"{method}_{i}")
        y_file_path = os.path.join(base_path, dataset, model, f"{method}_{i}", "plot_episode_yvalues.npy")
        print(y_file_path)

        # Add only if file exists
        if os.path.exists(y_file_path):
            y_files.append(y_file_path)

    if y_files:  # Only add to dictionary if files exist
        y_dict[key] = y_files



# Print results to verify
print(f"x_file: {x_file}")
print(f"y_dict: {y_dict}")


save_path = "/home/kgc221/rankk/pplots/cifar10_20AL_simclr.png"

# Run the plotting function
plot_multiple_methods(x_file, y_dict, save_path)