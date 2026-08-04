import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

params_table = {}

base_path = "{your_path}/output"
dataset = "IMAGENET50"
model = "resnet50"
# methods = ["random_ff"]
# methods = ["uncertainty_ff"]
methods = ["margin_ff"]
# methods = ["entropy_ff"]
# methods = ["dbal_ff"]
# methods = ["typiclust_rp_ff"]
embeddings = ["moco", "mocov3", "byol", "simclr"]
# embeddings = ["byol"]
num_variants = 3
budget_size = 50
num_of_samples = 20

output_csv_dir = os.path.join(base_path, "parameter_csv")
os.makedirs(output_csv_dir, exist_ok=True)


def rankk_model(x, acc_max, delta, alpha, betha):
    return acc_max * (1 - (1 - delta) ** ((x + alpha) ** betha))


for method in methods:
    params_table[method] = {}
    for embedding in embeddings:
        y_values_list = []

        for i in range(1, num_variants + 1):
            dir_path = os.path.join(base_path, dataset, model, f"{method}_{embedding}_b{budget_size}_{i}")
            x_file_path = os.path.join(dir_path, "plot_episode_xvalues.npy")
            y_file_path = os.path.join(dir_path, "plot_episode_yvalues.npy")

            if os.path.exists(x_file_path) and os.path.exists(y_file_path):
                x_data = np.load(x_file_path)
                y_data = np.load(y_file_path)

                if len(x_data) != len(y_data):
                    print(f"⚠️ Warning: Mismatched x and y lengths in {dir_path}. Skipping.")
                    continue
                print(x_data, y_data)
                # Remove duplicates in x and corresponding y
                unique_x, indices = np.unique(x_data, return_index=True)
                x_clean = x_data[np.sort(indices)]
                y_clean = y_data[np.sort(indices)]
                print(x_clean, y_clean)

                # Overwrite cleaned files
                np.save(x_file_path, x_clean)
                np.save(y_file_path, y_clean)

                if len(y_clean) >= num_of_samples:
                    y_values_list.append(y_clean[:num_of_samples])
                else:
                    print(f"⚠️ Warning: {y_file_path} has only {len(y_clean)} samples, needed {num_of_samples}. Skipping.")
            else:
                print(f"⚠️ Warning: Files do not exist in {dir_path}")

        if y_values_list:
            x_data = np.load(os.path.join(base_path, dataset, model, f"{method}_{embedding}_b{budget_size}_1", "plot_episode_xvalues.npy"))[:num_of_samples]
            y_avg = np.mean(y_values_list, axis=0)
            initial_guesses = [90, 0.1, 5, 1]
            bounds = ([0, 0, 0, -1], [100, 1, 20, 10])
            popt, _ = curve_fit(rankk_model, x_data, y_avg, p0=initial_guesses, bounds=bounds)
            acc_max, delta, alpha, betha = popt
            params_table[method][embedding] = [round(acc_max, 8), round(delta, 15), round(alpha, 8), round(betha, 8)]

rows = []
for method, embeddings_data in params_table.items():
    for embedding, params in embeddings_data.items():
        rows.append([method, embedding] + params)

param_df = pd.DataFrame(rows, columns=["Method", "Embedding", "acc_max", "delta", "alpha", "betha"])

csv_path = os.path.join(output_csv_dir, f"parameter_table_{dataset}_{methods}.csv")
param_df.to_csv(csv_path, index=False)

print(param_df)
