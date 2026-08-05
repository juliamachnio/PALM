import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl

mpl.rcParams['font.family'] = 'serif'

# Mapping for embedding display names
embedding_display_names = {
    "moco": "MoCov2+",
    "mocov3": "MoCov3",
    "byol": "BYOL",
    "simclr": "SimCLR"
}

base_path = "{your_path}/output"
dataset = "IMAGENET50"
# methods = ["random_ff"]
# methods = ["uncertainty_ff"]
# methods = ["dbal_ff"]
methods = ["margin_ff"]
# methods = ["entropy_ff"]
# methods = ["typiclust_rp_ff"]
embeddings = ["moco", "mocov3", "byol", "simclr"]
num_variants = 3
model = "resnet50"
budget_size = 50
num_of_samples = 20

output_dir = os.path.join(base_path, "plots")
os.makedirs(output_dir, exist_ok=True)

def rankk_model(x, acc_max, delta, alpha, beta):
    return acc_max * (1 - (1 - delta) ** ((x + alpha)**beta))

x_file = os.path.join(base_path, dataset, model, f"random_ff_{embeddings[0]}_b{budget_size}_1", "plot_episode_xvalues.npy")
if os.path.exists(x_file):
    full_x_data = np.load(x_file)
else:
    raise FileNotFoundError(f"x-values file not found: {x_file}")

colors = plt.get_cmap("tab10").colors

param_df = pd.read_csv(os.path.join(base_path, "parameter_csv", f"parameter_table_{dataset}_{methods}.csv"))

for method in methods:
    # Increase figure height to add space for the legend
    plt.figure(figsize=(8, 6))  # height increased from 4 to 5.5

    for idx, embedding in enumerate(embeddings):
        filtered_params = param_df[(param_df["Method"] == method) & (param_df["Embedding"] == embedding)]

        if filtered_params.empty:
            print(f"⚠ Warning: No parameters found for {method} with {embedding}. Skipping.")
            continue

        params = filtered_params.iloc[0]
        acc_max, delta, alpha, beta = params["acc_max"], params["delta"], params["alpha"], params["betha"]

        # LaTeX formatted parameter text
        param_text = (
            f"{embedding_display_names[embedding]} "
            f"(A$_{{\\max}}$={acc_max:.2f}, "
            f"$\\delta$={delta:.4f}, "
            f"$\\alpha$={alpha:.2f}, "
            f"$\\beta$={beta:.2f})"
        )

        y_values_list = []
        for i in range(1, num_variants + 1):
            y_file_path = os.path.join(base_path, dataset, model, f"{method}_{embedding}_b{budget_size}_{i}", "plot_episode_yvalues.npy")
            if os.path.exists(y_file_path):
                y_data = np.load(y_file_path)
                y_values_list.append(y_data)

        if not y_values_list:
            print(f"Skipping {method} ({embedding}) - no data found.")
            continue

        min_length = min(len(y) for y in y_values_list)
        y_values_list = [y[:min_length] for y in y_values_list]
        y_avg_array = np.mean(y_values_list, axis=0)
        x_full_budget = (full_x_data[:len(y_avg_array)] + 1) * budget_size

        cumulative_budget = (full_x_data[:min_length] + 1) * budget_size
        y_pred = rankk_model(full_x_data[:min_length], acc_max, delta, alpha, beta)

        plt.plot(cumulative_budget, y_pred, '-', label=param_text, color=colors[idx % len(colors)])
        plt.plot(cumulative_budget, y_avg_array[:min_length], '.', color=colors[idx % len(colors)], alpha=0.5)

    plt.title(f"{method.replace('_ff', '').replace('_nf', '').title()}", fontsize=16)
    plt.xlabel("Cumulative Training Budget", fontsize=15)
    plt.ylabel("Test Accuracy (%)", fontsize=15)
    plt.xlim([0, max(cumulative_budget)])
    plt.ylim([0, 100])
    plt.grid(True, linestyle="--", alpha=0.6)

    # Move legend clearly below the plot with proper spacing
    plt.legend(
        fontsize=13,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.18),
        ncol=1,
        frameon=False
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)  # add space for the legend

    save_path_png = os.path.join(output_dir, f"{dataset}_{method}_predicted_curves.png")
    save_path_svg = os.path.join(output_dir, f"{dataset}_{method}_predicted_curves.svg")

    plt.savefig(save_path_png, bbox_inches="tight")
    plt.savefig(save_path_svg, bbox_inches="tight")
    plt.close()

    print(f"Saved combined predicted curve plot for {method}: {save_path_png}, {save_path_svg}")
