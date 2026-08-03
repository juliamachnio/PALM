# 🌴 [ICCV25] PALM: *Performance Analysis of Active Learning Models*

This is the **official repository of “To Label or Not to Label: PALM – A Predictive Model for Evaluating Sample Efficiency in Active Learning Models”**, presented at **ICCV 2025**.

The goal of PALM is to provide a unified and interpretable mathematical model designed to **predict and analyze the behavior of Active Learning (AL) methods**.

This repository brings together three connected active-learning contributions:

1. **PALM** — a predictive model for active-learning learning curves.
2. **ALDA** — a deployment advisor that uses PALM outputs to recommend a method for a target performance level; accepted for oral presentation at the EMA4MICCAI Workshops.
3. **Mechanism-Driven Theory** — computational proxies, phase analysis, and a hard-switch baseline; its implementation is being added on this release branch.

<table>
<tr>
<td width="50%" valign="top">

PALM provides a **predictive description** of learning dynamics from partial observations, enabling:
- Estimation of **future performance** from early-stage results  
- **Principled comparison** across different AL strategies  
- Quantitative analysis of key factors affecting sample efficiency  

</td>
<td width="50%" valign="top" align="center">

<img src="PALM.jpg" alt="PALM curve visualization" width="300"/>

</td>
</tr>
</table>


<div style="display: flex; justify-content: center;">
  <img src="Page 2.png" alt="PALM curve visualization" style="max-width: 100%; height: auto;" />
</div>

---

## 🏝️ The PALM Model

PALM models AL performance trajectories using four interpretable parameters:

$$
A = A_{\text{max}} \left( 1 - (1 - \delta)^{\left(\frac{B}{b} + \alpha\right)^{\beta}} \right)
$$

| Parameter | Meaning | Interpretation |
|------------|----------|----------------|
| **A** | Accuracy | Accuracy achieved by the model in the current AL episode. |
| **Aₘₐₓ** | Achievable accuracy | The asymptotic upper bound of accuracy achievable by the model. |
| **δ (delta)** | Coverage efficiency | Higher δ indicates better utilization of each labeled sample, improving data-space coverage. |
| **α (alpha)** | Initial learning efficiency | Lower α boosts initial accuracy, crucial in low-budget scenarios. |
| **β (beta)** | Scalability | Higher β increases the accuracy improvement rate as more samples are labeled. |
| **b** | Budget efficiency | Smaller `b` means more frequent updates with smaller batches, allowing smoother learning progression. |

---

## ⚙️ Usage Guide

PALM can be used in two main ways:

1. **Standalone mode:** as a separate script integrated into your AL framework.  
   Input required: cumulative budget values per episode and corresponding accuracy results.

2. **Framework mode (recommended):** as a follow-up step inside the [TypiClust framework](https://github.com/avihu111/TypiClust/tree/main) for reproducible evaluation.

---

### 🧩 Environment Installation — Case 1 (Standalone PALM)

Use this setup if you only want to run **PALM.py** for curve fitting, without the full Active Learning framework.

#### 1️⃣ Create an environment
```bash
conda create -n palm pytho=3.12 -y
conda activate palm
conda install numpy scipy matplotlib
```

### 🧩 Environment Installation — Case 2 (Framework mode)

To use PALM as a part of AL framework please refer to [USAGE_Typiclust](typiclust_original_instructions/USAGE_Typiclust.md). Note that original TypiClust uses python 3.7. If you prefere to use python 3.11, [here](Python_311_instalation_guide.md) is the additional guide. 

### ▶️ Running PALM
In both Cases, PALM.py automatically searches for .npy files in the following structure:

```
/path/to/AL/output/<DATASET>/<MODEL>/
├── <METHOD_1>_1/plot_episode_yvalues.npy
├── <METHOD_1>_2/plot_episode_yvalues.npy
├── <METHOD_2>_2/plot_episode_yvalues.npy
├── <METHOD_2>_2/plot_episode_yvalues.npy
└── ...
```
Each .npy file should be a 1D array of accuracies recorded after each Active Learning episode. The <METHOD_1> is name of your AL method and _N is the repetition under the same settings. 


**A) Fit on episode index (default mode) - Example**

```
python PALM.py \
  --base_path /path/to/output/IMAGENET50/resnet50 \
  --dataset IMAGENET50 \
  --model resnet50 \
  --method random_ff_moco_b50 \
  --num_variants 5 \
  --num_samples 15 \
  --x_mode episodes \
  --out_dir palm_fit_out \
  --plot \
  --save_csv

```

**B) Fit on cumulative-normalized budget - Example**

```
python PALM.py \
  --base_path /path/to/output/IMAGENET50/resnet50 \
  --dataset IMAGENET50 \
  --model resnet50 \
  --method random_ff_moco_b50 \
  --num_variants 5 \
  --num_samples 15 \
  --x_mode cumulative_normalized \
  --budget_size 100 \
  --out_dir palm_fit_out \
  --plot \
  --save_csv

```

### 📦 Output Files

After running `PALM.py`, the following files will be saved in the directory specified by `--out_dir`  
(default: `./palm_fit_out`):

| File | Description |
|------|--------------|
| `palm_params.json` | Fitted PALM parameters and metadata (dataset, model, method, episodes, etc.) |
| `palm_params.csv` | (Optional) CSV summary of the parameters, created if `--save_csv` is used |
| `y_avg.npy` | Averaged accuracy values across all selected runs |
| `y_fit.npy` | Fitted PALM curve values corresponding to the averaged data |
| `palm_fit.png` | Visualization of the PALM fit (generated if `--plot` is used) |

> 💡 These outputs allow you to easily reproduce, visualize, or compare fitted curves across Active Learning methods.

---

### 🚩 Common Flags

| Flag | Description |
|------|--------------|
| `--base_path` | Path to the folder containing run subdirectories (e.g. `output/CIFAR10/resnet50/`) |
| `--dataset` | Dataset name (for logging and output files) |
| `--model` | Model name (for logging and output files) |
| `--method` | Prefix of run folders (e.g. `typiclust_moco_b50`) |
| `--num_variants` | Number of repeated runs to average (e.g. 5 for `_1`, `_2`, `_3`, `_4`, `_5`) |
| `--num_samples` | Number of Active Learning episodes to use per run (`-1` = use all) |
| `--x_mode` | X-axis type: `episodes` (default) or `cumulative_normalized` |
| `--budget_size` | Budget per episode (only used if `x_mode=cumulative_normalized`) |
| `--out_dir` | Directory to save PALM outputs (default: `./palm_fit_out`) |
| `--plot` | Display and save the fitted PALM curve as a figure |
| `--save_csv` | Save fitted parameters to `palm_params.csv` in addition to JSON |
| `--init_A_max`, `--init_delta`, `--init_alpha`, `--init_beta` | Optional custom initial guesses for the curve-fitting procedure |
| `--alpha_lo`, `--alpha_hi`, `--beta_hi` | Bounds for fitting parameters (advanced use) |

> 🔹 Use `--help` to see the full list of options and defaults:
> ```bash
> python PALM.py --help
> ```
---

## 🩺 ALDA: Active Learning Deployment Advisor

ALDA is the companion deployment-advisor workflow for PALM. It helps choose
between active-learning methods after observing their early learning curves:

```text
active-learning experiment → PALM fit per method → ALDA recommendation
```

For every candidate method, ALDA reads the PALM output, fits the observed
trajectory in a common cumulative-label space, and estimates the label budget
needed to reach a target score. It recommends the reachable method with the
lowest estimated budget.

ALDA was accepted for oral presentation at the EMA4MICCAI Workshops under the
title *Active Learning Deployment Advisor for Medical Image Classification*.

### What ALDA needs

The recommended interface consumes one output directory from `PALM.py` for
each candidate method. Each directory must contain:

| File | Produced by PALM | Used by ALDA |
| --- | --- | --- |
| `palm_params.json` | Fit metadata, including dataset, method, and acquisition budget | Identifies and aligns methods |
| `y_avg.npy` | Mean score trajectory across selected runs | Fits the ALDA decision model |

Run all candidate methods on the same dataset, task, evaluation metric, and
data split. The methods may use different acquisition batch sizes; ALDA uses
the budget stored by PALM to compare them in cumulative labeled samples.

### Installation

The ALDA core requires Python, NumPy, and SciPy. If you are using the bundled
Deep-AL framework, install its dependencies from the repository root:

```bash
pip install -r deep-al/requirements.txt
```

For standalone PALM fitting, install at least:

```bash
pip install numpy scipy matplotlib
```

### Quick start

#### 1. Run your candidate active-learning methods

Each run should produce `plot_episode_yvalues.npy` under the output layout
already supported by `PALM.py`:

```text
results/CIFAR10/resnet18/
├── typiclust_1/plot_episode_yvalues.npy
├── typiclust_2/plot_episode_yvalues.npy
├── coreset_1/plot_episode_yvalues.npy
└── coreset_2/plot_episode_yvalues.npy
```

#### 2. Fit PALM once per method

From the repository root, create a separate PALM output directory for every
method. Set `--budget_size` to the number of labels acquired per AL episode.

```bash
python PALM.py \
  --base_path results/CIFAR10/resnet18 \
  --dataset CIFAR10 \
  --model resnet18 \
  --method typiclust \
  --num_variants 2 \
  --budget_size 100 \
  --out_dir palm_outputs/typiclust

python PALM.py \
  --base_path results/CIFAR10/resnet18 \
  --dataset CIFAR10 \
  --model resnet18 \
  --method coreset \
  --num_variants 2 \
  --budget_size 100 \
  --out_dir palm_outputs/coreset
```

#### 3. Ask ALDA for a recommendation

Choose a target in percentage points. For example, a target of `80` means
80% accuracy when PALM scores are percentages.

```bash
python deep-al/tools/alda/advisor.py \
  --palm-output palm_outputs/typiclust palm_outputs/coreset \
  --output-dir alda_outputs \
  --target 80
```

ALDA writes two files:

| File | Contents |
| --- | --- |
| `alda_fits.csv` | PALM parameters, fit quality, observed final score, and estimated budget at the target for every method |
| `alda_advice.csv` | One recommendation per dataset, its target, decision reason, and estimated label budget |

### Advising from partial trajectories

To simulate a decision made before the full experiment is complete, pass only
the first *N* observations from each PALM output:

```bash
python deep-al/tools/alda/advisor.py \
  --palm-output palm_outputs/typiclust palm_outputs/coreset \
  --output-dir alda_early_outputs \
  --target 80 \
  --max-points 8
```

At least four unique trajectory points are required for a fit. Early fits are
estimates, not guarantees: use the fit quality in `alda_fits.csv`, compare
methods under matched experimental conditions, and validate a recommendation
before committing a large annotation budget.

### Decision rule

For each dataset, ALDA:

1. Fits the PALM curve to every candidate method’s observed mean trajectory.
2. Estimates the cumulative labels required to reach the requested target.
3. Recommends the method with the lowest reachable estimate.
4. If no method is estimated to reach the target, recommends the method with
   the highest estimated asymptotic score and records that fallback reason.

This makes the decision criterion explicit. It does not claim that a fitted
curve eliminates experimental uncertainty or replaces domain validation.

### Optional CSV interface

Use `--input` only when curves come from an external active-learning framework
and cannot be fitted by this repository’s `PALM.py` workflow. The CSV must
contain these columns:

```text
dataset,method,cumulative_budget,score
```

Scores may be proportions (`0`–`1`) or percentages (`0`–`100`); the target
must use the same scale as the input.

```bash
python deep-al/tools/alda/advisor.py \
  --input external_curves.csv \
  --output-dir alda_outputs \
  --target 0.80
```

### Reproducibility and privacy

ALDA does not require datasets, images, checkpoints, or patient-derived data
to be committed to this repository. The forthcoming paper-reproduction layer
will add dataset acquisition instructions, split manifests, configurations,
and validation tools. Do not commit local run folders or generated ALDA/PALM
outputs to the public repository.

### Validation

Run the synthetic smoke test from `deep-al/tools`:

```bash
python alda/tests/test_advisor.py
```

The test covers both the optional CSV path and the standard direct
`PALM.py → ALDA` handoff.

### Citation

Citation metadata and the official paper link will be added once available.
Until then, please refer to the paper title above and cite PALM when using the
underlying learning-curve model.

---

## 🧭 Mechanism-Driven Theory

The next component will add the computational proxies, phase/regime analysis,
and proxy-derived `hard_switch` baseline from the mechanism-driven theory
paper. It deliberately excludes ALPS, urgency estimation, and soft-allocation
work, which belong to separate research projects.

---

## 📚 Citing this Repository

If you find **PALM** useful in your research, please consider citing our ICCV 2025 paper and the repositories our work builds upon.

This repository builds on concepts and frameworks designed by [TypiClust](https://github.com/avihu111/TypiClust), [SCAN](https://github.com/wvangansbeke/Unsupervised-Classification), and [Deep-AL](https://github.com/decile-team/deep-active-learning). Please consider citing their work along with ours.

---

### 🏝️ PALM (ICCV 2025)

```
@article{machnio2025label,
  title={To Label or Not to Label: PALM--A Predictive Model for Evaluating Sample Efficiency in Active Learning Models},
  author={Machnio, Julia and Nielsen, Mads and Ghazi, Mostafa Mehdipour},
  journal={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2025}
}
```

### 📘 Related References

```
@article{hacohen2022active,
  title={Active learning on a budget: Opposite strategies suit high and low budgets},
  author={Hacohen, Guy and Dekel, Avihu and Weinshall, Daphna},
  journal={arXiv preprint arXiv:2202.02794},
  year={2022}
}

@article{yehudaActiveLearningCovering2022,
  title = {Active {{Learning Through}} a {{Covering Lens}}},
  author = {Yehuda, Ofer and Dekel, Avihu and Hacohen, Guy and Weinshall, Daphna},
  journal={arXiv preprint arXiv:2205.11320},
  year={2022}
}

@article{mishal2024dcom,
      title={DCoM: Active Learning for All Learners}, 
      author={Mishal, Inbal and Weinshall, Daphna},
      journal={arXiv preprint arXiv:2407.01804},
      year={2024}
}

@inproceedings{vangansbeke2020scan,
  title={Scan: Learning to classify images without labels},
  author={Van Gansbeke, Wouter and Vandenhende, Simon and Georgoulis, Stamatios and Proesmans, Marc and Van Gool, Luc},
  booktitle={Proceedings of the European Conference on Computer Vision},
  year={2020}
}

@article{Chandra2021DeepAL,
    Author = {Akshay L Chandra and Vineeth N Balasubramanian},
    Title = {Deep Active Learning Toolkit for Image Classification in PyTorch},
    Journal = {https://github.com/acl21/deep-active-learning-pytorch},
    Year = {2021}
}

@article{Munjal2020TowardsRA,
  title={Towards Robust and Reproducible Active Learning Using Neural Networks},
  author={Prateek Munjal and N. Hayat and Munawar Hayat and J. Sourati and S. Khan},
  journal={ArXiv},
  year={2020},
  volume={abs/2002.09564}
}
```

## License
This toolkit and PALM is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.
