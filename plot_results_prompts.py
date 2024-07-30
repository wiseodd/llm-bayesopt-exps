import torch
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import utils.plots as plot_utils
import os
import seaborn as sns


PROBLEMS_ASSORTED = ["redox-mer", "photoswitch"]
PROBLEMS = ["redox-mer", "laser", "solvation", "photoswitch", "kinase", "pce"]
PROMPT_TYPES = ["just-smiles", "completion", "single-number", "naive"]
IS_MAX = {
    "redox-mer": False,
    "solvation": False,
    "kinase": False,
    "laser": True,
    "pce": True,
    "photoswitch": True,
}
FEATURE_NAMES_BASE = [
    # 'fingerprints',
    # 'molformer'
]
REAL_FEATURE_NAMES_LLM = [
    "t5-base",
    # 'gpt2-medium',
    # 'gpt2-large',
    "llama-2-7b",
    "t5-base-chem",
]
REAL_FEATURE_NAMES = FEATURE_NAMES_BASE + REAL_FEATURE_NAMES_LLM
METHODS = [
    # 'random',
    # 'gp',
    "laplace"
]

n_colors = len(REAL_FEATURE_NAMES) * len(METHODS)
orig_palette = sns.color_palette("tab10", n_colors)
sns.set_palette(reversed(orig_palette), n_colors)

PROBLEM2TITLE = {
    "redox-mer": "Redoxmer",
    "solvation": "Solvation",
    "kinase": "Kinase",
    "laser": "Laser",
    "pce": "Photovoltaics",
    "photoswitch": "Photoswitches",
}
PROBLEM2LABEL = {
    "redox-mer": r"Redox Potential ($\downarrow$)",
    "solvation": r"Solvation Energy ($\downarrow$)",
    "kinase": r"Docking Score ($\downarrow$)",
    "laser": r"Strength ($\uparrow$)",
    "pce": r"PCE ($\uparrow$)",
    "photoswitch": r"Wavelength ($\uparrow$)",
}
METHOD2LABEL = {"random": "RS", "gp": "GP", "laplace": "LA"}
FEATURE2LABEL = {
    "fingerprints": "FP",
    "molformer": "MolFormer",
    "gpt2-medium": "GPT2-M",
    "gpt2-large": "GPT2-L",
    "llama-2-7b": "LL2-7B",
    "t5-base": "T5",
    "t5-base-chem": "T5-Chem",
}
FEATURE2COLOR = {
    "random-fingerprints": (0.4980392156862745, 0.4980392156862745, 0.4980392156862745),
    "gp-fingerprints": (0.8901960784313725, 0.4666666666666667, 0.7607843137254902),
    "laplace-fingerprints": (
        0.5490196078431373,
        0.33725490196078434,
        0.29411764705882354,
    ),
    "laplace-molformer": (0.5803921568627451, 0.403921568627451, 0.7411764705882353),
    "laplace-t5-base": (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
    "laplace-gpt2-medium": (
        0.17254901960784313,
        0.6274509803921569,
        0.17254901960784313,
    ),
    "laplace-llama-2-7b": (1.0, 0.4980392156862745, 0.054901960784313725),
    "laplace-t5-base-chem": (
        0.12156862745098039,
        0.4666666666666667,
        0.7058823529411765,
    ),
}
RANDSEEDS = [1, 2, 3, 4, 5]

# Plot summary
# ---------------------------------
FIG_WIDTH = 1
FIG_HEIGHT = 0.2
rc_params, fig_width, fig_height = plot_utils.get_mpl_rcParams(
    FIG_WIDTH, FIG_HEIGHT, single_col=True
)
plt.rcParams.update(rc_params)

fig, axs = plt.subplots(
    len(PROBLEMS_ASSORTED),
    len(PROMPT_TYPES),
    sharex=True,
    sharey="row",
    constrained_layout=False,
)
fig.set_size_inches(fig_width, fig_height)
plt.subplots_adjust(
    left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5
)

for i, problem in enumerate(PROBLEMS_ASSORTED):
    if problem is None:
        continue

    # Plot optimal val
    targets = torch.load(
        f"data/cache/{problem}/llama-2-7b-completion-average_targets.bin"
    )
    targets = torch.tensor(targets).flatten()
    best_val = targets.max() if IS_MAX[problem] else -targets.max()
    print(best_val)

    for j, prompt_type in enumerate(PROMPT_TYPES):
        ax = axs[i, j]
        ax.axhline(best_val, c="k", ls="dashed", zorder=1000)

        # Plot methods
        for feature_name in REAL_FEATURE_NAMES:
            for method in METHODS:
                if (
                    method == "random" or method == "gp"
                ) and feature_name != "fingerprints":
                    continue

                path = f"results/{problem}/fixed/laplace/{feature_name}-{prompt_type}-average"
                if not os.path.exists(path):
                    os.makedirs(path)

                trace_best_y = np.stack(
                    [np.load(f"{path}/trace_best_y_10_ts_{rs}.npy") for rs in RANDSEEDS]
                )
                mean = np.mean(trace_best_y, axis=0)[1:]  # Over randseeds
                sem = st.sem(trace_best_y, axis=0)[1:]  # Over randseeds
                T = np.arange(len(mean)) + 1

                c = FEATURE2COLOR[f"{method}-{feature_name}"]
                ax.plot(
                    T,
                    mean,
                    color=c,
                    label=f"{METHOD2LABEL[method]}-{FEATURE2LABEL[feature_name]}",
                )
                ax.fill_between(T, mean - sem, mean + sem, color=c, alpha=0.25)

        title = f"{prompt_type}"
        if i == 0:  # First row only
            ax.set_title(title)
        if j == 0:  # Only for the first column
            ax.set_ylabel(PROBLEM2LABEL[problem])
        if i == len(PROBLEMS_ASSORTED) - 1:  # Only for the bottom row
            ax.set_xlabel(r"$t$")
        ax.set_xlim(1, 100)

        handles, labels = ax.get_legend_handles_labels()

fig.legend(
    handles, labels, loc="upper center", bbox_to_anchor=(0, 1.065, 1, 0.005), ncols=8
)

# Save results
path = "../paper/figs"
if not os.path.exists(path):
    os.makedirs(path)

fname = "prompts_assorted"
plt.savefig(f"{path}/{fname}.pdf", bbox_inches="tight")


# Plot all
# ---------------------------------
FIG_WIDTH = 1
FIG_HEIGHT = 0.6
rc_params, fig_width, fig_height = plot_utils.get_mpl_rcParams(
    FIG_WIDTH, FIG_HEIGHT, single_col=False
)
plt.rcParams.update(rc_params)

fig, axs = plt.subplots(
    len(PROBLEMS), len(PROMPT_TYPES), sharex=True, sharey="row", constrained_layout=True
)
fig.set_size_inches(fig_width, fig_height)

for i, problem in enumerate(PROBLEMS):
    if problem is None:
        continue

    # Plot optimal val
    targets = torch.load(
        f"data/cache/{problem}/llama-2-7b-completion-average_targets.bin"
    )
    targets = torch.tensor(targets).flatten()
    best_val = targets.max() if IS_MAX[problem] else -targets.max()
    print(best_val)

    for j, prompt_type in enumerate(PROMPT_TYPES):
        ax = axs[i, j]
        ax.axhline(best_val, c="k", ls="dashed", zorder=1000)

        # Plot methods
        for feature_name in REAL_FEATURE_NAMES:
            for method in METHODS:
                if (
                    method == "random" or method == "gp"
                ) and feature_name != "fingerprints":
                    continue

                path = f"results/{problem}/fixed/laplace/{feature_name}-{prompt_type}-average"
                if not os.path.exists(path):
                    os.makedirs(path)

                trace_best_y = np.stack(
                    [np.load(f"{path}/trace_best_y_10_ts_{rs}.npy") for rs in RANDSEEDS]
                )
                mean = np.mean(trace_best_y, axis=0)[1:]  # Over randseeds
                sem = st.sem(trace_best_y, axis=0)[1:]  # Over randseeds
                T = np.arange(len(mean)) + 1

                c = FEATURE2COLOR[f"{method}-{feature_name}"]
                ax.plot(
                    T,
                    mean,
                    color=c,
                    label=f"{METHOD2LABEL[method]}-{FEATURE2LABEL[feature_name]}",
                )
                ax.fill_between(T, mean - sem, mean + sem, color=c, alpha=0.25)

        title = f"{prompt_type}"
        if i == 0:  # First row only
            ax.set_title(title)
        if j == 0:  # Only for the first column
            ax.set_ylabel(PROBLEM2LABEL[problem])
        if i == len(PROBLEMS) - 1:  # Only for the bottom row
            ax.set_xlabel(r"$t$")
        ax.set_xlim(1, 100)

        handles, labels = ax.get_legend_handles_labels()

fig.legend(
    handles, labels, loc="upper center", bbox_to_anchor=(0, 1.065, 1, 0.002), ncols=8
)

# Save results
path = "../paper/figs"
if not os.path.exists(path):
    os.makedirs(path)

fname = "prompts"
plt.savefig(f"{path}/{fname}.pdf", bbox_inches="tight")
