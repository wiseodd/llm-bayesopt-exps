import torch
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import utils.plots as plot_utils
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    "--prompt_type",
    choices=["just-smiles", "completion", "single-number", "naive"],
    default="just-smiles",
)
parser.add_argument("--dont_average_llm_feature", default=False, action="store_true")
parser.add_argument("--acqf", choices=["ei", "ucb", "ts"], default="ts")
parser.add_argument("--prompt_type_in_title", default=False, action="store_true")
args = parser.parse_args()

PROBLEMS = ["redox-mer", "laser", "solvation", "pce", "kinase", "photoswitch"]
IS_MAX = {
    "redox-mer": False,
    "solvation": False,
    "kinase": False,
    "laser": True,
    "pce": True,
    "photoswitch": True,
}
FEATURE_NAMES_BASE = ["fingerprints", "molformer"]
REAL_FEATURE_NAMES_LLM = [
    "t5-base",
    "gpt2-medium",
    # 'gpt2-large',
    "llama-2-7b",
    "t5-base-chem",
]
FEATURE_NAMES_LLM = [f"{n}-{args.prompt_type}" for n in REAL_FEATURE_NAMES_LLM]
FEATURE_NAMES_LLM = [
    f'{n}{"-average" if not args.dont_average_llm_feature else ""}'
    for n in FEATURE_NAMES_LLM
]
FEATURE_NAMES = FEATURE_NAMES_BASE + FEATURE_NAMES_LLM
REAL_FEATURE_NAMES = FEATURE_NAMES_BASE + REAL_FEATURE_NAMES_LLM
METHODS = ["random", "gp", "laplace"]
FEATURE2COLOR = {
    "random-fingerprints": (0.4980392156862745, 0.4980392156862745, 0.4980392156862745),
    "gp-fingerprints": (0.8901960784313725, 0.4666666666666667, 0.7607843137254902),
    "laplace-fingerprints": (
        0.5490196078431373,
        0.33725490196078434,
        0.29411764705882354,
    ),
    "laplace-molformer": (0.5803921568627451, 0.403921568627451, 0.7411764705882353),
    "laplace-t5-base-just-smiles-average": (
        0.8392156862745098,
        0.15294117647058825,
        0.1568627450980392,
    ),
    "laplace-gpt2-medium-just-smiles-average": (
        0.17254901960784313,
        0.6274509803921569,
        0.17254901960784313,
    ),
    "laplace-llama-2-7b-just-smiles-average": (
        1.0,
        0.4980392156862745,
        0.054901960784313725,
    ),
    "laplace-t5-base-chem-just-smiles-average": (
        0.12156862745098039,
        0.4666666666666667,
        0.7058823529411765,
    ),
}

PROBLEM2TITLE = {
    "redox-mer": r"Redoxmer ($\downarrow$)",
    "solvation": r"Solvation ($\downarrow$)",
    "kinase": r"Kinase ($\downarrow$)",
    "laser": r"Laser ($\uparrow$)",
    "pce": r"Photovoltaics ($\uparrow$)",
    "photoswitch": r"Photoswitches ($\uparrow$)",
}
PROBLEM2LABEL = {
    "redox-mer": "Redox Potential",
    "solvation": "Solvation Energy",
    "kinase": "Docking Score",
    "laser": "Strength",
    "pce": "PCE",
    "photoswitch": "Wavelength",
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
RANDSEEDS = [1, 2, 3, 4, 5]


FIG_WIDTH = 1
FIG_HEIGHT = 0.225
rc_params, fig_width, fig_height = plot_utils.get_mpl_rcParams(
    FIG_WIDTH, FIG_HEIGHT, single_col=False
)
plt.rcParams.update(rc_params)

fig, axs = plt.subplots(2, len(PROBLEMS) // 2, sharex=True, constrained_layout=True)
fig.set_size_inches(fig_width, fig_height)

for i, (problem, ax) in enumerate(zip(PROBLEMS, axs.flatten())):
    if problem is None:
        continue

    # Plot optimal val
    targets = torch.load(f"data/cache/{problem}/fingerprints_targets.bin")
    targets = torch.tensor(targets).flatten()
    best_val = targets.max() if IS_MAX[problem] else -targets.max()
    print(best_val)
    ax.axhline(best_val, c="k", ls="dashed", zorder=1000)

    # Plot methods
    for feature_name, real_feature_name in zip(FEATURE_NAMES, REAL_FEATURE_NAMES):
        for method in METHODS:
            if (
                method == "random" or method == "gp"
            ) and feature_name != "fingerprints":
                continue

            path = f"results/{problem}/fixed/{method}/{feature_name}"
            if not os.path.exists(path):
                os.makedirs(path)

            trace_best_y = np.stack(
                [
                    np.load(f"{path}/trace_best_y_10_{args.acqf}_{rs}.npy")
                    for rs in RANDSEEDS
                ]
            )
            mean = np.mean(trace_best_y, axis=0)[1:]  # Over randseeds
            sem = st.sem(trace_best_y, axis=0)[1:]  # Over randseeds
            T = np.arange(len(mean)) + 1

            c = FEATURE2COLOR[f"{method}-{feature_name}"]
            ax.plot(
                T,
                mean,
                color=c,
                label=f"{METHOD2LABEL[method]}-{FEATURE2LABEL[real_feature_name]}",
            )
            ax.fill_between(T, mean - sem, mean + sem, color=c, alpha=0.25)

    title = f"{PROBLEM2TITLE[problem]}"
    if args.prompt_type_in_title:
        title += " - Prompt: {args.prompt_type}"
    ax.set_title(title)
    if i >= 3:  # Only for the bottom row
        ax.set_xlabel(r"$t$")
    ax.set_ylabel(PROBLEM2LABEL[problem])
    ax.set_xlim(1, 100)

    handles, labels = ax.get_legend_handles_labels()

    # if i == 0:
    #     ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.5), ncols=8, frameon=True)

fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0, 1.065, 1, 0.005),
    mode="expand",
    ncols=8,
)

# Save results
path = "../paper/figs"
if not os.path.exists(path):
    os.makedirs(path)

fname = f'fixed_feat_{args.prompt_type}{"-average" if not args.dont_average_llm_feature else ""}'
plt.savefig(f"{path}/{fname}.pdf", bbox_inches="tight")