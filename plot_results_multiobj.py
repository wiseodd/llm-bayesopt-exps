import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import torch
from botorch.utils.multi_objective.hypervolume import (Hypervolume,
                                                       infer_reference_point)
from botorch.utils.multi_objective.pareto import is_non_dominated

import utils.plots as plot_utils

parser = argparse.ArgumentParser()
parser.add_argument("--poster", default=False, action="store_true")
args = parser.parse_args()

PROBLEMS = ["redox-mer", "laser"]
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
    "t5-base-chem",
]
# FEATURE_NAMES_LLM = [f'{n}-just-smiles' for n in REAL_FEATURE_NAMES_LLM]
FEATURE_NAMES_LLM = [f"{n}-average" for n in REAL_FEATURE_NAMES_LLM]
FEATURE_NAMES = FEATURE_NAMES_BASE + FEATURE_NAMES_LLM
REAL_FEATURE_NAMES = FEATURE_NAMES_BASE + REAL_FEATURE_NAMES_LLM
METHODS = [
    # 'random',
    # 'gp',
    "laplace"
]
PROBLEM2TITLE = {
    "redox-mer": r"Multi-Redox ($\uparrow$)",
    "laser": r"Multi-Laser ($\uparrow$)",
}
PROBLEM2LABELS = {
    "redox-mer": ["Redox Potential", "Solvation Energy"],
    "laser": ["Strength", "Electronic Gap"],
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
    "laplace-t5-base-average": (
        0.8392156862745098,
        0.15294117647058825,
        0.1568627450980392,
    ),
    "laplace-gpt2-medium-average": (
        0.17254901960784313,
        0.6274509803921569,
        0.17254901960784313,
    ),
    "laplace-llama-2-7b-average": (1.0, 0.4980392156862745, 0.054901960784313725),
    "laplace-t5-base-chem-average": (
        0.12156862745098039,
        0.4666666666666667,
        0.7058823529411765,
    ),
}
RANDSEEDS = [1, 2, 3, 4, 5]


FIG_WIDTH = 1
FIG_HEIGHT = 0.225
rc_params, fig_width, fig_height = plot_utils.get_mpl_rcParams(
    FIG_WIDTH, FIG_HEIGHT, single_col=True, poster=args.poster
)
plt.rcParams.update(rc_params)

# Scalarized objective over time
# ------------------------------
fig, axs = plt.subplots(len(PROBLEMS), 1, sharex=True, constrained_layout=True)
fig.set_size_inches(fig_width, fig_height)

for i, (problem, ax) in enumerate(
    zip(PROBLEMS, axs.flatten() if isinstance(axs, np.ndarray) else [axs])
):
    if problem is None:
        continue

    # Plot optimal val
    targets = torch.load(
        f"data/cache/multiobj/{problem}/fingerprints_multi_targets.bin"
    )  # (n_data, n_obj)
    # Get reference point for hypervolume computation
    Y_pareto = targets[is_non_dominated(targets)]
    ref_point = infer_reference_point(Y_pareto)
    hv = Hypervolume(ref_point)
    max_hypervolume = hv.compute(Y_pareto)
    ax.axhline(max_hypervolume, c="k", ls="dashed", zorder=1000)

    # Plot methods
    for feature_name, real_feature_name in zip(FEATURE_NAMES, REAL_FEATURE_NAMES):
        for method in METHODS:
            if (
                method == "random" or method == "gp"
            ) and feature_name != "fingerprints":
                continue

            path = f"results/multiobj/{problem}/{method}/{feature_name}"
            if not os.path.exists(path):
                os.makedirs(path)

            trace_best_y = np.stack(
                [
                    np.load(f"{path}/trace_hypervolume_10_ts_{rs}.npy")
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
    ax.set_title(title)
    if i >= 1:  # Only for the bottom row
        ax.set_xlabel(r"$t$")
    ax.set_ylabel("Hypervolume")
    ax.set_xlim(1, 100)

handles, labels = axs[-1].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0, 1.065, 1, 0.05),
    mode="expand",
    ncols=8,
)

# Save results
path = "../poster/figs" if args.poster else "../paper/figs"
if not os.path.exists(path):
    os.makedirs(path)

fname = "multiobj_fixed_feat_hypervolume"
fname += "_poster" if args.poster else ""
plt.savefig(f"{path}/{fname}.pdf", bbox_inches="tight")
