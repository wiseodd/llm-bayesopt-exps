import argparse
import os
import warnings

import numpy as np
import scipy.stats as st
import torch

warnings.filterwarnings("error")
from collections import defaultdict

import matplotlib.pyplot as plt
import tqdm

import utils.plots as plot_utils

parser = argparse.ArgumentParser()
parser.add_argument("--format", default="markdown", choices=["markdown", "latex"])
parser.add_argument("--poster", default=False, action="store_true")
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
FEATURE_NAMES_BASE = [
    # "fingerprints",
    # "molformer"
]
FEATURE_NAMES_LLM = [
    "t5-base",
    # "gpt2-medium",
    # "llama-2-7b",
    "t5-base-chem",
]
FEATURE_NAMES_LLM_PROMPT = [f + "-just-smiles-average" for f in FEATURE_NAMES_LLM]
METHODS = [
    # "random",
    # "gp",
    "laplace",
]
RANDSEEDS = [1, 2, 3, 4, 5]

PROBLEM2TITLE = {
    "redox-mer": r"Redoxmer",
    "solvation": r"Solvation",
    "kinase": r"Kinase",
    "laser": r"Laser",
    "pce": r"Photovoltaics",
    "photoswitch": r"Photoswitches",
}
FEATURE2LABEL = {
    "fingerprints": "FP",
    "molformer": "MolFormer",
    "gpt2-medium": "GPT2-M",
    "llama-2-7b": "LL2-7B",
    "t5-base": "T5",
    "t5-base-chem": "T5-Chem",
}
METHOD2LABEL = {
    "random-fingerprints": "RS",
    "gp-fingerprints": "GP-FP",
    "laplace-fingerprints": "LA-FP",
    "laplace-molformer": "LA-MolFormer",
    "laplace-t5-base": "LA-T5",
    "laplace-gpt2-medium": "LA-GPT2-M",
    "laplace-llama-2-7b": "LA-LL2-7B",
    "gp-t5-base-chem": "GP-T5-Chem",
    "laplace-t5-base-chem": "LA-T5-Chem",
    # Finetuning
    "laplace-t5-base-ft": "LA-T5-FT",
    "laplace-t5-base-chem-ft": "LA-T5-Chem-FT",
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
    "gp-t5-base-chem": (0.09019607843137255, 0.7450980392156863, 0.8117647058823529),
    "laplace-t5-base-chem": (
        0.12156862745098039,
        0.4666666666666667,
        0.7058823529411765,
    ),
    # Finetuning
    "laplace-t5-base-ft": (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
    "laplace-t5-base-chem-ft": (
        0.12156862745098039,
        0.4666666666666667,
        0.7058823529411765,
    ),
}


def compute_gap(trace_best_y, y_best, t):
    y_0 = trace_best_y[0]
    y_t = trace_best_y[t]
    try:
        return np.nan_to_num((y_t - y_0) / (y_best - y_0), nan=1.0)
    except RuntimeWarning:
        # print(y_0, y_t, y_best)
        return None


all_res_mean = defaultdict(list)
all_res_sem = defaultdict(list)

times = range(0, 100, 5)
for t in tqdm.tqdm(times):
    res = defaultdict(list)

    for i, problem in enumerate(PROBLEMS):
        # Optimal val
        targets = torch.load(f"data/cache/{problem}/fingerprints_targets.bin")
        targets = torch.tensor(targets).flatten()
        best_val = targets.max().item() if IS_MAX[problem] else -targets.max().item()

        # Plot methods
        for feature_name, real_feature_name in zip(
            FEATURE_NAMES_BASE + FEATURE_NAMES_LLM_PROMPT,
            FEATURE_NAMES_BASE + FEATURE_NAMES_LLM,
        ):
            for method in METHODS:
                if method == "random" and feature_name != "fingerprints":
                    continue
                if method == "gp" and feature_name not in [
                    "fingerprints",
                    "t5-base-chem-just-smiles-average",
                ]:
                    continue

                # Fixed
                path = f"results/{problem}/fixed/{method}/{feature_name}"
                for rs in RANDSEEDS:
                    # Hotfix: ignore the 0 index; it's ignored in `run_*.py`
                    trace_best_y = np.load(
                        f"{path}/trace_best_y_10_ts_{rs}.npy", allow_pickle=True
                    )[1:]

                    gap = compute_gap(trace_best_y, best_val, t=t)
                    if gap is not None:
                        res[f"{method}-{real_feature_name}"].append(gap)

                # Finetuning
                path = f"results/{problem}/finetuning/{real_feature_name}"
                for rs in RANDSEEDS:
                    # Hotfix: ignore the 0 index; it's ignored in `run_*.py`
                    trace_best_y = np.load(
                        f"{path}/just-smiles_trace_best_y_10_ts_all_layer_{rs}.npy",
                        allow_pickle=True,
                    )[1:]

                    gap = compute_gap(trace_best_y, best_val, t=t)
                    if gap is not None:
                        res[f"{method}-{real_feature_name}-ft"].append(gap)

    for method, gaps in res.items():
        all_res_mean[method].append(np.mean(gaps))
        all_res_sem[method].append(st.sem(gaps))


FIG_WIDTH = 1
FIG_HEIGHT = 0.15
rc_params, fig_width, fig_height = plot_utils.get_mpl_rcParams(
    FIG_WIDTH, FIG_HEIGHT, single_col=True, poster=args.poster
)
plt.rcParams.update(rc_params)

fig, ax = plt.subplots(1, 1, constrained_layout=True)
fig.set_size_inches(fig_width, fig_height)

for method in all_res_mean.keys():
    mean, sem = np.array(all_res_mean[method]), np.array(all_res_sem[method])
    linestyle = "dashed" if "ft" in method else "solid"
    ax.plot(
        times,
        mean,
        label=METHOD2LABEL[method],
        color=FEATURE2COLOR[method],
        linestyle=linestyle,
    )
    ax.fill_between(
        times, mean - sem, mean + sem, alpha=0.1, color=FEATURE2COLOR[method]
    )
    ax.legend()

ax.set_title("Average GAP Across Datasets and Random Seeds")
ax.set_xlim(0, 95)
ax.set_ylim(0, 1)
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"GAP ($\uparrow$)")

path = "../poster/figs" if args.poster else "../paper/figs"
fname = "gap_finetuning"
fname += "_poster" if args.poster else ""
plt.savefig(f"{path}/{fname}.pdf")
