import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st

import utils.plots as plot_utils

parser = argparse.ArgumentParser()
parser.add_argument(
    "--prompt_type",
    choices=["just-smiles", "completion", "single-number", "naive"],
    default="just-smiles",
)
parser.add_argument("--dont_average_llm_feature", default=False, action="store_true")
parser.add_argument("--acqf", choices=["ei", "ucb", "ts"], default="ts")
parser.add_argument("--prompt_type_in_title", default=False, action="store_true")
parser.add_argument("--poster", default=False, action="store_true")
args = parser.parse_args()

PROBLEMS = [
    "redox-mer",
]
IS_MAX = {
    "redox-mer": False,
    "solvation": False,
    "kinase": False,
    "laser": True,
    "pce": True,
    "photoswitch": True,
}
REAL_FEATURE_NAMES_LLM = [
    "t5-base-chem-just-smiles",
    "llama-2-7b-just-smiles",
    "llama-2-7b-completion",
    "gpt4-completion",
]
# FEATURE_NAMES_LLM = [f'{n}-{args.prompt_type}' for n in REAL_FEATURE_NAMES_LLM]
FEATURE_NAMES_LLM = REAL_FEATURE_NAMES_LLM
FEATURE_NAMES_LLM = [
    f'{n}{"-average" if not args.dont_average_llm_feature else ""}'
    for n in FEATURE_NAMES_LLM
]
FEATURE_NAMES = FEATURE_NAMES_LLM
REAL_FEATURE_NAMES = REAL_FEATURE_NAMES_LLM
METHODS = ["laplace", "bolift"]
FEATURE2COLOR = {
    "bolift-gpt4-completion-average": (
        0.09019607843137255,
        0.7450980392156863,
        0.8117647058823529,
    ),
    "bolift-llama-2-7b-completion-average": (
        0.7372549019607844,
        0.7411764705882353,
        0.13333333333333333,
    ),
    # 'laplace-llama-2-7b-completion-average': (1.0, 0.4980392156862745, 0.054901960784313725),
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
}
PROBLEM2LABEL = {
    "redox-mer": "Redox Potential",
}
METHOD2LABEL = {"laplace": "LA", "bolift": "BO-LIFT"}
FEATURE2LABEL = {
    "llama-2-7b-completion": "LL2-7B",
    "llama-2-7b-just-smiles": "LL2-7B",
    "t5-base-chem-just-smiles": "T5-Chem",
    "gpt4-completion": "GPT4",
}
RANDSEEDS = [1, 2, 3, 4, 5]


FIG_WIDTH = 1
FIG_HEIGHT = 0.15
rc_params, fig_width, fig_height = plot_utils.get_mpl_rcParams(
    FIG_WIDTH, FIG_HEIGHT, single_col=True, poster=args.poster
)
plt.rcParams.update(rc_params)

# fig, axs = plt.subplots(2, len(PROBLEMS) // 2, sharex=True, constrained_layout=True)
fig, ax = plt.subplots(1, len(PROBLEMS), sharex=True, constrained_layout=True)
fig.set_size_inches(fig_width, fig_height)

# for i, (problem, ax) in enumerate(zip(PROBLEMS, axs.flatten())):
for i, problem in enumerate(PROBLEMS):
    # Plot optimal val
    # targets = torch.load(f'data/cache/{problem}/fingerprints_targets.bin')
    dataset = pd.read_csv(f'data/random_subset_200/{problem.replace("-", "_")}.csv')
    # targets = torch.tensor(targets).flatten()
    # dataset['Ered_orig'] = dataset['Ered']
    OBJ_COL = "Ered"
    best_val = dataset[OBJ_COL].max() if IS_MAX[problem] else dataset[OBJ_COL].min()

    # best_val = targets.max() if IS_MAX[problem] else -targets.max()
    # best_val=1.6
    print(best_val)
    ax.axhline(best_val, c="k", ls="dashed", zorder=1000)

    # Plot methods
    for feature_name, real_feature_name in zip(FEATURE_NAMES, REAL_FEATURE_NAMES):
        for method in METHODS:
            if method == "bolift" and real_feature_name not in [
                "gpt4-completion",
                "llama-2-7b-completion",
            ]:
                continue

            if method == "laplace" and real_feature_name == "llama-2-7b-completion":
                continue

            path = f"results/icl_experiments/{problem}/fixed/{method}/{feature_name}"
            print(path, f"{method}-{feature_name}")
            if f"{method}-{feature_name}" not in FEATURE2COLOR:
                continue
            if not os.path.exists(path):
                os.makedirs(path)

            trace_best_y = np.stack(
                [
                    np.load(f"{path}/trace_best_y_5_{args.acqf}_{rs}.npy")
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
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(PROBLEM2LABEL[problem])
    ax.set_xlim(1, 15)

handles, labels = ax.get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0, 1.065, 1, 0.09),
    mode="expand",
    ncols=8,
)

# Save results
path = "../poster/figs" if args.poster else "../paper/figs"
if not os.path.exists(path):
    os.makedirs(path)

fname = f'fixed_feat_vs_icl_200N_{args.prompt_type}{"-average" if not args.dont_average_llm_feature else ""}'
fname += "_poster" if args.poster else ""
plt.savefig(f"{path}/{fname}.pdf", bbox_inches="tight")
