import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import utils.plots as plot_utils
import os


PROBLEMS = [
    "redox-mer",
    # 'solvation', 'kinase',
    "laser",
    # 'pce',
    "photoswitch",
]
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
    # 'molformer',
]
REAL_FEATURE_NAMES_LLM = [
    # 'gpt2-medium',
    # 'llama-2-7b',
    "t5-base",
    "t5-base-chem",
]
REAL_FEATURE_NAMES = FEATURE_NAMES_BASE + REAL_FEATURE_NAMES_LLM
METHODS = [
    # 'random',
    # 'gp',
    "laplace"
]
PROBLEM2TITLE = {
    "redox-mer": "Redoxmer (1407)",
    "solvation": "Solvation",
    "kinase": "Kinase",
    "laser": "Laser (10000)",
    "pce": "Photovoltaics",
    "photoswitch": "Photoswitches (392)",
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

FILENAMES = [
    "timing_train_10_ei_all_layer",
    "timing_preds_10_ei_all_layer",
]
YLABELS = [
    "Training Time (s)",
    "Pred. Time (s)",
]


FIG_WIDTH = 1
FIG_HEIGHT = 0.3
rc_params, fig_width, fig_height = plot_utils.get_mpl_rcParams(
    FIG_WIDTH, FIG_HEIGHT, single_col=False
)
plt.rcParams.update(rc_params)

fig, axs = plt.subplots(2, 3, sharex=True, sharey=False, constrained_layout=True)
fig.set_size_inches(fig_width, fig_height)

for row_idx, (row, ylabel, fname) in enumerate(zip(axs, YLABELS, FILENAMES)):
    for col_idx, (problem, ax) in enumerate(zip(PROBLEMS, row)):
        if problem is None:
            continue

        method = "laplace"

        # Plot finetuned
        for i, real_feature_name in enumerate(REAL_FEATURE_NAMES):
            path = f"results/{problem}/finetuning/{real_feature_name}"

            MAX_T = 100
            trace_best_y = np.zeros(shape=[len(RANDSEEDS), MAX_T])
            for i, rs in enumerate(RANDSEEDS):
                arr = np.load(f"{path}/{fname}_{rs}.npy")[:MAX_T]
                trace_best_y[i, : len(arr)] = arr

            mean = np.mean(trace_best_y, axis=0)[1:]  # Over randseeds
            sem = st.sem(trace_best_y, axis=0)[1:]  # Over randseeds

            idx_last = MAX_T - 1
            T = np.arange(1, idx_last + 1)
            c = FEATURE2COLOR[f"{method}-{real_feature_name}"]
            print(T.shape, mean[:idx_last].shape)
            ax.plot(
                T,
                mean[:idx_last],
                c=c,
                ls="dotted",
                label=f"{METHOD2LABEL[method]}-{FEATURE2LABEL[real_feature_name]}-FT",
            )
            ax.fill_between(
                T, (mean - sem)[:idx_last], (mean + sem)[:idx_last], color=c, alpha=0.25
            )

        if row_idx == 0:
            ax.set_title(f"{PROBLEM2TITLE[problem]}")

        if row_idx == axs.shape[0] - 1:
            ax.set_xlabel(r"$t$")

        if col_idx == 0:
            ax.set_ylabel(ylabel)

        # ax.set_yticks(np.arange(0, 100 + 1, 20))
        ax.set_xlim(1, 100)
        # ax.set_ylim(0, 50)


handles, labels = axs.flatten()[-1].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0, 1, 1, 0.075),
    ncols=4,
    handlelength=2,
)

# Save results
path = "../paper/figs"
if not os.path.exists(path):
    os.makedirs(path)

fname = "finetuning_timing"
plt.savefig(f"{path}/{fname}.pdf", bbox_inches="tight")
