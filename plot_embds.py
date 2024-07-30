import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import argparse
import umap
import utils.plots as plot_utils
from utils import helpers
import os
from sklearn.manifold import TSNE

parser = argparse.ArgumentParser()
parser.add_argument("--method", choices=["umap", "tsne"], default="umap")
parser.add_argument("--average", default=False, action="store_true")
parser.add_argument("--problem", choices=["redox-mer"], default="redox-mer")
args = parser.parse_args()

np.random.seed(1)
torch.manual_seed(1)

AVERAGE_STR = "average" if args.average else ""


def get_embeddings(feature_type, prompt_type=None):
    # Load data
    if feature_type not in ["fingerprints", "gnn"]:  # LLM features
        feature_name = f"{feature_type}-{prompt_type}-{AVERAGE_STR}"
    else:
        feature_name = feature_type

    cache_path = f"data/cache/{args.problem}/"
    features = torch.load(
        cache_path + f"{feature_name}_feats.bin"
    )  # list of feature vectors
    targets = torch.load(cache_path + f"{feature_name}_targets.bin")
    features, targets = (
        torch.stack(features).numpy(),
        torch.stack(targets).numpy().flatten(),
    )

    # Transform targets to their original values
    if args.problem in ["redox-mer"]:
        MAXIMIZATION = False
    else:
        MAXIMIZATION = True
    targets = helpers.y_transform(targets, MAXIMIZATION)

    # UMAP
    dim_reducer = umap.UMAP() if args.method == "umap" else TSNE()
    embedding = dim_reducer.fit_transform(features)
    assert len(embedding.shape) == 2 and embedding.shape[-1] == 2
    return embedding, targets


def plot(ax, feature_type, embeddings, targets):
    cmap = sns.cubehelix_palette(rot=-0.2, as_cmap=True)
    scatter = ax.scatter(embeddings[:, 0], embeddings[:, 1], s=10, c=targets, cmap=cmap)

    # Mark the optimum
    opt_idx = targets.argmin()
    opt_x, opt_y = embeddings[opt_idx, 0], embeddings[opt_idx, 1]
    ax.scatter(opt_x, opt_y, c="red", s=10)

    return scatter


# Plotting
FIG_WIDTH = 0.24
FIG_HEIGHT = 0.165
rc_params, fig_width, fig_height = plot_utils.get_mpl_rcParams(FIG_WIDTH, FIG_HEIGHT)
plt.rcParams.update(rc_params)

fig, ax = plt.subplots(1, 1, constrained_layout=True)
fig.set_size_inches(fig_width, fig_height)

embeddings, targets = get_embeddings("fingerprints")
scatter = plot(ax, "fingerprints", embeddings, targets)
fig.colorbar(scatter)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("fingerprints")

path = f"figs/{args.problem}/dim_reduction"
if not os.path.exists(path):
    os.makedirs(path)
plt.savefig(
    f"../paper/figs/{args.problem}/dim_reduction/{args.method}_fingerprints.pdf"
)

# LLM features
FIG_WIDTH = 0.725
FIG_HEIGHT = 0.45
rc_params, fig_width, fig_height = plot_utils.get_mpl_rcParams(FIG_WIDTH, FIG_HEIGHT)
plt.rcParams.update(rc_params)

fig, axs = plt.subplots(3, 3, constrained_layout=True, sharex=True, sharey=True)
fig.set_size_inches(fig_width, fig_height)

for i, prompt_type in enumerate(["single-number", "completion", "just-smiles"]):
    for j, feature_type in enumerate(["gpt2-medium", "llama-2-7b", "t5-base-chem"]):
        embeddings, targets = get_embeddings(feature_type, prompt_type)
        plot(axs[i, j], feature_type, embeddings, targets)

        axs[i, j].set_xticks([])
        axs[i, j].set_yticks([])

        print(i, j, prompt_type, feature_type)
        if i == 0:
            axs[i, j].set_title(feature_type)
        if j == 0:
            axs[i, j].set_ylabel(prompt_type)

plt.savefig(
    f"../paper/figs/{args.problem}/dim_reduction/{args.method}_llm_feats_{AVERAGE_STR}.pdf"
)
