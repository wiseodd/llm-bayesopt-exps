import numpy as np
import pandas as pd
import torch
import torch.utils.data as data_utils
import tqdm
import argparse
import sys
import os
import time
from bayesopt.laplace_botorch import LaplaceBoTorch
from bayesopt.acqf import ucb, thompson_sampling_multivariate, scalarize
from utils import helpers
from sklearn.utils import shuffle as skshuffle
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.hypervolume import infer_reference_point, Hypervolume

parser = argparse.ArgumentParser()
parser.add_argument("--problem", choices=["redox-mer", "laser"], default="redox-mer")
parser.add_argument("--method", choices=["random", "laplace"], default="laplace")
parser.add_argument(
    "--feature_type",
    choices=[
        "fingerprints",
        "molformer",
        "roberta-large",
        "t5-base",
        "t5-large",
        "t5-base-chem",
        "gpt2-medium",
        "gpt2-large",
        "llama-2-7b",
    ],
    default="t5-base-chem",
)
parser.add_argument(
    "--feature_reduction", choices=["default", "average"], default="average"
)
parser.add_argument("--acqf", choices=["ucb", "ts"], default="ts")
parser.add_argument("--n_init_data", type=int, default=10)
parser.add_argument("--exp_len", type=int, default=200)
parser.add_argument("--cuda", default=False, action="store_true")
parser.add_argument("--randseed", type=int, default=1)
args = parser.parse_args()

np.random.seed(args.randseed)
torch.manual_seed(args.randseed)

if args.cuda:
    if not torch.cuda.is_available():
        print("No CUDA detected!")
        sys.exit(1)
DEVICE = "cuda" if args.cuda else "cpu"

if args.problem == "redox-mer":
    dataset = pd.read_csv("data/redox_mer.csv")
    OBJ_COLS = ["Ered", "Gsol"]
    MAXIMIZATIONS = [False, False]
elif args.problem == "laser":
    dataset = pd.read_csv("data/laser_multi10k.csv.gz")
    SMILES_COL = "SMILES"
    OBJ_COLS = ["Fluorescence Oscillator Strength", "Electronic Gap"]
    MAXIMIZATIONS = [True, True]
else:
    print("Invalid test function!")
    sys.exit(1)

# Dataset
PROBLEM_NAME = args.problem
CACHE_PATH = f"data/cache/multiobj/{PROBLEM_NAME}/"
if args.feature_type not in ["fingerprints", "molformer"]:  # LLM features
    FEATURE_NAME = args.feature_type
    FEATURE_NAME += "-average" if args.feature_reduction == "average" else ""
else:
    FEATURE_NAME = args.feature_type
features = torch.load(CACHE_PATH + f"{FEATURE_NAME}_multi_feats.bin")
targets = torch.load(CACHE_PATH + f"{FEATURE_NAME}_multi_targets.bin")
features, targets = skshuffle(features, targets, random_state=args.randseed)
feature_dim = features[0].shape[-1]
scalarized_ground_truth_max = scalarize(targets).max()
scalarized_ground_truth_max_idx = scalarize(targets).argmax()
ground_truth_maxs = targets[scalarized_ground_truth_max_idx]
assert len(ground_truth_maxs) == len(OBJ_COLS)

# Get reference point for hypervolume computation
Y_pareto = targets[is_non_dominated(targets)]
ref_point = infer_reference_point(Y_pareto)
hv = Hypervolume(ref_point)
max_hypervolume = hv.compute(Y_pareto)

# Convert to list so that we can pop
targets = list(targets)

print()
print(f"Ground-truth best: {ground_truth_maxs}")
print(f"Scalarized ground-truth max: {scalarized_ground_truth_max:.3f}")

print()
if args.feature_type in ["fingerprints", "molformer"]:
    print(
        f"Multiobj --- Test Function: {PROBLEM_NAME}; Feature Type: {args.feature_type}; Randseed: {args.randseed}"
    )
else:
    print(
        f"Multiobj --- Test Function: {PROBLEM_NAME}; Foundation LLM: {args.feature_type}; Reduction: {args.feature_reduction}; Randseed: {args.randseed}"
    )
print(
    "--------------------------------------------------------------------------------------------------------------"
)
print()

train_x, train_y = [], []
while len(train_x) < args.n_init_data:
    idx = np.random.randint(len(features))
    train_x.append(features.pop(idx))
    train_y.append(targets.pop(idx))
train_x, train_y = torch.stack(train_x), torch.stack(train_y)
assert train_x.shape == (args.n_init_data, features[0].shape[0])
assert train_y.shape == (args.n_init_data, len(OBJ_COLS))

if args.method == "laplace":

    def get_net():
        activation = (
            torch.nn.Tanh if args.feature_type == "fingerprints" else torch.nn.ReLU
        )
        return torch.nn.Sequential(
            torch.nn.Linear(feature_dim, 50),
            activation(),
            torch.nn.Linear(50, 50),
            activation(),
            torch.nn.Linear(50, len(OBJ_COLS)),
        )

    model = LaplaceBoTorch(
        get_net,
        train_x,
        train_y,
        noise_var=0.01 if args.problem == "laser" else 0.001,
        hess_factorization="kron" if args.feature_type != "llama-2-7b" else "diag",
    )
    model = model.to(DEVICE) if model is not None else model
else:
    model = None

scalarized_best_y = scalarize(train_y).max().item()
best_y_idx = scalarize(train_y).argmax()
best_y = train_y[best_y_idx]
pbar = tqdm.trange(args.exp_len)
pbar.set_description(f"[Best scal'd f(x) = {scalarized_best_y:.3f}]")

trace_best_y = (args.exp_len + 1) * [
    [
        helpers.y_transform(gt, maximization)
        for gt, maximization in zip(ground_truth_maxs, MAXIMIZATIONS)
    ]
]
trace_scalarized_best_y = (args.exp_len + 1) * [scalarized_ground_truth_max]
trace_hypervolume = (args.exp_len + 1) * [max_hypervolume]
trace_timing = [0.0] * (args.exp_len + 1)

for i in pbar:
    if DEVICE == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
    else:
        start = time.time()

    if args.method == "random":
        idx = np.random.randint(len(features))
        new_x = features.pop(idx)
        new_y = targets.pop(idx)

        train_y = torch.cat([train_y, new_y.unsqueeze(0)], dim=0)

        # Evaluate hypervolume to measure performance
        volume = hv.compute(train_y[is_non_dominated(train_y)])
    else:
        dataloader = data_utils.DataLoader(
            data_utils.TensorDataset(torch.stack(features), torch.stack(targets)),
            batch_size=256,
            shuffle=False,
        )

        preds, uncerts, labels = [], [], []
        acq_vals = []
        for x, y in dataloader:
            f_mean, f_cov = model.posterior(x, return_mean_cov_only=True)
            f_var = torch.diagonal(f_cov, dim1=-2, dim2=-1)

            if args.acqf == "ucb":
                acq_vals.append(scalarize(ucb(f_mean, f_var)))
            else:
                samples = thompson_sampling_multivariate(f_mean, f_cov)
                acq_vals.append(scalarize(samples))

            preds.append(f_mean)
            uncerts.append(f_var.sqrt())
            labels.append(y)

        acq_vals = torch.cat(acq_vals, dim=0).cpu().squeeze()
        preds, uncerts, labels = (
            torch.cat(preds, dim=0).cpu(),
            torch.cat(uncerts, dim=0).cpu(),
            torch.cat(labels, dim=0),
        )
        test_loss = torch.nn.MSELoss()(preds, labels).item()

        # Pick a molecule (a row in the current dataset) that maximizes the acquisition
        idx_best = torch.argmax(acq_vals).item()
        new_x, new_y = features.pop(idx_best), targets.pop(idx_best)

    # Update the current best y
    if scalarize(new_y).item() > scalarized_best_y:
        best_y = new_y
        scalarized_best_y = scalarize(new_y).item()

    if args.method == "random":
        pbar.set_description(
            f"[Best scal'd f(x) = {scalarize(best_y).item():.3f}, "
            + f"curr scal'd f(x) = {scalarize(new_y).item():.3f}], "
            + f"curr hypervolume = {volume:.3f}, "
        )
    else:
        # Update surrogate
        model = model.condition_on_observations(new_x.unsqueeze(0), new_y.unsqueeze(0))

        # Evaluate hypervolume to measure performance
        volume = hv.compute(model.train_Y[is_non_dominated(model.train_Y)])

        pbar.set_description(
            f"[Best scal'd f(x) = {scalarize(best_y).item():.3f}, "
            + f"curr scal'd f(x) = {scalarize(new_y).item():.3f}, "
            + f"curr hypervolume = {volume:.3f}, "
            + f"test MSE = {test_loss:.3f}]"
        )

    # Housekeeping
    if DEVICE == "cuda":
        end.record()
        torch.cuda.synchronize()
        timing = start.elapsed_time(end) / 1000
    else:
        timing = start - time.time()

    trace_best_y[i + 1] = [
        helpers.y_transform(by, maximization)
        for by, maximization in zip(best_y, MAXIMIZATIONS)
    ]
    trace_scalarized_best_y[i + 1] = scalarized_best_y
    trace_hypervolume[i + 1] = volume
    trace_timing[i + 1] = timing

    # Early stopping if we already got the max
    # if scalarized_best_y >= scalarized_ground_truth_max:
    #     break
    if volume >= max_hypervolume:
        break

# Save results
path = f"results/multiobj/{PROBLEM_NAME}/{args.method}/{FEATURE_NAME}"
if not os.path.exists(path):
    os.makedirs(path)

# np.save(f'{path}/trace_best_y_{args.n_init_data}_{args.acqf}_{args.randseed}.npy', trace_best_y)
# np.save(f'{path}/trace_scalarized_best_y_{args.n_init_data}_{args.acqf}_{args.randseed}.npy', trace_scalarized_best_y)
np.save(
    f"{path}/trace_hypervolume_{args.n_init_data}_{args.acqf}_{args.randseed}.npy",
    trace_hypervolume,
)
# np.save(f'{path}/trace_timing_{args.n_init_data}_{args.acqf}_{args.randseed}.npy', trace_timing)
