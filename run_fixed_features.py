import numpy as np
import pandas as pd
import torch
import torch.utils.data as data_utils
import tqdm
import argparse
import sys
import os
import time
from gpytorch.kernels import ScaleKernel, MaternKernel
from bayesopt.kernels.tanimoto_kernel import TanimotoKernel
from gpytorch.likelihoods import GaussianLikelihood
from bayesopt.gp_baselines import MLLGP
from bayesopt.laplace_botorch import LaplaceBoTorch
from bayesopt.acqf import ucb, ei, thompson_sampling
from utils import helpers
from sklearn.utils import shuffle as skshuffle
from sklearn.preprocessing import StandardScaler

parser = argparse.ArgumentParser()
parser.add_argument(
    "--problem",
    choices=["redox-mer", "solvation", "kinase", "laser", "pce", "photoswitch"],
    default="redox-mer",
)
parser.add_argument("--method", choices=["random", "gp", "laplace"])
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
    default="gpt2-medium",
)
parser.add_argument(
    "--feature_reduction", choices=["default", "average"], default="average"
)
parser.add_argument(
    "--prompt_type",
    choices=["single-number", "just-smiles", "completion", "naive"],
    default="just-smiles",
)
parser.add_argument("--acqf", choices=["ei", "ucb", "ts"], default="ts")
parser.add_argument("--n_init_data", type=int, default=10)
parser.add_argument("--exp_len", type=int, default=200)
parser.add_argument("--cuda", default=False, action="store_true")
parser.add_argument("--randseed", type=int, default=1)
parser.add_argument("--iupac", default=False, action="store_true")
parser.add_argument("--normalize_y", default=False, action="store_true")
parser.add_argument("--run_subset_only", default=False, action="store_true")
args = parser.parse_args()

if args.iupac and args.problem not in ["redox-mer", "solvation"]:
    print("IUPAC option is only available for redox-mer and solvation")
    sys.exit(1)

np.random.seed(args.randseed)
torch.manual_seed(args.randseed)

if args.cuda:
    if not torch.cuda.is_available():
        print("No CUDA detected!")
        sys.exit(1)
DEVICE = "cuda" if args.cuda else "cpu"

if args.problem == "redox-mer":
    if args.iupac:
        if args.run_subset_only:
            dataset = pd.read_csv("data/random_subset_200/redox_mer_with_iupac.csv.gz")
        else:
            dataset = pd.read_csv("data/redox_mer_with_iupac.csv.gz")
    else:
        dataset = pd.read_csv("data/random_subset_200/redox_mer.csv")
    OBJ_COL = "Ered"
    MAXIMIZATION = False
elif args.problem == "solvation":
    if args.iupac:
        if args.run_subset_only:
            dataset = pd.read_csv("data/random_subset_200/redox_mer_with_iupac.csv.gz")
        else:
            dataset = pd.read_csv("data/redox_mer_with_iupac.csv.gz")
    else:
        dataset = pd.read_csv("data/random_subset_200/redox_mer.csv")
    OBJ_COL = "Gsol"
    MAXIMIZATION = False
elif args.problem == "kinase":
    if args.run_subset_only:
        dataset = pd.read_csv("data/random_subset_200/enamine10k.csv.gz")
    else:
        dataset = pd.read_csv("data/enamine10k.csv.gz")
    SMILES_COL = "SMILES"
    OBJ_COL = "score"
    MAXIMIZATION = False
elif args.problem == "laser":
    if args.run_subset_only:
        dataset = pd.read_csv("data/random_subset_200/laser_emitters10k.csv.gz")
    else:
        dataset = pd.read_csv("data/laser_emitters10k.csv.gz")
    SMILES_COL = "SMILES"
    OBJ_COL = "Fluorescence Oscillator Strength"
    MAXIMIZATION = True
elif args.problem == "pce":
    if args.run_subset_only:
        dataset = pd.read_csv("data/photovoltaics_pce10k.csv.gz")
    else:
        dataset = pd.read_csv("data/photovoltaics_pce10k.csv.gz")

    SMILES_COL = "SMILES"
    OBJ_COL = "pce"
    MAXIMIZATION = True
elif args.problem == "photoswitch":
    if args.run_subset_only:
        dataset = pd.read_csv("data/random_subset_200/photoswitches.csv.gz")
    else:
        dataset = pd.read_csv("data/photoswitches.csv.gz")
    SMILES_COL = "SMILES"
    OBJ_COL = "Pi-Pi* Transition Wavelength"
    MAXIMIZATION = True
else:
    print("Invalid test function!")
    sys.exit(1)

# Dataset
PROBLEM_NAME = args.problem + ("-iupac" if args.iupac else "")
CACHE_PATH = f"data/cache/{PROBLEM_NAME}/"
if args.feature_type not in ["fingerprints", "molformer"]:  # LLM features
    FEATURE_NAME = f"{args.feature_type}-{args.prompt_type}"
    FEATURE_NAME += "-average" if args.feature_reduction == "average" else ""
else:
    FEATURE_NAME = args.feature_type

features = torch.load(CACHE_PATH + f"{FEATURE_NAME}_feats.bin")
targets = torch.load(CACHE_PATH + f"{FEATURE_NAME}_targets.bin")
if args.run_subset_only:
    features_to_get = dataset["Entry Number"].to_numpy()
    features_subset, targets_subset = [], []
    for ii in features_to_get:
        features_subset.append(features[ii])
        targets_subset.append(targets[ii])
    features = features_subset
    targets = targets_subset
else:
    features, targets = skshuffle(features, targets, random_state=args.randseed)
feature_dim = features[0].shape[-1]
ground_truth_max = torch.tensor(targets).flatten().max()

# Normalize targets
if args.normalize_y:
    # TO-DO: map back during the iteration
    y_preprocessor = StandardScaler()
    # The format is list(torch.Tensor)
    targets = list(
        torch.tensor(
            y_preprocessor.fit_transform(np.array(targets).reshape(-1, 1))
        ).float()
    )

print()
if args.feature_type in ["fingerprints", "molformer"]:
    print(
        f"Test Function: {PROBLEM_NAME}; Feature Type: {args.feature_type}; Randseed: {args.randseed}"
    )
else:
    print(
        f"Test Function: {PROBLEM_NAME}; Foundation LLM: {args.feature_type}; Prompt Type: {args.prompt_type}; Reduction: {args.feature_reduction}; Randseed: {args.randseed}"
    )
print(
    "---------------------------------------------------------------------------------------------------------"
)
print()

train_x, train_y = [], []
while len(train_x) < args.n_init_data:
    idx = np.random.randint(len(features))
    # Make sure that the optimum is not included
    if targets[idx].item() >= ground_truth_max:
        continue
    train_x.append(features.pop(idx))
    train_y.append(targets.pop(idx))
train_x, train_y = torch.stack(train_x), torch.stack(train_y)

if args.method == "laplace":

    def get_net():
        activation = (
            torch.nn.Tanh
            if args.feature_type == "fingerprints" and args.problem != "photoswitch"
            else torch.nn.ReLU
        )
        return torch.nn.Sequential(
            torch.nn.Linear(feature_dim, 50),
            activation(),
            torch.nn.Linear(50, 50),
            activation(),
            torch.nn.Linear(50, 1),
        )

    model = LaplaceBoTorch(
        get_net,
        train_x,
        train_y,
        noise_var=0.001,
        hess_factorization="kron" if args.feature_type != "llama-2-7b" else "diag",
    )
    model = model.to(DEVICE) if model is not None else model
elif args.method == "gp":
    kernel = TanimotoKernel() if args.feature_type == "fingerprints" else MaternKernel()
    model = MLLGP(train_x, train_y, ScaleKernel(kernel), GaussianLikelihood())
else:  # Random search
    model = None

best_y = train_y.max().item()
pbar = tqdm.trange(args.exp_len)
pbar.set_description(f"[Best f(x) = {helpers.y_transform(best_y, MAXIMIZATION):.3f}]")

trace_best_y = [helpers.y_transform(ground_truth_max, MAXIMIZATION)] * (
    args.exp_len + 1
)
trace_timing = [0.0] * (args.exp_len + 1)
trace_best_acqval = []

timing_train = []
timing_preds = []

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
    else:
        dataloader = data_utils.DataLoader(
            data_utils.TensorDataset(torch.stack(features), torch.stack(targets)),
            batch_size=256,
            shuffle=False,
        )

        preds, uncerts, labels = [], [], []
        acq_vals = []
        start_pred = time.time()

        for x, y in dataloader:
            posterior = model.posterior(x)
            f_mean, f_var = posterior.mean, posterior.variance
            if args.acqf == "ei":
                acq_vals.append(ei(f_mean, f_var, best_y))
            elif args.acqf == "ucb":
                acq_vals.append(ucb(f_mean, f_var))
            else:
                acq_vals.append(thompson_sampling(f_mean, f_var))

            preds.append(f_mean)
            uncerts.append(f_var.sqrt())
            labels.append(y)

        timing_preds.append(time.time() - start_pred)

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

        trace_best_acqval.append(torch.max(acq_vals).item())

    # Update the current best y
    if new_y.item() > best_y:
        best_y = new_y.item()

    if args.method == "random":
        pbar.set_description(
            f"[Best f(x) = {helpers.y_transform(best_y, MAXIMIZATION):.3f}, "
            + f"curr f(x) = {helpers.y_transform(new_y.item(), MAXIMIZATION):.3f}]"
        )
    else:
        pbar.set_description(
            f"[Best f(x) = {helpers.y_transform(best_y, MAXIMIZATION):.3f}, "
            + f"curr f(x) = {helpers.y_transform(new_y.item(), MAXIMIZATION):.3f}, test MSE = {test_loss:.3f}]"
        )

        # Update surrogate
        start_train = time.time()
        model = model.condition_on_observations(new_x.unsqueeze(0), new_y.unsqueeze(0))
        timing_train.append(time.time() - start_train)

    # Housekeeping
    if DEVICE == "cuda":
        end.record()
        torch.cuda.synchronize()
        timing = start.elapsed_time(end) / 1000
    else:
        timing = time.time() - start

    trace_best_y[i + 1] = helpers.y_transform(best_y, MAXIMIZATION)
    trace_timing[i + 1] = timing

    # Early stopping if we already got the max
    if best_y >= ground_truth_max:
        break

# Save results
if args.run_subset_only:
    path = f"results/icl_experiments/{PROBLEM_NAME}/fixed/{args.method}/{FEATURE_NAME}"
else:
    path = f"results/{PROBLEM_NAME}/fixed/{args.method}/{FEATURE_NAME}"

if not os.path.exists(path):
    os.makedirs(path)

np.save(
    f"{path}/timing_train_{args.n_init_data}_{args.acqf}_{args.randseed}.npy",
    timing_train,
)
np.save(
    f"{path}/timing_preds_{args.n_init_data}_{args.acqf}_{args.randseed}.npy",
    timing_preds,
)
np.save(
    f"{path}/trace_best_acqval_{args.n_init_data}_{args.acqf}_{args.randseed}.npy",
    trace_best_acqval,
)
np.save(
    f"{path}/trace_best_y_{args.n_init_data}_{args.acqf}_{args.randseed}.npy",
    trace_best_y,
)
np.save(
    f"{path}/trace_timing_{args.n_init_data}_{args.acqf}_{args.randseed}.npy",
    trace_timing,
)
