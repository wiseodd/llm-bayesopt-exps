import os
import psutil
import json
import torch
import numpy as np
import bolift
import pandas as pd
import tqdm
from sklearn.preprocessing import StandardScaler

from utils import helpers
from bayesopt.acqf import ei, thompson_sampling
from problems.data_processor import (
    RedoxDataProcessor,
)
from problems.prompting import PromptBuilder
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--feature_type",
    choices=["gpt4", "llama-2-7b", "gpt-3.5-turbo-1106", "gpt-4-1106-preview"],
    default="gpt4",
)
parser.add_argument(
    "--problem",
    choices=["redox-mer", "solvation", "kinase", "laser", "pce", "photoswitch"],
    default="redox-mer",
)
parser.add_argument("--method", choices=["bolift"], default="bolift")
parser.add_argument(
    "--foundation_model",
    default="gpt4",
    choices=["llama-2-7b", "gpt4", "gpt-3.5-turbo-1106", "gpt-4-1106-preview"],
)
parser.add_argument(
    "--prompt_type",
    choices=["single-number", "just-smiles", "completion"],
    default="completion",
)
parser.add_argument(
    "--laplace_type", choices=["last_layer", "all_layer"], default="all_layer"
)
parser.add_argument("--acqf", choices=["ei", "ucb", "ts"], default="ts")
parser.add_argument("--n_init_data", type=int, default=5)
parser.add_argument("--exp_len", type=int, default=15)
parser.add_argument("--randseed", type=int, default=1)
args = parser.parse_args()

command_run = psutil.Process(os.getpid())
command_run = " ".join(command_run.cmdline())

np.random.seed(args.randseed)
torch.manual_seed(args.randseed)

if args.feature_type not in ["fingerprints", "molformer"]:  # LLM features
    FEATURE_NAME = f"{args.feature_type}-{args.prompt_type}"
else:
    FEATURE_NAME = args.feature_type

if args.problem == "redox-mer":
    dataset = pd.read_csv("../data/random_subset_300/redox_mer.csv")
    print(dataset)
    dataset["Ered_orig"] = dataset["Ered"]
    y_preprocessor = StandardScaler()
    OBJ_COL = "Ered"  # Preprocessed
    OBJ_COL_ORI = "Ered_orig"
    MAXIMIZATION = False
    prompt_builder = PromptBuilder(kind=args.prompt_type)
    tokenizer = None
    data_processor = RedoxDataProcessor(prompt_builder, tokenizer)

# Turn into a maximization problem if necessary
if not MAXIMIZATION:
    dataset[OBJ_COL] = -dataset[OBJ_COL]
    dataset[OBJ_COL_ORI] = -dataset[OBJ_COL_ORI]
ground_truth_max = dataset[OBJ_COL].max()
ground_truth_max_ori = dataset[OBJ_COL_ORI].max()

print()
print(
    f"Test Function: {args.problem}; Foundation Model: {args.foundation_model}; Prompt Type: {args.prompt_type}; Randseed: {args.randseed}"
)
print(
    "---------------------------------------------------------------------------------------------------------------"
)
print()

dataset_train = []
while len(dataset_train) < args.n_init_data:
    idx = np.random.randint(len(dataset))
    # Make sure that the optimum is not included
    if dataset.loc[idx][OBJ_COL] >= ground_truth_max:
        continue
    dataset_train.append(helpers.pop_df(dataset, idx))

best_y = pd.DataFrame(dataset_train)[OBJ_COL].max()
best_y_ori = pd.DataFrame(dataset_train)[OBJ_COL_ORI].max()
pbar = tqdm.trange(args.exp_len, position=0, colour="green", leave=True)
pbar.set_description(
    f"[Best f(x) = {helpers.y_transform(best_y_ori, MAXIMIZATION):.3f}]"
)


trace_best_y = [helpers.y_transform(ground_truth_max_ori, MAXIMIZATION)] * (
    args.exp_len + 1
)
trace_timing = [0.0] * (args.exp_len + 1)

SAVE_CACHE = False
LOAD_CACHE = False

APPEND_EOS = False
if args.foundation_model == "llama-2-7b":
    model_name = "Llama-2-7b-chat-hf"
elif args.foundation_model == "gpt4":
    model_name = "gpt-4"
else:
    model_name = args.foundation_model
print("loading model.....", model_name)

model = bolift.AskTellFewShotTopk(
    x_formatter=lambda x: f"smiles {x}",
    y_name="redox potential",
    y_formatter=lambda y: f"{y:.2f}",
    model=model_name,
    selector_k=5,
    temperature=0.7,
)

checkpoint = {0: []}

for row in dataset_train:
    # continue
    model.tell(row["SMILES"], row[OBJ_COL])
    checkpoint[0].append([row["SMILES"], row[OBJ_COL]])
    print("telling the model::{}|||{}".format(row["SMILES"], row[OBJ_COL]))

CHECKPOINT_PATH = f"checkpoints/{args.feature_type}_{args.problem}_{args.randseed}"
PROBLEM_NAME = args.problem
FEATURE_NAME = args.feature_type

if not os.path.exists(CHECKPOINT_PATH):
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)

print(f"command executed: {command_run}\n")
print(f"args: {args}\n")
with open(f"{CHECKPOINT_PATH}/logs.txt", "a") as f:
    f.write(f"command executed: {command_run}\n\n")
    f.write(f"args: {args}\n\n")

with open(f"{CHECKPOINT_PATH}/checkpoint_0.json", "w") as f:
    json.dump(checkpoint, f)


for i in pbar:
    print("STARTING ROUND -- {}".format(i))
    # Timing
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    # BO iteration
    APPEND_EOS = False

    dataloader = data_processor.get_dataloader(
        dataset, batch_size=16, shuffle=False, append_eos=APPEND_EOS
    )

    preds, uncerts, labels = [], [], []
    acq_vals = []
    sub_pbar = tqdm.tqdm(
        dataloader,
        position=1,
        colour="blue",
        desc="[Prediction over dataset]",
        leave=False,
    )

    for data in sub_pbar:
        f_mean, f_var = [], []
        for ii in range(len(data["SMILES"])):
            x = data["SMILES"][ii]
            y = data["labels"][0][ii]
            yhat = model.predict(x)
            yhat_mean = yhat.mean()
            yhat_std = yhat.std()

            f_mean.append(yhat_mean)
            f_var.append(yhat_std)
        f_mean = torch.FloatTensor(f_mean).to("cuda")
        f_var = torch.FloatTensor(f_var).to("cuda")
        if args.acqf == "ei":
            acq_vals.append(ei(f_mean, f_var, best_y))
        else:
            acq_vals.append(thompson_sampling(f_mean, f_var))
        preds.append(f_mean)
        uncerts.append(f_var)
        labels.append(data["labels"][0])
    acq_vals = torch.cat(acq_vals, dim=0).cpu().squeeze()
    preds, uncerts, labels = (
        torch.cat(preds, dim=0).cpu(),
        torch.cat(uncerts, dim=0).cpu(),
        torch.cat(labels, dim=0),
    )
    test_loss = torch.nn.MSELoss()(preds, labels).item()

    _, idx = acq_vals.topk(k=10)
    for l, p, u, a in zip(labels[idx], preds[idx], uncerts[idx], acq_vals[idx]):
        print(
            f"True: {l.item():.3f}, Mean: {p.item():.3f}, Std: {u.item():.3f}, Acqf: {a.item():.3f}"
        )

    # Pick a molecule (a row in the current dataset) that maximizes the acquisition
    idx_best = torch.argmax(acq_vals).item()
    new_data = helpers.pop_df(dataset, idx_best)

    # Update the current best y
    if new_data[OBJ_COL] > best_y:
        best_y = new_data[OBJ_COL]
        best_y_ori = new_data[OBJ_COL_ORI]
        print(best_y_ori)

    pbar.set_description(
        f"[Best f(x) = {helpers.y_transform(best_y_ori, MAXIMIZATION):.3f}, "
        + f"curr f(x) = {helpers.y_transform(new_data[OBJ_COL_ORI], MAXIMIZATION):.3f}, "
        + f"test MSE: {test_loss:.3f}]"
    )

    # Save results
    end.record()
    torch.cuda.synchronize()
    timing = start.elapsed_time(end) / 1000
    trace_best_y[i + 1] = helpers.y_transform(best_y_ori, MAXIMIZATION)
    trace_timing[i + 1] = timing

    # Update surrogate
    print("telling the model::{}|||{}".format(new_data["SMILES"], new_data[OBJ_COL]))
    model.tell(new_data["SMILES"], new_data[OBJ_COL])

    if i not in checkpoint:
        checkpoint[i] = []

    checkpoint[i].append([new_data["SMILES"], new_data[OBJ_COL]])

    with open(f"{CHECKPOINT_PATH}/checkpoint_{i}.json", "w") as f:
        json.dump(checkpoint, f)

    path = f"results_temp/{PROBLEM_NAME}/fixed/{args.method}/{FEATURE_NAME}"
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    np.save(
        f"{path}/trace_best_y_{args.n_init_data}_{args.acqf}_{args.randseed}_step{i}.npy",
        trace_best_y,
    )
    np.save(
        f"{path}/trace_timing_{args.n_init_data}_{args.acqf}_{args.randseed}_step{i}.npy",
        trace_timing,
    )

    if best_y >= ground_truth_max:
        break

path = f"results/icl_experiments/{PROBLEM_NAME}/fixed/{args.method}/{FEATURE_NAME}"
if not os.path.exists(path):
    os.makedirs(path, exist_ok=True)

np.save(
    f"{path}/trace_best_y_{args.n_init_data}_{args.acqf}_{args.randseed}.npy",
    trace_best_y,
)
np.save(
    f"{path}/trace_timing_{args.n_init_data}_{args.acqf}_{args.randseed}.npy",
    trace_timing,
)
