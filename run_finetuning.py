import numpy as np
import pandas as pd
import torch
import tqdm
import argparse
import sys
import os
from foundation_models import (
    MolFormerRegressor,
    RobertaRegressor,
    T5Regressor,
    GPT2Regressor,
    Llama2Regressor,
)
from foundation_models import (
    get_molformer_tokenizer,
    get_roberta_tokenizer,
    get_t5_tokenizer,
    get_gpt2_tokenizer,
    get_llama2_tokenizer,
)
from llm_bayesopt import LoRALLMBayesOpt
from bayesopt.acqf import ucb, ei, thompson_sampling
from problems.data_processor import (
    RedoxDataProcessor,
    SolvationDataProcessor,
    KinaseDockingDataProcessor,
    LaserEmitterDataProcessor,
    PhotovoltaicsPCEDataProcessor,
    PhotoswitchDataProcessor,
)
from problems.prompting import PromptBuilder
from utils import helpers
from utils.configs import LaplaceConfig, LLMFeatureType
from peft import LoraConfig, get_peft_model
from sklearn.preprocessing import StandardScaler
import math


parser = argparse.ArgumentParser()
parser.add_argument(
    "--problem",
    choices=["redox-mer", "solvation", "kinase", "laser", "pce", "photoswitch"],
    default="redox-mer",
)
parser.add_argument(
    "--foundation_model",
    default="gpt2-medium",
    choices=[
        "molformer",
        "roberta-large",
        "t5-base",
        "t5-base-chem",
        "gpt2-medium",
        "gpt2-large",
        "llama-2-7b",
    ],
)
parser.add_argument(
    "--prompt_type",
    choices=["single-number", "just-smiles", "completion"],
    default="just-smiles",
)
parser.add_argument(
    "--laplace_type", choices=["last_layer", "all_layer"], default="all_layer"
)
parser.add_argument("--acqf", choices=["ei", "ucb", "ts"], default="ts")
parser.add_argument("--n_init_data", type=int, default=10)
parser.add_argument("--exp_len", type=int, default=200)
parser.add_argument("--randseed", type=int, default=1)
args = parser.parse_args()

# Molformer expects only SMILES
if args.foundation_model == "molformer":
    args.prompt_type = "just-smiles"

np.random.seed(args.randseed)
torch.manual_seed(args.randseed)

if args.foundation_model == "molformer":
    tokenizer = get_molformer_tokenizer()
elif "roberta" in args.foundation_model:
    tokenizer = get_roberta_tokenizer(args.foundation_model)
elif "t5" in args.foundation_model:
    if "chem" in args.foundation_model:
        foundation_model_real = "GT4SD/multitask-text-and-chemistry-t5-base-augm"
    else:
        foundation_model_real = args.foundation_model
    tokenizer = get_t5_tokenizer(foundation_model_real)
elif "gpt2" in args.foundation_model:
    tokenizer = get_gpt2_tokenizer(args.foundation_model)
elif "llama-2" in args.foundation_model:
    tokenizer = get_llama2_tokenizer(args.foundation_model)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    # print(tokenizer.pad_token, tokenizer.pad_token_id, tokenizer.eos_token_id)

if args.problem == "redox-mer":
    dataset = pd.read_csv("data/redox_mer_with_iupac.csv.gz")
    dataset["Ered_orig"] = dataset["Ered"]
    y_preprocessor = StandardScaler()
    # dataset['Ered'] = y_preprocessor.fit_transform(dataset['Ered'].to_numpy().reshape(-1, 1)).flatten()
    OBJ_COL = "Ered"  # Preprocessed
    OBJ_COL_ORI = "Ered_orig"
    MAXIMIZATION = False
    prompt_builder = PromptBuilder(kind=args.prompt_type)
    data_processor = RedoxDataProcessor(prompt_builder, tokenizer)
elif args.problem == "solvation":
    dataset = pd.read_csv("data/redox_mer_with_iupac.csv.gz")
    dataset["Gsol_orig"] = dataset["Gsol"]
    y_preprocessor = StandardScaler()
    # dataset['Gsol'] = y_preprocessor.fit_transform(dataset['Gsol'].to_numpy().reshape(-1, 1)).flatten()
    OBJ_COL = "Gsol"  # Preprocessed
    OBJ_COL_ORI = "Gsol_orig"
    MAXIMIZATION = False
    prompt_builder = PromptBuilder(kind=args.prompt_type)
    data_processor = SolvationDataProcessor(prompt_builder, tokenizer)
elif args.problem == "kinase":
    dataset = pd.read_csv("data/enamine10k.csv.gz")
    dataset["score_orig"] = dataset["score"]
    y_preprocessor = StandardScaler()
    # dataset['score'] = y_preprocessor.fit_transform(dataset['score'].to_numpy().reshape(-1, 1)).flatten()
    OBJ_COL = "score"  # Preprocessed
    OBJ_COL_ORI = "score_orig"
    MAXIMIZATION = False
    prompt_builder = PromptBuilder(kind=args.prompt_type)
    data_processor = KinaseDockingDataProcessor(prompt_builder, tokenizer)
elif args.problem == "laser":
    dataset = pd.read_csv("data/laser_multi10k.csv.gz")
    OBJ_COL = "Fluorescence Oscillator Strength"  # Preprocessed
    OBJ_COL_ORI = "Fluorescence Oscillator Strength_orig"
    dataset[OBJ_COL_ORI] = dataset[OBJ_COL]
    y_preprocessor = StandardScaler()
    # dataset[OBJ_COL] = y_preprocessor.fit_transform(dataset[OBJ_COL].to_numpy().reshape(-1, 1)).flatten()
    MAXIMIZATION = True
    prompt_builder = PromptBuilder(kind=args.prompt_type)
    data_processor = LaserEmitterDataProcessor(prompt_builder, tokenizer)
elif args.problem == "pce":
    dataset = pd.read_csv("data/photovoltaics_pce10k.csv.gz")
    OBJ_COL = "pce"  # Preprocessed
    OBJ_COL_ORI = "pce_orig"
    dataset[OBJ_COL_ORI] = dataset[OBJ_COL]
    y_preprocessor = StandardScaler()
    # dataset[OBJ_COL] = y_preprocessor.fit_transform(dataset[OBJ_COL].to_numpy().reshape(-1, 1)).flatten()
    MAXIMIZATION = True
    prompt_builder = PromptBuilder(kind=args.prompt_type)
    data_processor = PhotovoltaicsPCEDataProcessor(prompt_builder, tokenizer)
elif args.problem == "photoswitch":
    dataset = pd.read_csv("data/photoswitches.csv.gz")
    SMILES_COL = "SMILES"
    OBJ_COL = "Pi-Pi* Transition Wavelength"
    OBJ_COL_ORI = "Pi-Pi* Transition Wavelength_orig"
    dataset[OBJ_COL_ORI] = dataset[OBJ_COL]
    y_preprocessor = StandardScaler()
    # dataset[OBJ_COL] = y_preprocessor.fit_transform(dataset[OBJ_COL].to_numpy().reshape(-1, 1)).flatten()
    MAXIMIZATION = True
    prompt_builder = PromptBuilder(kind=args.prompt_type)
    data_processor = PhotoswitchDataProcessor(prompt_builder, tokenizer)
else:
    print("Invalid test function!")
    sys.exit(1)

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


def get_model():
    if args.foundation_model == "molformer":
        model = MolFormerRegressor(tokenizer)
        target_modules = ["query", "value"]
    elif "roberta" in args.foundation_model:
        model = RobertaRegressor(
            kind=args.foundation_model,
            tokenizer=tokenizer,
            reduction=LLMFeatureType.AVERAGE,
        )
        target_modules = ["query", "value"]
    elif "gpt2" in args.foundation_model:
        model = GPT2Regressor(
            kind=args.foundation_model,
            tokenizer=tokenizer,
            reduction=LLMFeatureType.AVERAGE,
        )
        target_modules = ["c_attn"]
    elif "llama-2" in args.foundation_model:
        model = Llama2Regressor(
            kind=args.foundation_model,
            tokenizer=tokenizer,
            reduction=LLMFeatureType.AVERAGE,
        )
        target_modules = ["q_proj", "v_proj"]
    elif "t5" in args.foundation_model:
        if "chem" in args.foundation_model:
            model = T5Regressor(
                kind="GT4SD/multitask-text-and-chemistry-t5-base-augm",
                tokenizer=tokenizer,
                reduction=LLMFeatureType.AVERAGE,
            )
        else:
            model = T5Regressor(
                kind=args.foundation_model,
                tokenizer=tokenizer,
                reduction=LLMFeatureType.AVERAGE,
            )
        target_modules = ["q", "v"]
    else:
        raise NotImplementedError

    config = LoraConfig(
        r=4,
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
        modules_to_save=["head"],
    )
    lora_model = get_peft_model(model, config)
    for p in lora_model.base_model.head.original_module.parameters():
        p.requires_grad = False
    # for n, p in lora_model.named_parameters():
    #     if p.requires_grad:
    #         print(n)
    return lora_model


# Train + Laplace
if args.laplace_type == "all_layer":
    config = LaplaceConfig(
        n_epochs=50,
        noise_var=0.001,
        hess_factorization="kron",
        subset_of_weights="all",
        marglik_mode="posthoc",
        prior_prec_structure="layerwise",
    )
else:
    config = LaplaceConfig(
        n_epochs=30,
        noise_var=0.001,
        hess_factorization="full",
        subset_of_weights="last_layer",
    )

if args.problem == "photoswitch":
    config.lr = 1e-2
    config.lr_lora = 3e-3

APPEND_EOS = args.foundation_model != "molformer" and (
    "t5" not in args.foundation_model
)
model = LoRALLMBayesOpt(
    get_model,
    dataset_train,
    data_processor,
    dtype="float32",
    laplace_config=config,
    append_eos=APPEND_EOS,
)

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
trace_acqvals = [-math.inf] * (args.exp_len + 1)

timing_train = []
timing_preds = []

for i in pbar:
    # Timing
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()

    # BO iteration
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

    start_pred = torch.cuda.Event(enable_timing=True)
    end_pred = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start_pred.record()

    for data in sub_pbar:
        posterior = model.posterior(data)
        f_mean, f_var = posterior.mean, posterior.variance
        if args.acqf == "ei":
            acq_vals.append(ei(f_mean, f_var, best_y))
        elif args.acqf == "ucb":
            acq_vals.append(ucb(f_mean, f_var))
        else:
            acq_vals.append(thompson_sampling(f_mean, f_var))

        preds.append(f_mean)
        uncerts.append(f_var.sqrt())
        labels.append(data["labels"])

    end_pred.record()
    torch.cuda.synchronize()
    timing_preds.append(start_pred.elapsed_time(end_pred) / 1000)

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
    # input()

    # Pick a molecule (a row in the current dataset) that maximizes the acquisition
    idx_best = torch.argmax(acq_vals).item()
    new_data = helpers.pop_df(dataset, idx_best)

    # Update the current best y
    if new_data[OBJ_COL] > best_y:
        best_y = new_data[OBJ_COL]
        best_y_ori = new_data[OBJ_COL_ORI]
        print(best_y_ori)

    # Early stopping if we already got the max
    if best_y >= ground_truth_max:
        break

    start_train = torch.cuda.Event(enable_timing=True)
    end_train = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start_train.record()

    # Update surrogate
    model = model.condition_on_observations(new_data)

    end_train.record()
    torch.cuda.synchronize()
    timing_train.append(start_train.elapsed_time(end_train) / 1000)

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

# print('Train time (avg & sem)', f'{np.mean(timing_train):.1f}', f'{st.sem(timing_train):.1f}')
# print('Preds time (avg & sem)', f'{np.mean(timing_preds):.1f}', f'{st.sem(timing_preds):.1f}')

# Save results
path = f"results/{args.problem}/finetuning/{args.foundation_model}"
if not os.path.exists(path):
    os.makedirs(path)

np.save(
    f"{path}/timing_train_{args.n_init_data}_{args.acqf}_{args.laplace_type}_{args.randseed}.npy",
    timing_train,
)
np.save(
    f"{path}/timing_preds_{args.n_init_data}_{args.acqf}_{args.laplace_type}_{args.randseed}.npy",
    timing_preds,
)

np.save(
    f"{path}/trace_acqvals_{args.n_init_data}_{args.acqf}_{args.laplace_type}_{args.randseed}.npy",
    trace_acqvals,
)

if args.foundation_model == "molformer":
    np.save(
        f"{path}/trace_best_y_{args.n_init_data}_{args.acqf}_{args.laplace_type}_{args.randseed}.npy",
        trace_best_y,
    )
    np.save(
        f"{path}/trace_timing_{args.n_init_data}_{args.acqf}_{args.laplace_type}_{args.randseed}.npy",
        trace_timing,
    )
else:
    np.save(
        f"{path}/{args.prompt_type}_trace_best_y_{args.n_init_data}_{args.acqf}_{args.laplace_type}_{args.randseed}.npy",
        trace_best_y,
    )
    np.save(
        f"{path}/{args.prompt_type}_trace_timing_{args.n_init_data}_{args.acqf}_{args.laplace_type}_{args.randseed}.npy",
        trace_timing,
    )
