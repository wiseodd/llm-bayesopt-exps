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
    GPT2Regressor,
    Llama2Regressor,
    T5Regressor,
)
from foundation_models import (
    get_molformer_tokenizer,
    get_roberta_tokenizer,
    get_gpt2_tokenizer,
    get_llama2_tokenizer,
    get_t5_tokenizer,
)
from problems.data_processor import (
    RedoxDataProcessor,
    SolvationDataProcessor,
    KinaseDockingDataProcessor,
    LaserEmitterDataProcessor,
    PhotovoltaicsPCEDataProcessor,
    PhotoswitchDataProcessor,
)
from problems.prompting import PromptBuilder
from rdkit import Chem
from rdkit.Chem import AllChem


from utils import helpers
from utils.configs import LLMFeatureType

parser = argparse.ArgumentParser()
parser.add_argument(
    "--feature_type",
    choices=[
        "fingerprints",
        "molformer",
        "t5-base",
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
parser.add_argument(
    "--prompt_type",
    choices=["single-number", "just-smiles", "naive", "completion"],
    default="just-smiles",
)
parser.add_argument(
    "--problem",
    choices=["redox-mer", "solvation", "kinase", "laser", "pce", "photoswitch"],
    default="redox-mer",
)
parser.add_argument("--iupac", default=False, action="store_true")
parser.add_argument("--normalize_y", default=False, action="store_true")
args = parser.parse_args()

if args.iupac and args.problem not in ["redox-mer", "solvation"]:
    print("IUPAC option is only available for redox-mer and solvation")
    sys.exit(1)

if args.problem == "redox-mer":
    dataset = pd.read_csv("data/redox_mer_with_iupac.csv.gz")
    SMILES_COL = "IUPAC Name" if args.iupac else "SMILES"
    OBJ_COL = "Ered"
    MAXIMIZATION = False
elif args.problem == "solvation":
    dataset = pd.read_csv("data/redox_mer_with_iupac.csv.gz")
    SMILES_COL = "IUPAC Name" if args.iupac else "SMILES"
    OBJ_COL = "Gsol"
    MAXIMIZATION = False
elif args.problem == "kinase":
    dataset = pd.read_csv("data/enamine10k.csv.gz")
    SMILES_COL = "SMILES"
    OBJ_COL = "score"
    MAXIMIZATION = False
elif args.problem == "laser":
    dataset = pd.read_csv("data/laser_multi10k.csv.gz")
    SMILES_COL = "SMILES"
    OBJ_COL = "Fluorescence Oscillator Strength"
    MAXIMIZATION = True
elif args.problem == "pce":
    dataset = pd.read_csv("data/photovoltaics_pce10k.csv.gz")
    SMILES_COL = "SMILES"
    OBJ_COL = "pce"
    MAXIMIZATION = True
elif args.problem == "photoswitch":
    dataset = pd.read_csv("data/photoswitches.csv.gz")
    SMILES_COL = "SMILES"
    OBJ_COL = "Pi-Pi* Transition Wavelength"
    MAXIMIZATION = True
else:
    print("Invalid test function!")
    sys.exit(1)

if args.feature_type == "fingerprints":
    features = [
        torch.tensor(
            np.asarray(
                AllChem.GetMorganFingerprintAsBitVect(
                    Chem.MolFromSmiles(mol), radius=2, nBits=1024
                )
            )
        ).float()
        for mol in tqdm.tqdm(dataset[SMILES_COL])
    ]
    targets = list(
        helpers.y_transform(torch.tensor(dataset[OBJ_COL].to_numpy()), MAXIMIZATION)
        .unsqueeze(-1)
        .float()
    )
else:  # LLM & MolFormer features
    if args.feature_type == "molformer":
        tokenizer = get_molformer_tokenizer()
    elif "roberta" in args.feature_type:
        tokenizer = get_roberta_tokenizer(args.feature_type)
    elif "t5" in args.feature_type:
        if "chem" in args.feature_type:
            foundation_model_real = "GT4SD/multitask-text-and-chemistry-t5-base-augm"
        else:
            foundation_model_real = args.feature_type
        tokenizer = get_t5_tokenizer(foundation_model_real)
    elif "gpt2" in args.feature_type:
        tokenizer = get_gpt2_tokenizer(args.feature_type)
    elif "llama-2" in args.feature_type:
        tokenizer = get_llama2_tokenizer(args.feature_type)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        # print(tokenizer.pad_token, tokenizer.pad_token_id)

    DEFAULT_REDUCTIONS = {
        "gpt2-medium": LLMFeatureType.LAST_TOKEN,
        "llama-2-7b": LLMFeatureType.LAST_TOKEN,
        "t5-base": LLMFeatureType.LAST_TOKEN,
        "t5-base-chem": LLMFeatureType.LAST_TOKEN,
        "molformer": None,
    }
    reduction = (
        DEFAULT_REDUCTIONS[args.feature_type]
        if args.feature_reduction == "default"
        else LLMFeatureType.AVERAGE
    )

    if args.feature_type == "molformer":
        llm_feat_extractor = MolFormerRegressor(tokenizer)
    elif "roberta" in args.feature_type:
        llm_feat_extractor = RobertaRegressor(
            kind=args.feature_type, tokenizer=tokenizer, reduction=reduction
        )
    elif "t5" in args.feature_type:
        if "chem" in args.feature_type:
            llm_feat_extractor = T5Regressor(
                kind="GT4SD/multitask-text-and-chemistry-t5-base-augm",
                tokenizer=tokenizer,
                reduction=reduction,
            )
        else:
            llm_feat_extractor = T5Regressor(
                kind=args.feature_type, tokenizer=tokenizer, reduction=reduction
            )
    elif "gpt" in args.feature_type:
        llm_feat_extractor = GPT2Regressor(
            kind=args.feature_type, tokenizer=tokenizer, reduction=reduction
        )
    elif "llama-2" in args.feature_type:
        llm_feat_extractor = Llama2Regressor(
            kind=args.feature_type, tokenizer=tokenizer, reduction=reduction
        )
    else:
        raise NotImplementedError  # TO-DO!

    llm_feat_extractor.cuda()
    llm_feat_extractor.eval()
    llm_feat_extractor.freeze_params()

    prompt_builder = PromptBuilder(kind=args.prompt_type)
    DATA_PROCESSORS = {
        "redox-mer": RedoxDataProcessor,
        "solvation": SolvationDataProcessor,
        "kinase": KinaseDockingDataProcessor,
        "laser": LaserEmitterDataProcessor,
        "pce": PhotovoltaicsPCEDataProcessor,
        "photoswitch": PhotoswitchDataProcessor,
    }
    data_processor = DATA_PROCESSORS[args.problem](prompt_builder, tokenizer)
    append_eos = args.feature_type != "molformer" and ("t5" not in args.feature_type)
    dataloader = data_processor.get_dataloader(
        dataset, shuffle=False, append_eos=append_eos
    )

    features, targets = [], []
    for data in tqdm.tqdm(dataloader):
        with torch.no_grad():
            feat = llm_feat_extractor.forward_features(data)

        features += list(feat.cpu())
        targets += list(helpers.y_transform(data["labels"], MAXIMIZATION))

# Save to files
problem_name = args.problem + ("-iupac" if args.iupac else "")
cache_path = f"data/cache/{problem_name}/"
if not os.path.exists(cache_path):
    os.makedirs(cache_path)

if args.feature_type not in ["fingerprints", "molformer"]:  # LLM features
    feature_name = f"{args.feature_type}-{args.prompt_type}-{args.feature_reduction}"
else:
    feature_name = args.feature_type

torch.save(features, cache_path + f"{feature_name}_feats.bin")
torch.save(targets, cache_path + f"{feature_name}_targets.bin")
