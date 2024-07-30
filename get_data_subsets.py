import numpy as np
import pandas as pd
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--num_samples", type=int, default=200)
parser.add_argument("--randseed", type=int, default=1)
parser.add_argument("--iupac", default=False, action="store_true")
args = parser.parse_args()


SAVE_DIR = "data/random_subset_300"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR, exist_ok=True)

for problem in ["redox-mer", "solvation", "kinase", "laser", "pce", "photoswitch"]:
    np.random.seed(args.randseed)
    if args.iupac and problem not in ["redox-mer", "solvation"]:
        continue
    if problem == "redox-mer":
        if args.iupac:
            dataset = pd.read_csv("data/redox_mer_with_iupac.csv.gz")
            save_path = f"{SAVE_DIR}/redox_mer_with_iupac.csv.gz"
        else:
            dataset = pd.read_csv("data/redox_mer.csv")
            save_path = f"{SAVE_DIR}/redox_mer.csv"
    elif problem == "solvation":
        pass  # same data file as redox_mer
    elif problem == "kinase":
        dataset = pd.read_csv("data/enamine10k.csv.gz")
        save_path = f"{SAVE_DIR}/enamine10k.csv.gz"
    elif problem == "laser":
        dataset = pd.read_csv("data/laser_emitters10k.csv.gz")
        save_path = f"{SAVE_DIR}/laser_emitters10k.csv.gz"
    elif problem == "pce":
        dataset = pd.read_csv("data/photovoltaics_pce10k.csv.gz")
        save_path = f"{SAVE_DIR}/photovoltaics_pce10k.csv.gz"
    elif problem == "photoswitch":
        dataset = pd.read_csv("data/photoswitches.csv.gz")
        save_path = f"{SAVE_DIR}/photoswitches.csv.gz"
    print("old dataset", dataset)
    dataset = dataset.sample(n=300)
    print("new dataset", dataset)
    dataset.to_csv(save_path)
