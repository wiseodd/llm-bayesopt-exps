# Bayesian Optimization with LLMs

## Setup

Best done in a fresh conda/mamba environment (Python < 3.12). Note that the ordering is important.

1. Install PyTorch (with CUDA): <https://pytorch.org/get-started/locally/>
2. Install Huggingface libraries and others: `pip install transformers datasets peft tqdm`
3. Install a specific branch of laplace-torch: `pip install git+https://github.com/aleximmer/Laplace.git@mc-subset2`
4. Install a specific version of ASDL (to compute Hessians): `pip install git+https://github.com/wiseodd/asdl.git@dev`


## Fixed-Feature Experiments

Cache molecules in $\mathcal{D}_{\mathrm{cand}}$ (see full parameters in the Python file):

```
python cache_features.py --feature_type {FEATURE_TYPE} --problem {PROBLEM} --prompt_type {PROMPT_TYPE}
```

Then, do BO:

```
python run_fixed_features.py --feature_type {FEATURE_TYPE} --method {METHOD} --randseed {RANDSEED} --problem {PROBLEM}
```

Similarly for the multiobjective experiments (`cache_features_multiobj.py` and `run_multiobj.py`).


## Finetuning Experiments

Simply run the following.

```
python run_finetuning.py --foundation_model {FOUNDATION_MODEL} --randseed {RANDSEED} --problem {PROBLEM}
```

See the Python file for the full arguments.


## BO-LIFT In Context Learning Baseline

The script is in `baselines/run_bolift.py`. It has similar options as the fixed-feature script.
