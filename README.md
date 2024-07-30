# A Sober Look at LLMs for Material Discovery

Official experiment repo for the "A Sober Look at LLMs for Material Discovery" paper (ICML 2024).

!> [!TIP]
> If you just want to use the method as a library, check out the sister repo: <https://github.com/wiseodd/lapeft-bayesopt>.

## Setup

> [!IMPORTANT]
> Note that the ordering is important.

1. Install PyTorch (with CUDA): <https://pytorch.org/get-started/locally/>
2. Install Huggingface libraries and others: `pip install transformers datasets peft tqdm`
3. Install laplace-torch: `pip install git+https://github.com/aleximmer/laplace.git@0.2`

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

## Citation

```bib
@inproceedings{kristiadi2024sober,
  title={A Sober Look at {LLMs} for Material Discovery: {A}re They Actually Good for {B}ayesian Optimization Over Molecules?},
  author={Kristiadi, Agustinus and Strieth-Kalthoff, Felix and Skreta, Marta and Poupart, Pascal and Aspuru-Guzik, Al\'{a}n and Pleiss, Geoff},
  booktitle={ICML},
  year={2024}
}
```
