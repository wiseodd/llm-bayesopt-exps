import warnings

warnings.filterwarnings("ignore")

from transformers import logging

logging.set_verbosity_error()

import torch
from torch import nn, optim
from laplace import Laplace
from laplace.marglik_training import marglik_training
import math
import pandas as pd
from .base import LLMBayesOpt
from problems.data_processor import DataProcessor
from utils.configs import LaplaceConfig
import tqdm
from contextlib import nullcontext
from transformers import get_scheduler

from typing import *


class LoRALLMBayesOpt(LLMBayesOpt):
    """
    The Laplace approx. is applied on the regression head and the LoRA weights.
    """

    def __init__(
        self,
        get_model: Callable[[], torch.nn.Module],
        training_set: List[pd.Series],
        data_processor: DataProcessor,
        bnn: Laplace = None,
        laplace_config: LaplaceConfig = None,
        device: str = "cuda",
        dtype: str = "float32",
        append_eos: bool = True,
    ):
        self.dtype = dtype
        self.ptdtype = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[dtype]
        self.ctx = (
            nullcontext()
            if device == "cpu"
            else torch.amp.autocast(device_type="cuda", dtype=self.ptdtype)
        )
        self.enable_grad_scaler = dtype in ["float16", "bfloat16"]
        self.append_eos = append_eos
        super().__init__(
            get_model, training_set, data_processor, bnn, laplace_config, device
        )

    def train_model(self):
        del self.bnn
        cfg = self.laplace_config

        train_loader = self.data_processor.get_dataloader(
            pd.DataFrame(self.training_set),
            # batch_size=self.laplace_config.batch_size,
            batch_size=20,
            shuffle=True,
            append_eos=self.append_eos,
        )

        if self.laplace_config.marglik_mode == "posthoc":
            self._posthoc_laplace(train_loader)
        else:  # Online
            la, _, _, _ = marglik_training(
                # Ensure that the base net is re-initialized
                self.get_model().to(self.device),
                train_loader,
                likelihood="regression",
                hessian_structure=cfg.hess_factorization,
                n_epochs=cfg.n_epochs,
                backend=cfg.hessian_backend,
                optimizer_cls=optim.AdamW,
                optimizer_kwargs={"lr": cfg.lr},
                scheduler_cls=optim.lr_scheduler.CosineAnnealingLR,
                scheduler_kwargs={"T_max": cfg.n_epochs * len(train_loader)},
                marglik_frequency=cfg.online_marglik_freq,
                prior_structure=cfg.prior_prec_structure,
                sigma_noise_fixed=cfg.noise_var,
                progress_bar=True,
            )
            self.bnn = la

        # Override sigma_noise if self.noise_var is not None
        if cfg.noise_var is not None:
            self.bnn.sigma_noise = math.sqrt(cfg.noise_var)

    def posterior(self, data):
        f_mean, f_var = self.bnn(data)  # (B, 1) and (B, 1, 1)
        f_mean, f_var = f_mean.detach(), f_var.detach()
        f_var = f_var.squeeze(-1) + self.bnn.sigma_noise**2
        return torch.distributions.Normal(f_mean, f_var)

    def condition_on_observations(self, obs):
        self.training_set.append(obs)
        del self.bnn

        return LoRALLMBayesOpt(
            get_model=self.get_model,
            training_set=self.training_set,  # Modified
            data_processor=self.data_processor,
            bnn=None,  # Will be retrained
            laplace_config=self.laplace_config,
            device=self.device,
            dtype=self.dtype,
            append_eos=self.append_eos,
        )

    def _posthoc_laplace(self, train_loader):
        cfg = self.laplace_config
        model = self.get_model().to(
            self.device
        )  # Ensure that the base net is re-initialized
        model.train()
        loss_func = nn.MSELoss()

        lora_params = [
            p for n, p in model.named_parameters() if p.requires_grad and "lora" in n
        ]
        head_params = [
            p
            for n, p in model.named_parameters()
            if p.requires_grad and "lora" not in n
        ]
        optimizer_lora = optim.AdamW(lora_params, lr=cfg.lr_lora, weight_decay=5e-4)
        optimizer_head = optim.AdamW(head_params, lr=cfg.lr, weight_decay=5e-4)

        num_training_steps = cfg.n_epochs * len(train_loader)
        scheduler_lora = get_scheduler(
            name="linear",
            optimizer=optimizer_lora,
            # num_warmup_steps=0.06*num_training_steps,  # Following the warmup ratio in LoRA paper
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )
        scheduler_head = get_scheduler(
            name="cosine",
            optimizer=optimizer_head,
            # num_warmup_steps=0.06*num_training_steps,  # Following the warmup ratio in LoRA paper
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )
        scaler = torch.cuda.amp.GradScaler(enabled=self.enable_grad_scaler)

        for _ in tqdm.trange(
            cfg.n_epochs, position=1, leave=False, desc="[Training]", colour="blue"
        ):
            # for _ in range(cfg.n_epochs):
            for batch in train_loader:
                model.train()
                labels = batch["labels"].to(self.device, non_blocking=True)

                with self.ctx:
                    outputs = model(batch)
                    # print(outputs.shape, labels.shape); input()
                    loss = loss_func(outputs, labels)

                scaler.scale(loss).backward()

                if cfg.grad_clip != 0.0:
                    scaler.unscale_(optimizer_lora)
                    torch.nn.utils.clip_grad_norm_(lora_params, cfg.grad_clip)

                scaler.step(optimizer_lora)
                scaler.step(optimizer_head)
                scaler.update()
                scheduler_lora.step()
                scheduler_head.step()
                optimizer_lora.zero_grad(set_to_none=True)
                optimizer_head.zero_grad(set_to_none=True)

        for n, p in model.named_parameters():
            if "lora" in n:
                p.requires_grad = False

        optimizer_head = optim.AdamW(head_params, lr=1e-3, weight_decay=5e-4)
        scheduler_head = get_scheduler(
            name="cosine",
            optimizer=optimizer_head,
            # num_warmup_steps=0.06*num_training_steps,  # Following the warmup ratio in LoRA paper
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )

        for _ in tqdm.trange(
            100, position=1, leave=False, desc="[Training]", colour="blue"
        ):
            # for _ in range(cfg.n_epochs):
            for batch in train_loader:
                model.train()
                labels = batch["labels"].to(self.device, non_blocking=True)

                with self.ctx:
                    outputs = model(batch)
                    # print(outputs.shape, labels.shape); input()
                    loss = loss_func(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer_head)
                scaler.update()
                scheduler_head.step()
                optimizer_head.zero_grad(set_to_none=True)

        # So that it's considered by Laplace
        for n, p in model.named_parameters():
            if "lora" in n:
                p.requires_grad = True

        model.eval()

        # # Check training perf
        # preds, targets = [], []
        # for batch in train_loader:
        #     preds.append(model(batch))
        #     targets.append(batch['labels'])
        # preds, targets = torch.cat(preds, dim=0).cpu(), torch.cat(targets, dim=0)
        # print(f'Training MSE: {loss_func(preds, targets).item():.3f}')

        if cfg.subset_of_weights == "last_layer":
            self.bnn = Laplace(
                model,
                likelihood="regression",
                subset_of_weights=cfg.subset_of_weights,
                hessian_structure=cfg.hess_factorization,
                sigma_noise=1 if cfg.noise_var is None else math.sqrt(cfg.noise_var),
                last_layer_name=cfg.last_layer_name,
            )
        else:
            self.bnn = Laplace(
                model,
                likelihood="regression",
                subset_of_weights=cfg.subset_of_weights,
                hessian_structure=cfg.hess_factorization,
                sigma_noise=1 if cfg.noise_var is None else math.sqrt(cfg.noise_var),
            )
        # print('Fitting Laplace...')
        self.bnn.fit(train_loader)

        # print('Optimizing hyperparams...')
        prior_prec_shapes = {
            "scalar": 1,
            "layerwise": self.bnn.n_layers,
            "diagonal": self.bnn.n_params,
        }
        if cfg.noise_var is None:
            # Tune prior precision and observation noise
            log_prior = torch.ones(
                prior_prec_shapes[cfg.prior_prec_structure],
                requires_grad=True,
                device=self.device,
            )
            log_sigma = torch.ones(1, requires_grad=True, device=self.device)
            hyper_optimizer = torch.optim.Adam([log_prior, log_sigma], lr=1e-1)

            for _ in range(cfg.posthoc_marglik_iters):
                hyper_optimizer.zero_grad()
                neg_marglik = -self.bnn.log_marginal_likelihood(
                    log_prior.exp(), log_sigma.exp()
                )
                neg_marglik.backward()
                hyper_optimizer.step()

            self.bnn.prior_precision = log_prior.detach().exp()
            self.bnn.sigma_noise = log_sigma.detach().exp()
        else:
            # Tune only prior precision
            init_prior_prec = torch.ones(
                prior_prec_shapes[cfg.prior_prec_structure], device=self.device
            )
            self.bnn.optimize_prior_precision(
                n_steps=cfg.posthoc_marglik_iters, init_prior_prec=init_prior_prec
            )
            # print(self.bnn.prior_precision)
        # print('Done!')
