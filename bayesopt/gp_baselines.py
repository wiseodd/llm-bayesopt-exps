from __future__ import annotations
import warnings

warnings.filterwarnings("ignore")

import torch
from gpytorch.likelihoods import Likelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.gp_regression import SingleTaskGP
from gpytorch.kernels import Kernel


class MLLGP(SingleTaskGP):
    """
    Gaussian Process regressor (following the botorch API) for
    molecular fingerprint data (e.g. Morgan fingerprints),
    using the Tanimoto kernel from the gauche library.

    https://github.com/leojklarner/gauche.git
    """

    def __init__(
        self,
        train_X: torch.Tensor,
        train_Y: torch.Tensor,
        kernel: Kernel,
        likelihood: Likelihood | None = None,
        lr: float = 0.01,
        n_epochs: int = 500,
    ):
        SingleTaskGP.__init__(
            self,
            train_X=train_X,
            train_Y=train_Y,
            likelihood=likelihood,
            covar_module=kernel,
        )

        self.kernel = kernel
        self.lr = lr
        self.n_epochs = n_epochs

        self._train_model()

    def _train_model(self):
        """
        Implements a simple training procedure for the GP model
        (exact marginal log likelihood from gpytorch).
        """
        mll = ExactMarginalLogLikelihood(self.likelihood, self)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        self.train()
        self.likelihood.train()
        mll.train()

        for _ in range(self.n_epochs):
            optimizer.zero_grad()
            output = self(self.train_inputs[0])
            loss = (-mll(output, self.train_targets)).mean()
            loss.backward()
            optimizer.step()

        self.eval()
        self.likelihood.eval()

    def condition_on_observations(
        self, X: torch.Tensor, Y: torch.Tensor, **kwargs
    ) -> MLLGP:
        """
        Returns a new GP conditioned on the provided observations.
        """
        # ATTN: Do we want to use this implementation (i.e. re-training the GP from scratch) or the default
        #       implementation from botorch (i.e. using the fantasy model approach from gpytorch)?
        train_X = torch.cat([self.train_inputs[0], X])
        train_Y = torch.cat([self.train_targets.unsqueeze(-1), Y])
        return MLLGP(
            train_X, train_Y, self.kernel, self.likelihood, self.lr, self.n_epochs
        )
