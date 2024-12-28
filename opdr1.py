#!/usr/bin/env python3
# File: opdr1_modified_sharedparams.py
"""
Example script that:
  1) Implements a combined TuRBO + Weighted-PCA approach (LOCAL trust-region PCA).
  2) 3 other baseline algorithms:
       - Standalone TuRBO
       - Standalone PCA-BO
       - BAXUS
  3) Benchmarks them on selected BBOB problems via the `ioh` package
  4) Saves results for IOHanalyzer
  5) Plots runtimes comparison among the 4 methods.
"""

import sys
import time
import numpy as np
import torch
import gpytorch
from torch.quasirandom import SobolEngine
from sklearn.decomposition import PCA
from scipy.stats import rankdata
import matplotlib.pyplot as plt
from ioh import get_problem
from ioh import logger as ioh_logger

# ====================== SHARED HYPERPARAMETERS =====================
SHARED_PARAMS = {
    # Generic BO hyperparams
    "n_init": 5,          # Number of initial points
    "batch_size": 8,
    "use_ard": True,
    "pca_components": 2,  # For PCA-based methods
    # TuRBO trust-region settings
    "length_init": 0.8,
    "length_min": 0.5 ** 7,
    "length_max": 1.6,
    "success_tol": 3,
    "fail_tol": 3,
    # GP training
    "n_training_steps": 50,
    "lr": 0.01,
    # Candidate set
    # We often do: n_cand = min(50*d, 2500). We'll keep that logic inside each class.
    # BAXUS-specific
    "baxus_train_interval": 10,  # Only train GP every 10 steps
    "baxus_fail_tol_factor": 4.0,
    "baxus_init_target_dim": 5,
    "baxus_seed": 0,
}

# ======================== Basic GP for demonstration ========================

class SimpleGP(gpytorch.models.ExactGP):
    """
    A minimal Gaussian Process model using GPyTorch.
    Subclass gpytorch.models.ExactGP for use with ExactMarginalLogLikelihood.
    """

    def __init__(self, train_x, train_y, likelihood, use_ard=True):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()

        d = train_x.shape[1]
        if use_ard:
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=d)
            )
        else:
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(nu=2.5)
            )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train_gp(X, y, n_training_steps=50, use_ard=True, lr=0.01):
    """
    Train a GP on data X, y.
    """
    X_torch = torch.tensor(X, dtype=torch.float32)
    y_torch = torch.tensor(y, dtype=torch.float32)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = SimpleGP(X_torch, y_torch, likelihood, use_ard=use_ard)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for _ in range(n_training_steps):
        optimizer.zero_grad()
        output = model(X_torch)
        loss = -mll(output, y_torch)
        loss.backward()
        optimizer.step()

    return model, likelihood

# ======================== Helpers for scaling & PCA =========================

def to_unit_cube(X, lb, ub):
    return (X - lb) / (ub - lb)

def from_unit_cube(X_unit, lb, ub):
    return X_unit * (ub - lb) + lb

def _weighted_pca(X_scaled, f_scaled, n_components):
    """
    Weighted PCA using rank-based weights.
    """
    ranks = rankdata(f_scaled)  # rank 1..N
    N = len(f_scaled)
    weights = np.log(N) - np.log(ranks)
    weights /= np.sum(weights)

    X_mean = X_scaled.mean(axis=0)
    X_centered = X_scaled - X_mean

    X_weighted = X_centered * weights.reshape(-1, 1)

    d_eff = min(n_components, X_scaled.shape[1], X_scaled.shape[0])
    pca = PCA(n_components=d_eff, svd_solver='randomized')
    X_reduced = pca.fit_transform(X_weighted)

    return pca, X_reduced, X_mean

def _apply_pca_unweighted(X_scaled, pca, X_mean):
    X_centered = X_scaled - X_mean
    X_reduced = pca.transform(X_centered)
    return X_reduced


# ========== Combined TuRBO + Weighted PCA (LOCAL PCA) ==========

class TuRBO_PCA_BO:
    def __init__(self, f, lb, ub, hyperparams):
        self.f = f
        self.lb = lb
        self.ub = ub
        self.dim = len(lb)

        # Unpack shared hyperparams
        self.n_init = hyperparams["n_init"]
        self.max_evals = hyperparams["max_evals"]
        self.batch_size = hyperparams["batch_size"]
        self.use_ard = hyperparams["use_ard"]
        self.pca_components = hyperparams["pca_components"]
        self.length = hyperparams["length_init"]
        self.length_min = hyperparams["length_min"]
        self.length_max = hyperparams["length_max"]
        self.success_tol = hyperparams["success_tol"]
        self.fail_tol = hyperparams["fail_tol"]

        # Data storage
        self.X = np.zeros((0, self.dim))
        self.fX = np.zeros((0, 1))
        self.best_x = None
        self.best_f = float('inf')
        self.n_evals = 0

        self.success_count = 0
        self.fail_count = 0

        # Candidate set size
        self.n_cand = min(50 * self.dim, 2500)

    def initialize(self):
        X_init = np.random.uniform(self.lb, self.ub, size=(self.n_init, self.dim))
        f_init = []
        for x in X_init:
            f_init.append(self.f(x))
        f_init = np.array(f_init).reshape(-1, 1)

        self.X = np.vstack([self.X, X_init])
        self.fX = np.vstack([self.fX, f_init])
        self.n_evals += self.n_init

        idx_best = np.argmin(self.fX)
        self.best_f = float(self.fX[idx_best, 0])
        self.best_x = self.X[idx_best].copy()

    def _update_trust_region(self, fX_batch):
        if np.min(fX_batch) < self.best_f - 1e-3 * abs(self.best_f):
            self.success_count += 1
            self.fail_count = 0
        else:
            self.fail_count += 1
            self.success_count = 0

        if self.success_count >= self.success_tol:
            self.length = min(self.length * 2.0, self.length_max)
            self.success_count = 0
        if self.fail_count >= self.fail_tol:
            self.length = max(self.length / 2.0, self.length_min)
            self.fail_count = 0

    def _create_candidates(self, X_scaled, f_scaled):
        i_best = np.argmin(f_scaled)
        x_center = X_scaled[i_best:i_best+1, :]

        lb_tr = np.clip(x_center - self.length / 2.0, 0.0, 1.0)
        ub_tr = np.clip(x_center + self.length / 2.0, 0.0, 1.0)

        inside_mask = np.all((X_scaled >= lb_tr) & (X_scaled <= ub_tr), axis=1)
        X_local = X_scaled[inside_mask]
        f_local = f_scaled[inside_mask]

        # If too few points in TR, use all
        if X_local.shape[0] < 2 * self.pca_components:
            X_local = X_scaled
            f_local = f_scaled

        pca, X_reduced_local, X_mean = _weighted_pca(X_local, f_local, self.pca_components)

        # Train GP in reduced dim
        gp_model, gp_likelihood = train_gp(
            X_reduced_local,
            f_local.ravel(),
            n_training_steps=50,  # unchanged
            use_ard=self.use_ard
        )

        # Candidate set
        seed = np.random.randint(int(1e6))
        sobol = SobolEngine(self.dim, scramble=True, seed=seed)
        pert = sobol.draw(self.n_cand).numpy()
        X_cand = lb_tr + (ub_tr - lb_tr) * pert

        # Probability-based perturbation
        prob_perturb = min(20.0 / self.dim, 1.0)
        mask = np.random.rand(self.n_cand, self.dim) <= prob_perturb
        row_sum = np.sum(mask, axis=1)
        for i in range(len(row_sum)):
            if row_sum[i] == 0:
                j = np.random.randint(0, self.dim)
                mask[i, j] = True

        X_cand_masked = np.ones_like(X_cand) * x_center
        X_cand_masked[mask] = X_cand[mask]
        X_cand = X_cand_masked

        X_reduced_cand = _apply_pca_unweighted(X_cand, pca, X_mean)

        gp_model.eval()
        gp_likelihood.eval()
        with torch.no_grad():
            X_torch = torch.tensor(X_reduced_cand, dtype=torch.float32)
            dist = gp_likelihood(gp_model(X_torch))
            f_samps = dist.sample(torch.Size([1])).numpy().reshape(-1)

        idx_sorted = np.argsort(-f_samps)
        best_inds = idx_sorted[: self.batch_size]
        return X_cand[best_inds]

    def optimize(self):
        self.initialize()
        while self.n_evals < self.max_evals and self.length > self.length_min:
            X_scaled = to_unit_cube(self.X, self.lb, self.ub)
            f_scaled = self.fX.ravel()

            X_next_scaled = self._create_candidates(X_scaled, f_scaled)
            X_next = from_unit_cube(X_next_scaled, self.lb, self.ub)

            fX_next = []
            for xnew in X_next:
                fX_next.append(self.f(xnew))
            fX_next = np.array(fX_next).reshape(-1, 1)

            self.X = np.vstack([self.X, X_next])
            self.fX = np.vstack([self.fX, fX_next])
            self.n_evals += self.batch_size

            min_batch = np.min(fX_next)
            if min_batch < self.best_f:
                self.best_f = float(min_batch)
                self.best_x = X_next[np.argmin(fX_next)].copy()

            self._update_trust_region(fX_next)

        return self.best_x, self.best_f


# =============== Standalone TuRBO (single TR) ===============

class TurboOnly:
    def __init__(self, f, lb, ub, hyperparams):
        self.f = f
        self.lb = lb
        self.ub = ub
        self.dim = len(lb)

        # Unpack
        self.n_init = hyperparams["n_init"]
        self.max_evals = hyperparams["max_evals"]
        self.batch_size = hyperparams["batch_size"]
        self.length = hyperparams["length_init"]
        self.length_min = hyperparams["length_min"]
        self.length_max = hyperparams["length_max"]
        self.success_tol = hyperparams["success_tol"]
        self.fail_tol = hyperparams["fail_tol"]

        self.success_count = 0
        self.fail_count = 0

        self.X = np.zeros((0, self.dim))
        self.fX = np.zeros((0,1))
        self.n_evals = 0
        self.best_f = float('inf')
        self.best_x = None

    def _initialize(self):
        X_init = np.random.uniform(self.lb, self.ub, size=(self.n_init, self.dim))
        f_init = []
        for x in X_init:
            f_init.append(self.f(x))
        f_init = np.array(f_init).reshape(-1,1)

        self.X = np.vstack([self.X, X_init])
        self.fX = np.vstack([self.fX, f_init])
        self.n_evals += self.n_init

        ibest = np.argmin(self.fX)
        self.best_f = float(self.fX[ibest, 0])
        self.best_x = self.X[ibest].copy()

    def _update_trust_region(self, fX_next):
        if np.min(fX_next) < self.best_f - 1e-3 * abs(self.best_f):
            self.success_count += 1
            self.fail_count = 0
        else:
            self.fail_count += 1
            self.success_count = 0

        if self.success_count >= self.success_tol:
            self.length = min(self.length * 2.0, self.length_max)
            self.success_count = 0
        if self.fail_count >= self.fail_tol:
            self.length = max(self.length / 2.0, self.length_min)
            self.fail_count = 0

    def optimize(self):
        self._initialize()
        while self.n_evals < self.max_evals and self.length > self.length_min:
            lower_loc = np.maximum(self.lb, self.best_x - self.length/2)
            upper_loc = np.minimum(self.ub, self.best_x + self.length/2)

            X_next = np.random.uniform(lower_loc, upper_loc, size=(self.batch_size, self.dim))
            fX_next = []
            for xnew in X_next:
                fX_next.append(self.f(xnew))
            fX_next = np.array(fX_next).reshape(-1,1)

            self.X = np.vstack([self.X, X_next])
            self.fX = np.vstack([self.fX, fX_next])
            self.n_evals += self.batch_size

            mval = np.min(fX_next)
            if mval < self.best_f:
                self.best_f = float(mval)
                self.best_x = X_next[np.argmin(fX_next)].copy()

            self._update_trust_region(fX_next)

        return self.best_x, self.best_f


# =============== Standalone PCA-BO (simple) ===============

class SimplePCABO:
    def __init__(self, f, lb, ub, hyperparams):
        self.f = f
        self.lb = lb
        self.ub = ub
        self.dim = len(lb)

        self.n_init = hyperparams["n_init"]
        self.max_evals = hyperparams["max_evals"]
        self.batch_size = hyperparams["batch_size"]
        self.pca_components = hyperparams["pca_components"]
        self.use_ard = hyperparams["use_ard"]

        self.X = np.zeros((0, self.dim))
        self.fX = np.zeros((0, 1))
        self.n_evals = 0
        self.best_f = float('inf')
        self.best_x = None

    def initialize(self):
        X_init = np.random.uniform(self.lb, self.ub, size=(self.n_init, self.dim))
        f_init = []
        for x in X_init:
            f_init.append(self.f(x))
        f_init = np.array(f_init).reshape(-1,1)

        self.X = np.vstack([self.X, X_init])
        self.fX = np.vstack([self.fX, f_init])
        self.n_evals += self.n_init

        ibest = np.argmin(self.fX)
        self.best_f = float(self.fX[ibest, 0])
        self.best_x = self.X[ibest].copy()

    def optimize(self):
        self.initialize()
        while self.n_evals < self.max_evals:
            X_local = self.X
            y_local = self.fX.ravel()

            d_eff = min(self.pca_components, X_local.shape[1], X_local.shape[0])

            # Weighted PCA instead of plain PCA
            X_scaled = (X_local - self.lb) / (self.ub - self.lb)  # or do a 0-1 scaling if you wish
            pca, X_reduced, X_mean = _weighted_pca(
                X_scaled,  # the scaled design
                y_local,   # function values for weighting
                d_eff
            )

            # Now train the GP in the (weighted) PCA space
            gp_model, gp_likelihood = train_gp(
                X_reduced,
                y_local,
                n_training_steps=50,
                use_ard=self.use_ard
            )

            # Generate candidate points in the reduced space
            n_candidates = min(50 * d_eff, 2500)
            sobol = SobolEngine(d_eff, scramble=True, seed=np.random.randint(1e6))
            Z = sobol.draw(n_candidates).numpy() * 2.0 - 1.0

            # Predict from the GP
            gp_model.eval()
            gp_likelihood.eval()
            with torch.no_grad():
                Z_torch = torch.tensor(Z, dtype=torch.float32)
                dist = gp_likelihood(gp_model(Z_torch))
                f_samps = dist.sample(torch.Size([1])).numpy().reshape(-1)

            # A simple “EI‐like” approach is to pick those with the smallest predicted f-value 
            # or highest negative f_samps:
            idx_top = np.argsort(f_samps)[: self.batch_size]

            Z_next = Z[idx_top]

            # Map back from PCA space to original dimension
            # In your code you use `_apply_pca_unweighted(...)`. For example:
            X_scaled_next = _apply_pca_unweighted(Z_next, pca, X_mean)
            # Then unscale back to [lb, ub]
            X_next = X_scaled_next * (self.ub - self.lb) + self.lb

            # Evaluate
            fX_next = np.array([self.f(xi) for xi in X_next]).reshape(-1, 1)

            # Store them
            self.X = np.vstack([self.X, X_next])
            self.fX = np.vstack([self.fX, fX_next])
            self.n_evals += self.batch_size

            # Track best
            mbest = np.min(fX_next)
            if mbest < self.best_f:
                self.best_f = float(mbest)
                self.best_x = X_next[np.argmin(fX_next)].copy()

            if self.n_evals >= self.max_evals:
                break
        return self.best_x, self.best_f



# =============== AxUS & BAXUS placeholders ===============

class AxUSProjector:
    """
    Random embedding akin to AxUS (used by BAxUS).
    """

    def __init__(self, input_dim, target_dim, seed=0):
        self.input_dim = input_dim
        self.target_dim = target_dim
        self.rng = np.random.RandomState(seed)

        # For demonstration: a simple random ±1 embedding
        self.A = self.rng.choice([1.0, -1.0], size=(self.target_dim, self.input_dim))

    def project_up(self, Z):
        return np.matmul(Z, self.A)
    
    def increase_target_dimensionality(self, dims_and_bins):
        new_A_rows = []
        old_A = self.A
        keepers = set(range(self.target_dim)) - set(dims_and_bins.keys())

        # Keep old rows except splitted ones
        for i in sorted(list(keepers)):
            new_A_rows.append(old_A[i, :])

        # For splitted dims, replicate row with random sign flips
        for i in sorted(list(dims_and_bins.keys())):
            row_old = old_A[i, :]
            n_new = dims_and_bins[i]
            for _ in range(n_new):
                sign_flips = self.rng.choice([1.0, -1.0], size=row_old.shape)
                row_new = row_old * sign_flips
                new_A_rows.append(row_new)
        
        self.target_dim = len(new_A_rows)
        self.A = np.vstack(new_A_rows)


class BAXUS:
    """
    BAxUS-like class: random embedding -> trust region in latent space -> dimension-splitting.
    Retrains GP only every 10 steps (TRAIN_INTERVAL).
    """

    def __init__(self, f, lb, ub, hyperparams):
        self.f = f
        self.lb = lb
        self.ub = ub
        self.dim = len(lb)

        self.n_init = hyperparams["n_init"]
        self.max_evals = hyperparams["max_evals"]
        self.batch_size = hyperparams["batch_size"]

        self.target_dim = hyperparams["baxus_init_target_dim"]
        self.success_tol = hyperparams["success_tol"]
        self.fail_tol_factor = hyperparams["baxus_fail_tol_factor"]

        self.length = hyperparams["length_init"]
        self.length_min = hyperparams["length_min"]
        self.length_max = hyperparams["length_max"]

        self.use_ard = hyperparams["use_ard"]

        self.rng = np.random.RandomState(hyperparams["baxus_seed"])

        # Data
        self.X = np.zeros((0, self.dim))
        self.fX = np.zeros((0, 1))
        self.Z = np.zeros((0, self.target_dim))
        self.n_evals = 0
        self.best_f = float('inf')
        self.best_x = None

        self.success_count = 0
        self.fail_count = 0

        self.projector = AxUSProjector(input_dim=self.dim,
                                       target_dim=self.target_dim,
                                       seed=hyperparams["baxus_seed"])
        self.gp_model = None
        self.gp_likelihood = None

        self.TRAIN_INTERVAL = hyperparams["baxus_train_interval"]
        self.GP_TRAIN_STEPS = hyperparams["n_training_steps"]  # 50

    def _fail_tol(self):
        return max(self.fail_tol_factor, self.target_dim)

    def _initialize(self):
        Z_init = self.rng.uniform(-1.0, 1.0, size=(self.n_init, self.target_dim))
        X_up = self.projector.project_up(Z_init)

        # Map [-1,1] to [0,1] then to [lb, ub]
        X_mapped = 0.5 * (X_up + 1.0)
        X_mapped = X_mapped * (self.ub - self.lb) + self.lb

        fvals = []
        for x_ in X_mapped:
            fvals.append(self.f(x_))
        fvals = np.array(fvals).reshape(-1, 1)

        self.X = np.vstack([self.X, X_mapped])
        self.fX = np.vstack([self.fX, fvals])
        self.Z = np.vstack([self.Z, Z_init])
        self.n_evals += self.n_init

        ibest = np.argmin(self.fX)
        self.best_f = float(self.fX[ibest, 0])
        self.best_x = self.X[ibest].copy()

    def _update_trust_region(self, fX_batch):
        if np.min(fX_batch) < self.best_f - 1e-3 * abs(self.best_f):
            self.success_count += 1
            self.fail_count = 0
        else:
            self.fail_count += 1
            self.success_count = 0

        if self.success_count >= self.success_tol:
            self.length = min(self.length * 2.0, self.length_max)
            self.success_count = 0
        if self.fail_count >= self._fail_tol():
            self.length = max(self.length / 2.0, self.length_min)
            self.fail_count = 0

    def _choose_splitting_dims(self, gp_model):
        lengthscales = gp_model.covar_module.base_kernel.lengthscale.cpu().detach().numpy().flatten()
        importances = 1.0 / lengthscales
        sorted_idx = np.argsort(-importances)
        if self.target_dim >= self.dim:
            return {}
        best_dim = sorted_idx[0]
        dims_and_bins = {best_dim: 2}
        return dims_and_bins

    def _split_dimensions(self, dims_and_bins):
        self.projector.increase_target_dimensionality(dims_and_bins)
        old_td = self.target_dim
        self.target_dim = self.projector.target_dim

        # Reset the trust region and data
        self.Z = np.zeros((0, self.target_dim))
        self.X = np.zeros((0, self.dim))
        self.fX = np.zeros((0, 1))
        self.n_evals = 0
        self.best_f = float('inf')
        self.best_x = None
        self.length = 0.8

        self._initialize()

    def _create_candidates_and_select(self, gp_model, gp_likelihood):
        i_best = np.argmin(self.fX)
        z_best = self.Z[i_best:i_best+1, :]

        lb_loc = np.clip(z_best - self.length / 2.0, -1.0, 1.0)
        ub_loc = np.clip(z_best + self.length / 2.0, -1.0, 1.0)

        n_cand = min(50 * self.target_dim, 2500)
        sobol = SobolEngine(self.target_dim, scramble=True, seed=self.rng.randint(1e7))
        pert = sobol.draw(n_cand).numpy()
        Z_cand = lb_loc + (ub_loc - lb_loc) * pert

        prob_perturb = min(20.0 / self.target_dim, 1.0)
        mask = self.rng.rand(n_cand, self.target_dim) <= prob_perturb
        row_sum = np.sum(mask, axis=1)
        for i in range(len(row_sum)):
            if row_sum[i] == 0:
                j = self.rng.randint(0, self.target_dim)
                mask[i, j] = True

        Z_cand_masked = np.ones_like(Z_cand) * z_best
        Z_cand_masked[mask] = Z_cand[mask]
        Z_cand = Z_cand_masked

        gp_model.eval()
        gp_likelihood.eval()
        with torch.no_grad():
            Z_torch = torch.tensor(Z_cand, dtype=torch.float32)
            dist = gp_likelihood(gp_model(Z_torch))
            f_samps = dist.sample(torch.Size([1])).numpy().reshape(-1)

        idx_sorted = np.argsort(-f_samps)
        idx_top = idx_sorted[: self.batch_size]
        return Z_cand[idx_top]

    def optimize(self):
        self._initialize()

        while self.n_evals < self.max_evals and self.length > self.length_min:
            # Retrain GP every TRAIN_INTERVAL steps or if GP not trained yet
            if (self.n_evals == 0) or (self.n_evals % self.TRAIN_INTERVAL == 0) or (self.gp_model is None):
                Z_torch = torch.tensor(self.Z, dtype=torch.float32)
                f_torch = torch.tensor(self.fX.ravel(), dtype=torch.float32)

                self.gp_model, self.gp_likelihood = train_gp(
                    Z_torch.numpy(),
                    f_torch.numpy(),
                    n_training_steps=self.GP_TRAIN_STEPS,
                    use_ard=self.use_ard,
                    lr=0.01,
                )
                # Possibly split dimension
                if self.target_dim < self.dim and self.success_count == 0 and self.fail_count == 0:
                    dims_and_bins = self._choose_splitting_dims(self.gp_model)
                    if dims_and_bins:
                        self._split_dimensions(dims_and_bins)
                        # Re-train after splitting
                        Z_torch = torch.tensor(self.Z, dtype=torch.float32)
                        f_torch = torch.tensor(self.fX.ravel(), dtype=torch.float32)
                        self.gp_model, self.gp_likelihood = train_gp(
                            Z_torch.numpy(),
                            f_torch.numpy(),
                            n_training_steps=self.GP_TRAIN_STEPS,
                            use_ard=self.use_ard,
                            lr=0.01,
                        )

            Z_next = self._create_candidates_and_select(self.gp_model, self.gp_likelihood)
            X_up = self.projector.project_up(Z_next)

            X_mapped = 0.5 * (X_up + 1.0)
            X_mapped = X_mapped * (self.ub - self.lb) + self.lb

            fX_next = []
            for x_ in X_mapped:
                fX_next.append(self.f(x_))
            fX_next = np.array(fX_next).reshape(-1, 1)

            self.Z = np.vstack([self.Z, Z_next])
            self.X = np.vstack([self.X, X_mapped])
            self.fX = np.vstack([self.fX, fX_next])
            self.n_evals += self.batch_size

            min_batch = np.min(fX_next)
            if min_batch < self.best_f:
                self.best_f = float(min_batch)
                self.best_x = X_mapped[np.argmin(fX_next)].copy()

            self._update_trust_region(fX_next)

            if self.n_evals >= self.max_evals:
                break

        return self.best_x, self.best_f


# ============ Benchmark Loop ============

def run_one_experiment(method_name, fid, instance, dimension, n_reps=5, budget_multiplier=10):
    """
    Runs one method on a single (fid, instance, dimension) problem for n_reps times.
    Returns average runtime in seconds over the n_reps.
    """
    folder_name = f"data-{method_name}-f{fid}-dim{dimension}"
    times = []
    for rep in range(1, n_reps+1):
        problem = get_problem(fid=fid, instance=instance, dimension=dimension)

        log = ioh_logger.Analyzer(
            root=folder_name,
            algorithm_name=method_name,
            algorithm_info=f"{method_name}-f{fid}"
        )
        problem.attach_logger(log)

        max_evals = budget_multiplier * dimension

        # Prepare hyperparams
        local_hp = SHARED_PARAMS.copy()
        local_hp["max_evals"] = max_evals  # override
        # Start timing
        t0 = time.time()

        if method_name == "turbo_pca":
            opt = TuRBO_PCA_BO(
                f=problem,
                lb=np.array(problem.bounds.lb),
                ub=np.array(problem.bounds.ub),
                hyperparams=local_hp
            )
            opt.optimize()

        elif method_name == "turbo_only":
            opt = TurboOnly(
                f=problem,
                lb=np.array(problem.bounds.lb),
                ub=np.array(problem.bounds.ub),
                hyperparams=local_hp
            )
            opt.optimize()

        elif method_name == "pca_only":
            opt = SimplePCABO(
                f=problem,
                lb=np.array(problem.bounds.lb),
                ub=np.array(problem.bounds.ub),
                hyperparams=local_hp
            )
            opt.optimize()

        elif method_name == "baxus":
            opt = BAXUS(
                f=problem,
                lb=np.array(problem.bounds.lb),
                ub=np.array(problem.bounds.ub),
                hyperparams=local_hp
            )
            opt.optimize()
        else:
            raise ValueError(f"Unknown method: {method_name}")

        # Stop timing
        t1 = time.time()
        times.append(t1 - t0)

        problem.reset()

    return np.mean(times)


def main():
    # Example BBOB test set
    fids = [1, 8, 12, 15, 21]
    instances = [0, 1, 2]
    dimensions = [2, 10, 40, 100]
    methods = ["turbo_pca", "turbo_only", "pca_only", "baxus"]
    n_reps = 5
    budget_multiplier = 10

    total_runs = len(fids) * len(instances) * len(dimensions) * len(methods)
    run_count = 0
    t0_global = time.time()

    # Store average runtimes
    runtime_results = {}

    for fid in fids:
        for inst in instances:
            for dim in dimensions:
                for m in methods:
                    run_count += 1
                    print(f"\n[{run_count}/{total_runs}] method={m}, f{fid}, instance={inst}, dim={dim}")
                    sys.stdout.flush()

                    avg_time = run_one_experiment(
                        method_name=m,
                        fid=fid,
                        instance=inst,
                        dimension=dim,
                        n_reps=n_reps,
                        budget_multiplier=budget_multiplier
                    )
                    runtime_results[(m, fid, inst, dim)] = avg_time

                    elapsed = time.time() - t0_global
                    print(f"  -> Avg time over {n_reps} runs: {avg_time:.2f}s. Total elapsed: {elapsed:.2f}s.")

    print("\nAll experiments completed.")
    print("Logs are in data-<method>-f<fID>-dim<D>/ folders.\n")

    # ================== Plot the runtime differences ==================
    # For illustration, let's average across *all* fids, instances, dims for each method.
    method_avg_times = {}
    for m in methods:
        # Gather all keys for this method
        subset = [runtime_results[k] for k in runtime_results if k[0] == m]
        method_avg_times[m] = np.mean(subset) if len(subset) > 0 else 0.0

    plt.figure(figsize=(6, 4))
    plt.bar(method_avg_times.keys(), method_avg_times.values(), color=['blue', 'orange', 'green', 'red'])
    plt.ylabel("Average Runtime (s)")
    plt.title("Runtime Comparison Across Methods (All Problems)")
    plt.tight_layout()
    plt.savefig("runtime_comparison.png")


if __name__ == "__main__":
    main()
