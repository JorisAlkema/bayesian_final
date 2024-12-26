#!/usr/bin/env python3
# File: opdr1_modified.py
"""
Example script that:
  1) Implements a combined TuRBO + Weighted-PCA approach (with trust-region sampling).
  2) Includes placeholders for:
       - Standalone TuRBO
       - Standalone PCA-BO
       - BAXUS
  3) Benchmarks them on selected BBOB problems via the `ioh` package
  4) Saves results for IOHanalyzer
"""

import sys
import time
import numpy as np
import torch
import gpytorch
from torch.quasirandom import SobolEngine
from sklearn.decomposition import PCA
from scipy.stats import rankdata
from copy import deepcopy

# If you have ioh installed:
try:
    from ioh import get_problem, logger
    IOH_AVAILABLE = True
except ImportError:
    IOH_AVAILABLE = False

# ======================== Basic GP for demonstration ========================

class SimpleGP(gpytorch.models.ExactGP):
    """
    A minimal Gaussian Process model using GPyTorch.
    Subclass gpytorch.models.ExactGP for use with ExactMarginalLogLikelihood.
    """

    def __init__(self, train_x, train_y, likelihood, use_ard=True):
        """
        train_x: torch.Tensor of shape (N, d_reduced)
        train_y: torch.Tensor of shape (N,)
        likelihood: instance of gpytorch.likelihoods.GaussianLikelihood()
        use_ard: bool, whether to use ARD in the kernel
        """
        super().__init__(train_x, train_y, likelihood)
        
        self.mean_module = gpytorch.means.ZeroMean()

        # Figure out dimension from train_x
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
        """
        x: torch.Tensor of shape (..., d_reduced)
        returns a MultivariateNormal distribution
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train_gp(X, y, n_training_steps=50, use_ard=True, lr=0.01):
    """
    X: np.ndarray of shape (N, d_reduced)
    y: np.ndarray of shape (N,)
    Returns:
        model (ExactGP subclass)
        likelihood (GaussianLikelihood)
    """
    # Convert to torch tensors
    X_torch = torch.tensor(X, dtype=torch.float32)
    y_torch = torch.tensor(y, dtype=torch.float32)

    # Instantiate likelihood
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    # Instantiate the ExactGP model
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
    """Scale X from [lb, ub]^d to [0,1]^d."""
    return (X - lb) / (ub - lb)

def from_unit_cube(X_unit, lb, ub):
    """Scale X from [0,1]^d to [lb, ub]^d."""
    return X_unit * (ub - lb) + lb

def _weighted_pca(X_scaled, f_scaled, n_components):
    """
    Fit Weighted PCA on X_scaled (N,d), weighting each row by a function of the rank of f.
    Returns:
        pca: fitted PCA object
        X_reduced: shape (N, n_components)
        X_mean: centroid of X_scaled
    Notes:
      - Weighted PCA usually means: we rank f, compute weights w_i, then multiply each X_i by w_i.
      - We also subtract the mean from X before weighting (like normal PCA).
    """
    # Step 1: compute rank-based weights
    ranks = rankdata(f_scaled)  # rank 1..N
    N = len(f_scaled)
    weights = np.log(N) - np.log(ranks)
    weights /= np.sum(weights)

    # Step 2: center
    X_mean = X_scaled.mean(axis=0)
    X_centered = X_scaled - X_mean

    # Step 3: apply row-wise weighting
    # Each row X_centered[i,:] is multiplied by weights[i].
    # We'll store a weighted copy for PCA fitting:
    X_weighted = X_centered * weights.reshape(-1, 1)

    # Step 4: do PCA on the weighted data with randomized solver and parallel processing
    d_eff = min(n_components, X_scaled.shape[1], X_scaled.shape[0])
    pca = PCA(n_components=d_eff, svd_solver='randomized')
    X_reduced = pca.fit_transform(X_weighted)

    return pca, X_reduced, X_mean


def _apply_pca_unweighted(X_scaled, pca, X_mean):
    """
    Transform new points X_scaled (N, d) into PCA space using the same
    mean from training. We do NOT apply rank-based weighting for new points.
    """
    # Center by the same X_mean used during training:
    X_centered = X_scaled - X_mean
    X_reduced = pca.transform(X_centered)
    return X_reduced


# ======================== Combined TuRBO + Weighted PCA =========================
class TuRBO_PCA_BO:
    """
    Combined TuRBO + Weighted-PCA approach.
    - Maintains a single trust region in [0,1]^d (like TuRBO-1).
    - At each iteration, scales data to [0,1]^d, does Weighted PCA, trains a GP in the reduced dimension.
    - Generates candidate points in the scaled unit-cube trust region (like original TuRBO),
      partial-dimension "mask" approach, then picks the best batch by Thompson sampling in PCA space.
    """

    def __init__(
        self,
        f,
        lb,
        ub,
        n_init=5,
        max_evals=100,
        batch_size=1,
        use_ard=True,
        pca_components=2,
        length_init=0.8,
        length_min=0.5**7,
        length_max=1.6,
        success_tol=3,
        fail_tol=3,
    ):
        self.f = f
        self.lb = lb
        self.ub = ub
        self.dim = len(lb)

        self.n_init = n_init
        self.max_evals = max_evals
        self.batch_size = batch_size
        self.use_ard = use_ard
        self.pca_components = pca_components

        # TuRBO hyperparams
        self.length = length_init
        self.length_min = length_min
        self.length_max = length_max
        self.success_tol = success_tol
        self.fail_tol = fail_tol
        self.success_count = 0
        self.fail_count = 0

        # Data storage
        self.X = np.zeros((0, self.dim))
        self.fX = np.zeros((0, 1))
        self.best_x = None
        self.best_f = float('inf')
        self.n_evals = 0

        # For controlling how many candidate points we sample inside the TR
        # (like original TuRBO does "min(100*d, 5000)")
        self.n_cand = min(100 * self.dim, 5000)

    def initialize(self):
        """Random initialization of the design."""
        X_init = np.random.uniform(self.lb, self.ub, size=(self.n_init, self.dim))
        f_init = []
        for x in X_init:
            fx = self.f(x)
            f_init.append(fx)
        f_init = np.array(f_init).reshape(-1, 1)

        self.X = np.vstack([self.X, X_init])
        self.fX = np.vstack([self.fX, f_init])
        self.n_evals += self.n_init

        idx_best = np.argmin(self.fX)
        self.best_f = float(self.fX[idx_best, 0])  
        self.best_x = self.X[idx_best].copy()

    def _update_trust_region(self, fX_batch):
        """
        Expand / shrink the region based on improvement.
        """
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
        """
        Generate candidate points in [0,1]^d using trust region approach
        (like Turbo1._create_candidates), but we'll evaluate them with a GP in PCA space.
        Steps:
          1. Weighted PCA on (X_scaled, f_scaled) -> (pca, X_reduced).
          2. Train GP in the reduced dimension.
          3. Identify best point in X_scaled (lowest f_scaled).
          4. Build trust region around that point (side length = self.length).
          5. Create candidate set with partial-dim mask.
          6. Transform candidate set into PCA space, do Thompson sampling, keep top self.batch_size.
          7. Return (X_next in scaled space, updated GP/hypers, pca, etc.)
        """
        # 1) Weighted PCA
        pca, X_reduced, X_mean = _weighted_pca(X_scaled, f_scaled, n_components=self.pca_components)

        # 2) Train GP in the reduced dimension
        gp_model, gp_likelihood = train_gp(X_reduced, f_scaled.ravel(), n_training_steps=50, use_ard=self.use_ard)

        # 3) Identify best point in X_scaled
        i_best = np.argmin(f_scaled)
        x_center = X_scaled[i_best:i_best+1, :]  # shape (1, d)

        # 4) Build local bounds for trust region in scaled space
        lb_tr = np.clip(x_center - self.length / 2.0, 0.0, 1.0)
        ub_tr = np.clip(x_center + self.length / 2.0, 0.0, 1.0)

        # 5) Create candidate set with partial-dim mask
        #    We'll do what original TuRBO does: generate sobol in [lb_tr, ub_tr],
        #    then pick random subset of dims to actually perturb.
        seed = np.random.randint(int(1e6))
        sobol = SobolEngine(self.dim, scramble=True, seed=seed)
        pert = sobol.draw(self.n_cand).numpy()
        # scale from [0,1]^d to [lb_tr, ub_tr]
        X_cand = lb_tr + (ub_tr - lb_tr) * pert

        # partial-dim "masking" approach (like the original):
        prob_perturb = min(20.0 / self.dim, 1.0)
        mask = np.random.rand(self.n_cand, self.dim) <= prob_perturb
        # ensure at least one dimension is perturbed
        row_sum = np.sum(mask, axis=1)
        for i in range(len(row_sum)):
            if row_sum[i] == 0:
                j = np.random.randint(0, self.dim)
                mask[i, j] = True
        # enforce that the unmasked dims remain x_center's coords
        X_cand_masked = np.ones_like(X_cand) * x_center
        X_cand_masked[mask] = X_cand[mask]
        X_cand = X_cand_masked

        # 6) Thompson sampling in PCA space
        #    Transform candidates -> PCA space (unweighted).
        X_reduced_cand = _apply_pca_unweighted(X_cand, pca, X_mean)

        # sample from GP
        gp_model.eval()
        gp_likelihood.eval()
        with torch.no_grad():
            X_torch = torch.tensor(X_reduced_cand, dtype=torch.float32)
            dist = gp_likelihood(gp_model(X_torch))
            # single sample of shape (n_cand,)
            f_samps = dist.sample(torch.Size([1])).numpy().reshape(-1)
        # We want to MINIMIZE f, so the acquisition is -f_samps
        # pick top self.batch_size
        idx_sorted = np.argsort(-f_samps)  # descending
        best_inds = idx_sorted[: self.batch_size]
        X_next = X_cand[best_inds]

        return X_next, (pca, X_mean, gp_model, gp_likelihood)

    def optimize(self):
        self.initialize()
        while self.n_evals < self.max_evals and self.length > self.length_min:
            # Scale all data to [0,1]^d
            X_scaled = to_unit_cube(self.X, self.lb, self.ub)
            f_scaled = self.fX.ravel()

            # Create candidate points
            X_next_scaled, _ = self._create_candidates(X_scaled, f_scaled)

            # Map them back to original space
            X_next = from_unit_cube(X_next_scaled, self.lb, self.ub)

            # Evaluate f
            fX_next = []
            for xnew in X_next:
                val = self.f(xnew)
                fX_next.append(val)
            fX_next = np.array(fX_next).reshape(-1, 1)

            # Update global data
            self.X = np.vstack([self.X, X_next])
            self.fX = np.vstack([self.fX, fX_next])
            self.n_evals += self.batch_size

            # Update best
            min_batch = np.min(fX_next)
            if min_batch < self.best_f:
                self.best_f = float(min_batch)
                self.best_x = X_next[np.argmin(fX_next)].copy()

            # Update trust region
            self._update_trust_region(fX_next)

        return self.best_x, self.best_f


# ========================== Standalone TuRBO (minimal placeholder) ==========================
class TurboOnly:
    """
    Minimal example of a single-trust-region TuRBO approach.
    For demonstration only.
    """
    def __init__(
        self,
        f,
        lb,
        ub,
        n_init=5,
        max_evals=100,
        batch_size=1,
        length_init=0.8,
        length_min=0.5**7,
        length_max=1.6,
        success_tol=3,
        fail_tol=3,
    ):
        self.f = f
        self.lb = lb
        self.ub = ub
        self.dim = len(lb)

        self.n_init = n_init
        self.max_evals = max_evals
        self.batch_size = batch_size
        self.length = length_init
        self.length_min = length_min
        self.length_max = length_max
        self.success_tol = success_tol
        self.fail_tol = fail_tol

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
            fx = self.f(x)
            f_init.append(fx)
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
            # local region: [best_x - length/2, best_x + length/2]
            lower_loc = np.maximum(self.lb, self.best_x - self.length/2)
            upper_loc = np.minimum(self.ub, self.best_x + self.length/2)

            # sample random points in that region
            X_next = np.random.uniform(lower_loc, upper_loc, size=(self.batch_size, self.dim))
            fX_next = []
            for xnew in X_next:
                fv = self.f(xnew)
                fX_next.append(fv)
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


# ========================== Standalone PCA-BO (very simplified) ==========================
class SimplePCABO:
    """
    Minimal example of a PCA-based Bayesian Optimization without trust regions.
    """
    def __init__(self, f, lb, ub, n_init=5, max_evals=100, batch_size=1, pca_components=2, use_ard=True):
        self.f = f
        self.lb = lb
        self.ub = ub
        self.dim = len(lb)

        self.n_init = n_init
        self.max_evals = max_evals
        self.batch_size = batch_size
        self.pca_components = pca_components
        self.use_ard = use_ard

        self.X = np.zeros((0, self.dim))
        self.fX = np.zeros((0, 1))
        self.n_evals = 0
        self.best_f = float('inf')
        self.best_x = None

    def initialize(self):
        X_init = np.random.uniform(self.lb, self.ub, size=(self.n_init, self.dim))
        f_init = []
        for x in X_init:
            fv = self.f(x)
            f_init.append(fv)
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
            d_eff = min(self.pca_components, X_local.shape[0], self.dim)

            X_mean = X_local.mean(axis=0)
            X_centered = X_local - X_mean
            pca = PCA(n_components=d_eff, svd_solver='randomized')
            X_reduced = pca.fit_transform(X_centered)

            gp_model, gp_likelihood = train_gp(X_reduced, y_local, n_training_steps=50, use_ard=self.use_ard)

            # generate candidates in reduced space
            n_candidates = 100 * d_eff
            sobol = SobolEngine(d_eff, scramble=True, seed=np.random.randint(1e6))
            Z = sobol.draw(n_candidates).numpy() * 2.0 - 1.0  # in [-1,1]^d_eff

            # Thompson sampling
            gp_model.eval()
            gp_likelihood.eval()
            Z_torch = torch.tensor(Z, dtype=torch.float32)
            with torch.no_grad():
                dist = gp_likelihood(gp_model(Z_torch))
                f_samps = dist.sample(torch.Size([1])).numpy().reshape(-1)
            acq_vals = -f_samps

            idx_top = np.argsort(-acq_vals)[: self.batch_size]
            Z_next = Z[idx_top]
            X_centered_next = pca.inverse_transform(Z_next)
            X_next = X_centered_next + X_mean

            fX_next = []
            for xx in X_next:
                fv = self.f(xx)
                fX_next.append(fv)
            fX_next = np.array(fX_next).reshape(-1,1)

            self.X = np.vstack([self.X, X_next])
            self.fX = np.vstack([self.fX, fX_next])
            self.n_evals += self.batch_size

            mbest = np.min(fX_next)
            if mbest < self.best_f:
                self.best_f = float(mbest)
                self.best_x = X_next[np.argmin(fX_next)].copy()

            if self.n_evals >= self.max_evals:
                break

        return self.best_x, self.best_f


# ========================== Minimal BAXUS Placeholder ==========================
class BAXUS:
    """
    Placeholder for a BAXUS-like approach.
    In reality, BAXUS uses random embeddings, random affine transformations, etc.
    We'll just illustrate a naive approach that picks random subspaces for demonstration.
    """

    def __init__(self, f, lb, ub, n_init=5, max_evals=100, batch_size=1, embed_dim=5):
        self.f = f
        self.lb = lb
        self.ub = ub
        self.dim = len(lb)

        self.n_init = n_init
        self.max_evals = max_evals
        self.batch_size = batch_size
        self.embed_dim = min(embed_dim, self.dim)

        self.X = np.zeros((0, self.dim))
        self.fX = np.zeros((0,1))
        self.n_evals = 0
        self.best_f = float('inf')
        self.best_x = None

        # pick random subset of dimensions for embedding
        self.active_dims = np.random.choice(self.dim, size=self.embed_dim, replace=False)

    def initialize(self):
        X_init = np.random.uniform(self.lb, self.ub, size=(self.n_init, self.dim))
        f_init = []
        for x in X_init:
            fv = self.f(x)
            f_init.append(fv)
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
            # focus on random subspace
            # sample in that subspace, keep others the same as best_x?
            X_next = []
            for _ in range(self.batch_size):
                xnew = np.array(self.best_x)
                for d in self.active_dims:
                    xnew[d] = np.random.uniform(self.lb[d], self.ub[d])
                X_next.append(xnew)
            X_next = np.array(X_next)

            fX_next = []
            for x_ in X_next:
                fv = self.f(x_)
                fX_next.append(fv)
            fX_next = np.array(fX_next).reshape(-1,1)

            self.X = np.vstack([self.X, X_next])
            self.fX = np.vstack([self.fX, fX_next])
            self.n_evals += self.batch_size

            mbest = np.min(fX_next)
            if mbest < self.best_f:
                self.best_f = float(mbest)
                self.best_x = X_next[np.argmin(fX_next)].copy()

        return self.best_x, self.best_f


# ========================== Benchmark Loop with timing ==========================
def run_one_experiment(method_name, fid, instance, dimension, n_reps=5, budget_multiplier=10):
    """
    Runs a single method on one BBOB problem (fid, instance, dimension) for n_reps times
    and logs results for IOHanalyzer using the ioh logger.
    """
    if not IOH_AVAILABLE:
        print("ioh not installed, skipping actual BBOB experiment.")
        return

    from ioh import logger as ioh_logger
    folder_name = f"data-{method_name}-f{fid}-dim{dimension}"
    for rep in range(1, n_reps+1):
        problem = get_problem(fid=fid, instance=instance, dimension=dimension)

        # Attach logger
        log = ioh_logger.Analyzer(
            root=folder_name, 
            algorithm_name=method_name,
            algorithm_info=f"{method_name}-f{fid}"
        )
        problem.attach_logger(log)

        max_evals = budget_multiplier * dimension

        # Prepare bounds for convenience
        lb = np.array(problem.bounds.lb)
        ub = np.array(problem.bounds.ub)

        # Initialize method
        if method_name == "turbo_pca":
            opt = TuRBO_PCA_BO(
                f=problem,
                lb=lb,
                ub=ub,
                n_init=5,
                max_evals=max_evals,
                batch_size=1,
                use_ard=True,
                pca_components=2,
            )
            opt.optimize()

        elif method_name == "turbo_only":
            opt = TurboOnly(
                f=problem,
                lb=lb,
                ub=ub,
                n_init=5,
                max_evals=max_evals,
                batch_size=1
            )
            opt.optimize()

        elif method_name == "pca_only":
            opt = SimplePCABO(
                f=problem,
                lb=lb,
                ub=ub,
                n_init=5,
                max_evals=max_evals,
                batch_size=1,
                pca_components=2,
                use_ard=True
            )
            opt.optimize()

        elif method_name == "baxus":
            opt = BAXUS(
                f=problem,
                lb=lb,
                ub=ub,
                n_init=5,
                max_evals=max_evals,
                batch_size=1,
                embed_dim=5
            )
            opt.optimize()

        else:
            raise ValueError(f"Unknown method: {method_name}")

        # Ensure the run is recorded
        problem.reset()  # detach logger


def main():
    # BBOB test set (example)
    fids = [15]
    instances = [0, 1, 2]
    dimensions = [2, 10, 40, 100]
    methods = ["turbo_pca", "turbo_only", "pca_only", "baxus"]
    n_reps = 5
    budget_multiplier = 1

    # For simple progress estimates
    total_runs = len(fids) * len(instances) * len(dimensions) * len(methods)
    run_count = 0
    t0 = time.time()

    for fid in fids:
        for inst in instances:
            for dim in dimensions:
                for m in methods:
                    run_count += 1
                    start_time = time.time()
                    
                    print(f"\n[{run_count}/{total_runs}] Running method={m}, f{fid}, instance={inst}, dim={dim}")
                    sys.stdout.flush()

                    run_one_experiment(m, fid, inst, dim, n_reps, budget_multiplier)
                    
                    end_time = time.time()
                    duration = end_time - start_time
                    total_elapsed = end_time - t0
                    print(f"  -> Completed in {duration:.2f} seconds. Total elapsed: {total_elapsed:.2f} seconds.")

    print("\nAll experiments completed.")
    print("Logs are in data-<method>-f<fID>-dim<D>/ folders.")


if __name__ == "__main__":
    main()
