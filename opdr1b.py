#!/usr/bin/env python3
# File: opdr1_modified_sharedparams.py
"""
Example script that:
1) Implements a combined TuRBO + Weighted-PCA approach (LOCAL trust-region PCA).
2) Includes 3 other baseline algorithms:
    - Standalone TuRBO
    - Standalone PCA-BO
    - BAXUS
3) Benchmarks them on selected BBOB problems via the `ioh` package
4) Saves results for IOHanalyzer
5) Plots runtime comparisons among the 4 methods.
"""

import sys
import time
import math
import numpy as np
import torch
from torch.quasirandom import SobolEngine
from sklearn.decomposition import PCA
from scipy.stats import rankdata, qmc
import matplotlib.pyplot as plt
from ioh import get_problem
from ioh import logger as ioh_logger
from gpytorch.constraints import Interval
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP

# ====================== SHARED HYPERPARAMETERS =====================
SHARED_PARAMS = {
    # Generic BO hyperparams
    "n_init": 10,          # Number of initial points
    "batch_size": 32,
    "use_ard": True,
    "pca_components": 2,  # For PCA-based methods
    # TuRBO trust-region settings
    "length_init": 0.8,
    "length_min": 0.5 ** 7,
    "length_max": 3.2,     # Increased from 1.6
    "success_tol": 4,
    # GP training
    "n_training_steps": 50,
    "lr": 0.005, 
    # Candidate set
    # Logic: n_cand = min(100*d, 2500) kept inside each class
    # BAXUS-specific
    "baxus_train_interval": 5,  # Only train GP every 5 steps
    "baxus_fail_tol_factor": 4.0,
    "baxus_init_target_dim": 5,
    "baxus_seed": 0,
    "max_evals_multiplier": 5,  # Used to calculate max_evals
}

# ======================== Correct GP Implementation ========================

class GP(ExactGP):
    """
    A Gaussian Process model using GPyTorch with configurable hyperparameters and constraints.
    Inherits from gpytorch.models.ExactGP for use with ExactMarginalLogLikelihood.
    """
    def __init__(self, train_x, train_y, likelihood, lengthscale_constraint, outputscale_constraint, ard_dims):
        super(GP, self).__init__(train_x, train_y, likelihood)
        self.ard_dims = ard_dims
        self.mean_module = ConstantMean()
        base_kernel = MaternKernel(lengthscale_constraint=lengthscale_constraint, ard_num_dims=ard_dims, nu=2.5)
        self.covar_module = ScaleKernel(base_kernel, outputscale_constraint=outputscale_constraint)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


def train_gp(train_x, train_y, use_ard, num_steps, hypers={}, device=None, dtype=torch.float):
    """
    Fit a Gaussian Process model where train_x is in [0, 1]^d and train_y is standardized.

    Parameters:
    - train_x (torch.Tensor): Training inputs of shape (n_samples, n_features)
    - train_y (torch.Tensor): Training targets of shape (n_samples,)
    - use_ard (bool): Whether to use Automatic Relevance Determination
    - num_steps (int): Number of training iterations
    - hypers (dict): Optional hyperparameter initializations
    - device (torch.device): Device to run the GP on
    - dtype (torch.dtype): Data type for tensors

    Returns:
    - model (GP): Trained GP model
    - likelihood (GaussianLikelihood): Trained likelihood
    """

    # Create hyperparameter bounds
    noise_constraint = Interval(5e-4, 0.2)
    if use_ard:
        lengthscale_constraint = Interval(0.005, 2.0)
    else:
        lengthscale_constraint = Interval(0.005, math.sqrt(train_x.shape[1]))  # [0.005, sqrt(dim)]
    outputscale_constraint = Interval(0.05, 20.0)

    # Initialize likelihood with noise constraints
    likelihood = GaussianLikelihood(noise_constraint=noise_constraint).to(device=device, dtype=dtype)
    
    ard_dims = train_x.shape[1] if use_ard else None
    model = GP(
        train_x=train_x,
        train_y=train_y,
        likelihood=likelihood,
        lengthscale_constraint=lengthscale_constraint,
        outputscale_constraint=outputscale_constraint,
        ard_dims=ard_dims,
    ).to(device=device, dtype=dtype)

    # Initialize model hyperparameters
    if hypers:
        model.load_state_dict(hypers)
    else:
        hypers = {
            "covar_module.outputscale": 1.0,
            "covar_module.base_kernel.lengthscale": 0.5,
            "likelihood.noise": 0.005,
        }
        model.initialize(**hypers)

    model.train()
    likelihood.train()

    # Define the Marginal Log Likelihood
    mll = ExactMarginalLogLikelihood(likelihood, model)

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=SHARED_PARAMS["lr"])

    for _ in range(num_steps):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

    # Switch to evaluation mode
    model.eval()
    likelihood.eval()

    return model, likelihood

# ======================== Helpers for Scaling & PCA =========================

def to_unit_cube(X, lb, ub):
    """
    Scale input data to the unit cube [0, 1]^d.

    Parameters:
    - X (np.ndarray): Input data of shape (n_samples, n_features)
    - lb (np.ndarray): Lower bounds of shape (n_features,)
    - ub (np.ndarray): Upper bounds of shape (n_features,)

    Returns:
    - X_unit (np.ndarray): Scaled data
    """
    return (X - lb) / (ub - lb)


def from_unit_cube(X_unit, lb, ub):
    """
    Scale data from the unit cube [0, 1]^d back to original bounds.

    Parameters:
    - X_unit (np.ndarray): Scaled data of shape (n_samples, n_features)
    - lb (np.ndarray): Lower bounds of shape (n_features,)
    - ub (np.ndarray): Upper bounds of shape (n_features,)

    Returns:
    - X (np.ndarray): Unscaled data
    """
    return X_unit * (ub - lb) + lb


def _weighted_pca(X_scaled, f_scaled, n_components):
    """
    Perform Weighted PCA using rank-based weights.

    Parameters:
    - X_scaled (np.ndarray): Scaled input data of shape (n_samples, n_features)
    - f_scaled (np.ndarray): Scaled function values of shape (n_samples,)
    - n_components (int): Number of PCA components

    Returns:
    - pca (PCA): Fitted PCA object
    - X_reduced (np.ndarray): PCA-transformed data
    - X_mean (np.ndarray): Mean of the original data
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

def _apply_pca_unweighted(X_reduced, pca, X_mean):
    """
    Apply inverse PCA transformation to map reduced data back to the original space.

    Parameters:
    - X_reduced (np.ndarray): PCA-reduced data of shape (n_samples, n_components)
    - pca (PCA): Fitted PCA object
    - X_mean (np.ndarray): Mean of the original scaled data of shape (n_features,)

    Returns:
    - X_original (np.ndarray): Data transformed back to the original space of shape (n_samples, n_features)
    """
    # Inverse transform to original space
    X_original = pca.inverse_transform(X_reduced)
    
    # Add the mean back to the original space
    X_original += X_mean
    
    return X_original


def latin_hypercube(n_samples, dim):
    """
    Generate samples using the Latin Hypercube Sampling method.

    Parameters:
    - n_samples (int): Number of samples to generate
    - dim (int): Dimensionality of the samples

    Returns:
    - sample (np.ndarray): Generated samples of shape (n_samples, dim)
    """
    sampler = qmc.LatinHypercube(d=dim)
    sample = sampler.random(n=n_samples)
    return sample

# ================== TuRBO_PCA_BO Class with Proper Restart Mechanism ==================



class TuRBO_PCA_BO:
    """
    TuRBO combined with Weighted-PCA approach for Bayesian Optimization.
    Utilizes a local trust region and PCA for dimensionality reduction.
    """

    def __init__(self, f, lb, ub, hyperparams, verbose=False):
        """
        Initialize the TuRBO_PCA_BO optimizer.

        Parameters:
        - f (callable): Objective function to minimize
        - lb (np.ndarray): Lower bounds of the search space
        - ub (np.ndarray): Upper bounds of the search space
        - hyperparams (dict): Hyperparameters from SHARED_PARAMS
        - verbose (bool): If True, prints progress messages
        """
        self.f = f
        self.lb = lb
        self.ub = ub
        self.dim = len(lb)

        # Unpack shared hyperparams with default values
        self.n_init = hyperparams.get("n_init", 2 * self.dim)
        self.max_evals = hyperparams.get("max_evals_multiplier", 10) * self.dim
        self.batch_size = hyperparams.get("batch_size", 1)
        self.use_ard = hyperparams.get("use_ard", True)
        self.pca_components = hyperparams.get("pca_components", min(5, self.dim))
        self.length_init = hyperparams.get("length_init", 0.8)
        self.length_min = hyperparams.get("length_min", 0.5 ** 7)
        self.length_max = hyperparams.get("length_max", 1.6)
        self.success_tol = hyperparams.get("success_tol", 3)
        self.fail_tol = math.ceil(max(4.0 / self.batch_size, self.dim / self.batch_size))
        self.lr = hyperparams.get("lr", 0.01)
        self.n_training_steps = hyperparams.get("n_training_steps", 50)
        self.verbose = verbose

        # Device and dtype for GPyTorch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32  # Consistently use float32

        if self.verbose:
            print(f"Using device: {self.device}, dtype: {self.dtype}")

        # Data storage
        self.X = np.empty((0, self.dim))
        self.fX = np.empty((0, 1))
        self.best_x = None
        self.best_f = float('inf')
        self.n_evals = 0

        self.success_count = 0
        self.fail_count = 0

        # Candidate set size    
        self.n_cand = min(100 * self.dim, 2500)

        # Current trust region length
        self.length = self.length_init

        # Verbosity
        self.verbose = verbose

        # Hyperparameters storage
        self.hypers = {}

    def initialize(self, restart=True):
        """
        Initialize or restart the optimizer by sampling new points.

        Parameters:
        - restart (bool): If True, replaces existing data with new samples
        """
        if restart:
            # Restart: Sample new initial points using Latin Hypercube
            X_init = latin_hypercube(self.n_init, self.dim)
            X_init = from_unit_cube(X_init, self.lb, self.ub)
            f_init = np.array([self.f(x) for x in X_init]).reshape(-1, 1)

            # Replace existing data
            self.X = X_init
            self.fX = f_init

            # Reset counters
            self.success_count = 0
            self.fail_count = 0
        else:
            # Initial initialization: Latin Hypercube Sampling
            X_init = latin_hypercube(self.n_init, self.dim)
            X_init = from_unit_cube(X_init, self.lb, self.ub)
            f_init = np.array([self.f(x) for x in X_init]).reshape(-1, 1)

            # Append during initial initialization
            self.X = np.vstack([self.X, X_init])
            self.fX = np.vstack([self.fX, f_init])

        self.n_evals += self.n_init

        idx_best = np.argmin(self.fX)
        if self.fX[idx_best, 0] < self.best_f:
            self.best_f = float(self.fX[idx_best, 0])
            self.best_x = self.X[idx_best].copy()

        if self.verbose:
            print(f"{'Restarting' if restart else 'Initializing'} with {self.n_init} points. Best f: {self.best_f:.4f}")

    def _update_trust_region(self, fX_batch):
        """
        Update the trust region based on the latest batch of function evaluations.

        Parameters:
        - fX_batch (np.ndarray): Latest function evaluations of shape (batch_size, 1)
        """
        improvement = np.min(fX_batch) < self.best_f - 1e-3 * abs(self.best_f)
        if improvement:
            self.success_count += 1
            self.fail_count = 0
            if self.verbose:
                print(f"Success count increased to {self.success_count}")
        else:
            self.fail_count += 1
            self.success_count = 0
            if self.verbose:
                print(f"Fail count increased to {self.fail_count}")

        if self.success_count >= self.success_tol:
            old_length = self.length
            self.length = min(self.length * 2.0, self.length_max)
            self.success_count = 0
            if self.verbose:
                print(f"Expanding trust region from {old_length:.4f} to {self.length:.4f}")
        if self.fail_count >= self.fail_tol:
            old_length = self.length
            self.length = max(self.length / 2.0, self.length_min)
            self.fail_count = 0
            if self.verbose:
                print(f"Shrinking trust region from {old_length:.4f} to {self.length:.4f}")

    def _create_candidates(self, X_scaled, f_scaled):
        """
        Generate candidate points within the trust region using the GP model.

        Parameters:
        - X_scaled (np.ndarray): Scaled input data of shape (n_samples, n_features)
        - f_scaled (np.ndarray): Scaled function values of shape (n_samples,)

        Returns:
        - X_cand_selected_original (np.ndarray): Selected candidate points in original space of shape (batch_size, n_features)
        """
        # Standardize function values
        mu, sigma = np.median(f_scaled), f_scaled.std()
        sigma = 1.0 if sigma < 1e-6 else sigma
        f_standardized = (f_scaled - mu) / sigma

        if self.verbose:
            print("Performing Weighted PCA on local points...")
        
        # Perform Weighted PCA
        pca, X_reduced_local, X_mean = _weighted_pca(X_scaled, f_standardized, self.pca_components)

        if self.verbose:
            print(f"PCA completed. Reduced to {self.pca_components} components.")

        # Train GP on reduced and standardized data
        train_x = torch.tensor(X_reduced_local, dtype=self.dtype).to(self.device)
        train_y = torch.tensor(f_standardized, dtype=self.dtype).to(self.device)
        if self.verbose:
            print("Training GP on reduced and standardized data...")
        gp_model, gp_likelihood = train_gp(
            train_x=train_x,
            train_y=train_y,
            use_ard=self.use_ard,
            num_steps=self.n_training_steps,
            hypers=self.hypers,
            device=self.device,
            dtype=self.dtype
        )
        if self.verbose:
            print("GP training completed.")

        # Update hyperparameters
        self.hypers = gp_model.state_dict()

        # Generate candidate points using Sobol sequence in reduced space
        sobol = SobolEngine(self.pca_components, scramble=True)
        pert = sobol.draw(self.n_cand).numpy()
        X_cand_reduced = pert  # Candidates in reduced PCA space

        # Apply probability-based perturbation
        prob_perturb = min(20.0 / self.pca_components, 1.0)
        mask = np.random.rand(self.n_cand, self.pca_components) <= prob_perturb
        row_sum = np.sum(mask, axis=1)
        for i in range(len(row_sum)):
            if row_sum[i] == 0:
                j = np.random.randint(0, self.pca_components)
                mask[i, j] = True

        X_cand_masked = np.ones_like(X_cand_reduced) * X_reduced_local[np.argmin(f_standardized), :]
        X_cand_masked[mask] = X_cand_reduced[mask]
        X_cand_reduced = X_cand_masked

        if self.verbose:
            print("Evaluating GP predictions on candidate points...")

        # Evaluate GP predictions on candidates
        gp_model.eval()
        gp_likelihood.eval()
        with torch.no_grad():
            X_torch = torch.tensor(X_cand_reduced, dtype=self.dtype).to(self.device)
            dist = gp_likelihood(gp_model(X_torch))
            f_samps_standardized = dist.sample(torch.Size([1])).cpu().numpy().reshape(-1)
            f_samps = mu + sigma * f_samps_standardized  # Transform back to original scale

        # Select the best candidates based on sampled function values
        idx_sorted = np.argsort(f_samps)
        best_inds = idx_sorted[:self.batch_size]
        X_cand_selected_reduced = X_cand_reduced[best_inds]

        # Inverse PCA to original space
        X_cand_selected_original = _apply_pca_unweighted(X_cand_selected_reduced, pca, X_mean)

        if self.verbose:
            print(f"Selected {self.batch_size} candidate points.")

        return X_cand_selected_original

    def _restart_condition(self):
        """
        Determine if a restart is needed based on the current trust region size and evaluations.

        Returns:
        - bool: True if a restart is needed, False otherwise
        """
        return self.length <= self.length_min and self.n_evals < self.max_evals

    def optimize(self):
        """
        Run the optimization process until the evaluation budget is exhausted.
        
        Returns:
        - best_x (np.ndarray): Best found input
        - best_f (float): Best found function value
        """
        # Initial optimization
        self.length = self.length_init
        self.initialize(restart=False)

        while self.n_evals < self.max_evals:
            while self.n_evals < self.max_evals and self.length > self.length_min:
                # Scale X to unit cube
                X_scaled = to_unit_cube(self.X, self.lb, self.ub)
                f_scaled = self.fX.ravel()

                # Generate next candidates
                X_next = self._create_candidates(X_scaled, f_scaled)

                # Scale candidates back to original space
                X_next_original = from_unit_cube(X_next, self.lb, self.ub)

                # Evaluate new points
                fX_next = np.array([self.f(xnew) for xnew in X_next_original]).reshape(-1, 1)

                # Update data
                self.X = np.vstack([self.X, X_next_original])
                self.fX = np.vstack([self.fX, fX_next])
                self.n_evals += self.batch_size

                # Update best found point
                min_batch = np.min(fX_next)
                if min_batch < self.best_f:
                    self.best_f = float(min_batch)
                    self.best_x = X_next_original[np.argmin(fX_next)].copy()
                    if self.verbose:
                        print(f"New best f: {self.best_f:.4f}")

                # Update trust region
                self._update_trust_region(fX_next)

            # Check if restart is needed
            if self._restart_condition():
                if self.verbose:
                    print(f"Restarting TuRBO_PCA_BO: {self.n_evals}/{self.max_evals} evaluations used.")
                # Restart by sampling new points across the entire search space
                self.initialize(restart=True)
                # Reset trust region length
                self.length = self.length_init
            else:
                break  # No more evaluations or cannot restart

        return self.best_x, self.best_f

class TurboOnly:
    """
    Standalone TuRBO (Trust Region Bayesian Optimization) implementation.
    Utilizes a local trust region without PCA-based dimensionality reduction.
    """

    def __init__(self, f, lb, ub, hyperparams, verbose=False):
        """
        Initialize the TurboOnly optimizer.

        Parameters:
        - f (callable): Objective function to minimize.
        - lb (np.ndarray): Lower bounds of the search space.
        - ub (np.ndarray): Upper bounds of the search space.
        - hyperparams (dict): Hyperparameters from SHARED_PARAMS.
        - verbose (bool): If True, prints progress messages.
        """
        self.f = f
        self.lb = lb
        self.ub = ub
        self.dim = len(lb)

        # Unpack shared hyperparams
        self.n_init = hyperparams.get("n_init", 2 * self.dim)
        self.max_evals = hyperparams.get("max_evals_multiplier", 10) * self.dim
        self.batch_size = hyperparams.get("batch_size", 1)
        self.length_init = hyperparams.get("length_init", 0.8)
        self.length_min = hyperparams.get("length_min", 0.5 ** 7)
        self.length_max = hyperparams.get("length_max", 1.6)
        self.success_tol = hyperparams.get("success_tol", 3)
        self.fail_tol = math.ceil(max(4.0 / self.batch_size, self.dim / self.batch_size))
        self.lr = hyperparams.get("lr", 0.01)
        self.n_training_steps = hyperparams.get("n_training_steps", 50)
        self.use_ard = hyperparams.get("use_ard", True)

        # Data storage
        self.X = np.empty((0, self.dim))
        self.fX = np.empty((0, 1))
        self.best_x = None
        self.best_f = float('inf')
        self.n_evals = 0

        self.success_count = 0
        self.fail_count = 0

        # Candidate set size
        self.n_cand = min(100 * self.dim, 5000)

        # Current trust region length
        self.length = self.length_init

        # Verbosity
        self.verbose = verbose

    def initialize(self, restart=True):
        """
        Initialize or restart the optimizer by sampling new points.

        Parameters:
        - restart (bool): If True, replaces existing data with new samples.
        """
        if restart:
            # Restart: Sample new initial points using Latin Hypercube
            X_init = latin_hypercube(self.n_init, self.dim)
            X_init = from_unit_cube(X_init, self.lb, self.ub)
            f_init = np.array([self.f(x) for x in X_init]).reshape(-1, 1)

            # Replace existing data
            self.X = X_init
            self.fX = f_init

            # Reset counters
            self.success_count = 0
            self.fail_count = 0
        else:
            # Initial initialization: Latin Hypercube Sampling
            X_init = latin_hypercube(self.n_init, self.dim)
            X_init = from_unit_cube(X_init, self.lb, self.ub)
            f_init = np.array([self.f(x) for x in X_init]).reshape(-1, 1)

            # Append during initial initialization
            self.X = np.vstack([self.X, X_init])
            self.fX = np.vstack([self.fX, f_init])

        self.n_evals += self.n_init

        idx_best = np.argmin(self.fX)
        if self.fX[idx_best, 0] < self.best_f:
            self.best_f = float(self.fX[idx_best, 0])
            self.best_x = self.X[idx_best].copy()

        if self.verbose:
            print(f"{'Restarting' if restart else 'Initializing'} with {self.n_init} points. Best f: {self.best_f:.4f}")

    def _update_trust_region(self, fX_batch):
        """
        Update the trust region based on the latest batch of function evaluations.

        Parameters:
        - fX_batch (np.ndarray): Latest function evaluations of shape (batch_size, 1).
        """
        improvement = np.min(fX_batch) < self.best_f - 1e-3 * abs(self.best_f)
        if improvement:
            self.success_count += 1
            self.fail_count = 0
            if self.verbose:
                print(f"Success count increased to {self.success_count}")
        else:
            self.fail_count += 1
            self.success_count = 0
            if self.verbose:
                print(f"Fail count increased to {self.fail_count}")

        if self.success_count >= self.success_tol:
            old_length = self.length
            self.length = min(self.length * 2.0, self.length_max)
            self.success_count = 0
            if self.verbose:
                print(f"Expanding trust region from {old_length:.4f} to {self.length:.4f}")
        if self.fail_count >= self.fail_tol:
            old_length = self.length
            self.length = max(self.length / 2.0, self.length_min)
            self.fail_count = 0
            if self.verbose:
                print(f"Shrinking trust region from {old_length:.4f} to {self.length:.4f}")

    def _create_candidates(self, X_scaled, f_scaled, hypers):
        """
        Generate candidate points within the trust region using the GP model.

        Parameters:
        - X_scaled (np.ndarray): Scaled input data of shape (n_samples, n_features).
        - f_scaled (np.ndarray): Scaled function values of shape (n_samples,).
        - hypers (dict): Current hyperparameters for the GP.

        Returns:
        - X_cand_selected (np.ndarray): Selected candidate points of shape (batch_size, n_features).
        """
        # Standardize function values
        mu, sigma = np.median(f_scaled), f_scaled.std()
        sigma = 1.0 if sigma < 1e-6 else sigma
        f_standardized = (f_scaled - mu) / sigma

        # Train GP on standardized data
        train_x = torch.tensor(X_scaled, dtype=torch.float32)
        train_y = torch.tensor(f_standardized, dtype=torch.float32)
        gp_model, gp_likelihood = train_gp(
            train_x=train_x,
            train_y=train_y,
            use_ard=self.use_ard,
            num_steps=self.n_training_steps,
            hypers=hypers,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            dtype=torch.float32
        )

        # Find the best point
        i_best = np.argmin(f_scaled)
        x_center = X_scaled[i_best:i_best+1, :]

        # Define trust region bounds
        lb_tr = np.clip(x_center - self.length / 2.0, 0.0, 1.0)
        ub_tr = np.clip(x_center + self.length / 2.0, 0.0, 1.0)

        # Select points within the trust region
        inside_mask = np.all((X_scaled >= lb_tr) & (X_scaled <= ub_tr), axis=1)
        X_local = X_scaled[inside_mask]
        f_local = f_scaled[inside_mask]

        # If too few points in trust region, use all points
        if X_local.shape[0] < 2 * self.dim:
            X_local = X_scaled
            f_local = f_scaled

        # Retrain GP on potentially updated local points
        train_x = torch.tensor(X_local, dtype=torch.float32)
        train_y = torch.tensor((f_local - mu) / sigma, dtype=torch.float32)
        gp_model, gp_likelihood = train_gp(
            train_x=train_x,
            train_y=train_y,
            use_ard=self.use_ard,
            num_steps=self.n_training_steps,
            hypers=hypers,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            dtype=torch.float32
        )

        # Generate candidate points using Sobol sequence
        sobol = SobolEngine(self.dim, scramble=True)
        pert = sobol.draw(self.n_cand).numpy()
        X_cand = lb_tr + (ub_tr - lb_tr) * pert

        # Apply probability-based perturbation
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

        # Evaluate GP predictions on candidates
        gp_model.eval()
        gp_likelihood.eval()
        with torch.no_grad():
            X_torch = torch.tensor(X_cand, dtype=torch.float32)
            dist = gp_likelihood(gp_model(X_torch))
            f_samps = dist.sample(torch.Size([1])).numpy().reshape(-1)
            f_samps = mu + sigma * f_samps  # Transform back to original scale

        # Select the best candidates based on sampled function values
        idx_sorted = np.argsort(f_samps)
        best_inds = idx_sorted[:self.batch_size]
        X_cand_selected = X_cand[best_inds]

        return X_cand_selected

    def _restart_condition(self):
        """
        Determine if a restart is needed based on the current trust region size and evaluations.

        Returns:
        - bool: True if a restart is needed, False otherwise.
        """
        return self.length <= self.length_min and self.n_evals < self.max_evals

    def optimize(self):
        """
        Run the optimization process until the evaluation budget is exhausted.

        Returns:
        - best_x (np.ndarray): Best found input.
        - best_f (float): Best found function value.
        """
        # Initial optimization
        self.length = self.length_init
        self.initialize(restart=False)

        hypers = {}  # Initialize hyperparameters

        while self.n_evals < self.max_evals:
            while self.n_evals < self.max_evals and self.length > self.length_min:
                # Scale X to unit cube
                X_scaled = to_unit_cube(self.X, self.lb, self.ub)
                f_scaled = self.fX.ravel()

                # Generate next candidates
                X_next = self._create_candidates(X_scaled, f_scaled, hypers)

                # Evaluate new points
                fX_next = np.array([self.f(xnew) for xnew in X_next]).reshape(-1, 1)

                # Update data
                self.X = np.vstack([self.X, X_next])
                self.fX = np.vstack([self.fX, fX_next])
                self.n_evals += self.batch_size

                # Update best found point
                min_batch = np.min(fX_next)
                if min_batch < self.best_f:
                    self.best_f = float(min_batch)
                    self.best_x = X_next[np.argmin(fX_next)].copy()
                    if self.verbose:
                        print(f"New best f: {self.best_f:.4f}")

                # Update trust region
                self._update_trust_region(fX_next)

            # Check if restart is needed
            if self._restart_condition():
                if self.verbose:
                    print(f"Restarting TuRBO: {self.n_evals}/{self.max_evals} evaluations used.")
                # Restart by sampling new points across the entire search space
                self.initialize(restart=True)
                # Reset trust region length
                self.length = self.length_init
            else:
                break  # No more evaluations or cannot restart

        return self.best_x, self.best_f

# ================== Standalone PCA-BO (Simple) ==================

class SimplePCABO:
    """
    Standalone PCA-based Bayesian Optimization without trust regions.
    Utilizes weighted PCA for dimensionality reduction.
    """
    def __init__(self, f, lb, ub, hyperparams, verbose=False):
        """
        Initialize the SimplePCABO optimizer.

        Parameters:
        - f (callable): Objective function to minimize
        - lb (np.ndarray): Lower bounds of the search space
        - ub (np.ndarray): Upper bounds of the search space
        - hyperparams (dict): Hyperparameters from SHARED_PARAMS
        - verbose (bool): If True, prints progress messages
        """
        self.f = f
        self.lb = lb
        self.ub = ub
        self.dim = len(lb)

        # Unpack shared hyperparams
        self.n_init = hyperparams["n_init"]
        self.max_evals = hyperparams["max_evals_multiplier"] * self.dim
        self.batch_size = hyperparams["batch_size"]
        self.pca_components = hyperparams["pca_components"]
        self.use_ard = hyperparams["use_ard"]
        self.n_training_steps = hyperparams["n_training_steps"]

        # Data storage
        self.X = np.zeros((0, self.dim))
        self.fX = np.zeros((0, 1))
        self.best_x = None
        self.best_f = float('inf')
        self.n_evals = 0

        self.verbose = verbose

    def initialize(self):
        """
        Initialize the optimizer by sampling initial points.
        """
        X_init = np.random.uniform(self.lb, self.ub, size=(self.n_init, self.dim))
        f_init = np.array([self.f(x) for x in X_init]).reshape(-1, 1)

        self.X = np.vstack([self.X, X_init])
        self.fX = np.vstack([self.fX, f_init])
        self.n_evals += self.n_init

        idx_best = np.argmin(self.fX)
        self.best_f = float(self.fX[idx_best, 0])
        self.best_x = self.X[idx_best].copy()

        if self.verbose:
            print(f"Initializing with {self.n_init} points. Best f: {self.best_f:.4f}")

    def optimize(self):
        """
        Run the optimization process until the evaluation budget is exhausted.
        
        Returns:
        - best_x (np.ndarray): Best found input
        - best_f (float): Best found function value
        """
        self.initialize()
        while self.n_evals < self.max_evals:
            X_local = self.X
            y_local = self.fX.ravel()

            d_eff = min(self.pca_components, X_local.shape[1], X_local.shape[0])

            # Weighted PCA
            X_scaled = to_unit_cube(X_local, self.lb, self.ub)
            pca, X_reduced, X_mean = _weighted_pca(
                X_scaled,  # Scaled design
                y_local,   # Function values for weighting
                d_eff
            )

            # Train the GP in the PCA space
            train_x = torch.tensor(X_reduced, dtype=torch.float32)
            train_y = torch.tensor(y_local, dtype=torch.float32)
            gp_model, gp_likelihood = train_gp(
                train_x=train_x,
                train_y=train_y,
                use_ard=self.use_ard,
                num_steps=self.n_training_steps,
                device=train_x.device,
                dtype=train_x.dtype
            )

            # Generate candidate points in the reduced space using Sobol sequence
            n_candidates = min(100 * d_eff, 2500)
            sobol = SobolEngine(d_eff, scramble=True)
            Z = sobol.draw(n_candidates).numpy() * 2.0 - 1.0

            # Predict from the GP
            gp_model.eval()
            gp_likelihood.eval()
            with torch.no_grad():
                Z_torch = torch.tensor(Z, dtype=torch.float32)
                dist = gp_likelihood(gp_model(Z_torch))
                f_samps = dist.sample(torch.Size([1])).numpy().reshape(-1)

            # Select the best candidates based on sampled values
            idx_top = np.argsort(f_samps)[:self.batch_size]
            Z_next = Z[idx_top]

            # Inverse transform PCA and scale back to original bounds
            X_scaled_next = pca.inverse_transform(Z_next) + X_mean
            X_next = from_unit_cube(X_scaled_next, self.lb, self.ub)

            # Evaluate new points
            fX_next = np.array([self.f(xnew) for xnew in X_next]).reshape(-1, 1)

            # Update data
            self.X = np.vstack([self.X, X_next])
            self.fX = np.vstack([self.fX, fX_next])
            self.n_evals += self.batch_size

            # Track best
            min_batch = np.min(fX_next)
            if min_batch < self.best_f:
                self.best_f = float(min_batch)
                self.best_x = X_next[np.argmin(fX_next)].copy()
                if self.verbose:
                    print(f"New best f: {self.best_f:.4f}")

            if self.verbose:
                print(f"Evaluated {self.n_evals}/{self.max_evals} points.")

        return self.best_x, self.best_f

# ================== AxUS & BAXUS Implementations ==================


class AxUSProjector:
    """
    Random embedding akin to AxUS (used by BAxUS).
    """
    def __init__(self, input_dim, target_dim, seed=0):
        """
        Initialize the AxUSProjector.

        Parameters:
        - input_dim (int): Original dimensionality
        - target_dim (int): Target dimensionality for embedding
        - seed (int): Random seed for reproducibility
        """
        self.input_dim = input_dim
        self.target_dim = target_dim
        self.rng = np.random.RandomState(seed)

        # Simple random ±1 embedding
        self.A = self.rng.choice([1.0, -1.0], size=(self.target_dim, self.input_dim))

    def project_up(self, Z):
        """
        Project from latent space to original space.

        Parameters:
        - Z (np.ndarray): Latent points of shape (n_samples, target_dim)

        Returns:
        - X (np.ndarray): Projected points of shape (n_samples, input_dim)
        """
        return np.matmul(Z, self.A)

    def increase_target_dimensionality(self, dims_and_bins):
        """
        Increase the target dimensionality by splitting specified dimensions.

        Parameters:
        - dims_and_bins (dict): Mapping of dimension indices to number of splits
        """
        new_A_rows = []
        old_A = self.A
        keepers = set(range(self.target_dim)) - set(dims_and_bins.keys())

        # Keep old rows except split ones
        for i in sorted(list(keepers)):
            new_A_rows.append(old_A[i, :])

        # Replicate rows for split dimensions with random sign flips
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
    BAxUS-like optimizer: random embedding -> trust region in latent space -> dimension-splitting.
    Retrains GP only every TRAIN_INTERVAL steps.
    """
    def __init__(self, f, lb, ub, hyperparams, verbose=False):
        """
        Initialize the BAXUS optimizer.

        Parameters:
        - f (callable): Objective function to minimize
        - lb (np.ndarray): Lower bounds of the search space
        - ub (np.ndarray): Upper bounds of the search space
        - hyperparams (dict): Hyperparameters from SHARED_PARAMS
        - verbose (bool): If True, prints progress messages
        """
        self.f = f
        self.lb = lb
        self.ub = ub
        self.dim = len(lb)

        # Unpack shared hyperparams
        self.n_init = hyperparams["n_init"]
        self.max_evals = hyperparams["max_evals_multiplier"] * self.dim
        self.batch_size = hyperparams["batch_size"]
        self.lr = hyperparams["lr"]

        self.target_dim = hyperparams["baxus_init_target_dim"]
        self.success_tol = hyperparams["success_tol"]
        self.fail_tol_factor = hyperparams["baxus_fail_tol_factor"]

        self.length_init = hyperparams["length_init"]  # **Added this line**
        self.length = self.length_init
        self.length_min = hyperparams["length_min"]
        self.length_max = hyperparams["length_max"]

        self.use_ard = hyperparams["use_ard"]

        self.rng = np.random.RandomState(hyperparams["baxus_seed"])

        # Data storage
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
        self.GP_TRAIN_STEPS = hyperparams["n_training_steps"]

        self.verbose = verbose  # Ensure verbose flag is set

    def _fail_tol(self):
        """
        Calculate the failure tolerance based on target dimensionality.

        Returns:
        - float: Failure tolerance
        """
        return max(self.fail_tol_factor, self.target_dim)

    def _initialize(self):
        """
        Initialize the optimizer by projecting initial latent points and evaluating them.
        """
        Z_init = self.rng.uniform(-1.0, 1.0, size=(self.n_init, self.target_dim))
        X_up = self.projector.project_up(Z_init)

        # Correct and consistent mapping
        X_mapped = 0.5 * (X_up + 1.0)  # Scale from [-1, 1] to [0, 1]
        X_mapped = X_mapped * (self.ub - self.lb) + self.lb  # Scale to [lb, ub]

        fvals = np.array([self.f(x_) for x_ in X_mapped]).reshape(-1, 1)

        self.X = np.vstack([self.X, X_mapped])
        self.fX = np.vstack([self.fX, fvals])
        self.Z = np.vstack([self.Z, Z_init])
        self.n_evals += self.n_init

        idx_best = np.argmin(self.fX)
        if self.fX[idx_best, 0] < self.best_f:
            self.best_f = float(self.fX[idx_best, 0])
            self.best_x = self.X[idx_best].copy()

    def _update_trust_region(self, fX_batch):
        """
        Update the trust region based on the latest batch of function evaluations.

        Parameters:
        - fX_batch (np.ndarray): Latest function evaluations of shape (batch_size, 1)
        """
        if np.min(fX_batch) < self.best_f - 1e-3 * abs(self.best_f):
            self.success_count += 1
            self.fail_count = 0
        else:
            self.fail_count += 1
            self.success_count = 0

        if self.success_count >= self.success_tol:
            self.length = min(self.length * 2.0, self.length_max)
            if self.verbose:
                print(f"Increase trust region length to {self.length}")
            self.success_count = 0
        if self.fail_count >= self._fail_tol():
            self.length = max(self.length / 2.0, self.length_min)
            if self.verbose:
                print(f"Decrease trust region length to {self.length}")
            self.fail_count = 0

    def _choose_splitting_dims(self):
        """
        Choose which dimensions to split based on GP model's lengthscales.

        Returns:
        - dict: Mapping of dimension indices to number of splits
        """
        lengthscales = self.gp_model.covar_module.base_kernel.lengthscale.cpu().detach().numpy().flatten()
        importances = 1.0 / lengthscales
        sorted_idx = np.argsort(-importances)
        if self.target_dim >= self.dim:
            return {}
        best_dim = sorted_idx[0]
        dims_and_bins = {best_dim: 2}
        return dims_and_bins

    def _split_dimensions(self, dims_and_bins):
        """
        Split the specified dimensions and update the projector.

        Parameters:
        - dims_and_bins (dict): Mapping of dimension indices to number of splits
        """
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
        self.length = self.length_init

        if self.verbose:
            print(f"Split dimensions. New target dimensionality: {self.target_dim}")

        self._initialize()

    def _create_candidates_and_select(self):
        """
        Generate candidate latent points and select the best ones based on GP predictions.

        Returns:
        - Z_selected (np.ndarray): Selected latent points of shape (batch_size, target_dim)
        """

        i_best = np.argmin(self.fX)
        z_best = self.Z[i_best:i_best+1, :]

        lb_loc = np.clip(z_best - self.length / 2.0, -1.0, 1.0)
        ub_loc = np.clip(z_best + self.length / 2.0, -1.0, 1.0)

        n_cand = min(100 * self.target_dim, 2500)
        sobol = SobolEngine(self.target_dim, scramble=True, seed=self.rng.randint(1e6))
        pert = sobol.draw(n_cand).numpy()
        Z_cand = lb_loc + (ub_loc - lb_loc) * pert

        # Apply probability-based perturbation
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

        # Predict using GP
        self.gp_model.eval()
        self.gp_likelihood.eval()
        with torch.no_grad():
            Z_torch = torch.tensor(Z_cand, dtype=torch.float32)
            dist = self.gp_likelihood(self.gp_model(Z_torch))
            f_samps = dist.sample(torch.Size([1])).numpy().reshape(-1)

        # Select the best candidates based on sampled values
        idx_sorted = np.argsort(f_samps)
        best_inds = idx_sorted[:self.batch_size]
        return Z_cand[best_inds]

    def optimize(self):
        """
        Run the optimization process until the evaluation budget is exhausted.
        
        Returns:
        - best_x (np.ndarray): Best found input
        - best_f (float): Best found function value
        """
        self._initialize()

        while self.n_evals < self.max_evals:
            # Retrain GP every TRAIN_INTERVAL steps or if GP not trained yet
            if (self.n_evals == self.n_init) or (self.n_evals % self.TRAIN_INTERVAL == 0) or (self.gp_model is None):
                train_x = torch.tensor(self.Z, dtype=torch.float32)
                train_y = torch.tensor(self.fX.ravel(), dtype=torch.float32)
                self.gp_model, self.gp_likelihood = train_gp(
                    train_x=train_x,
                    train_y=train_y,
                    use_ard=self.use_ard,
                    num_steps=self.GP_TRAIN_STEPS,
                    device=train_x.device,
                    dtype=train_x.dtype
                )
                if self.verbose:
                    print(f"GP retrained at evaluation {self.n_evals}")

                # Possibly split dimension
                if self.target_dim < self.dim and self.success_count == 0 and self.fail_count == 0:
                    dims_and_bins = self._choose_splitting_dims()
                    if dims_and_bins:
                        self._split_dimensions(dims_and_bins)
                        # Re-train after splitting
                        train_x = torch.tensor(self.Z, dtype=torch.float32)
                        train_y = torch.tensor(self.fX.ravel(), dtype=torch.float32)
                        self.gp_model, self.gp_likelihood = train_gp(
                            train_x=train_x,
                            train_y=train_y,
                            use_ard=self.use_ard,
                            num_steps=self.GP_TRAIN_STEPS,
                            device=train_x.device,
                            dtype=train_x.dtype
                        )
                        if self.verbose:
                            print(f"GP retrained after dimension splitting at evaluation {self.n_evals}")

            # Generate and select candidates
            Z_next = self._create_candidates_and_select()
            X_up = self.projector.project_up(Z_next)

            # Corrected mapping from latent to original space
            X_mapped = 0.5 * (X_up + 1.0)  # Scale from [-1, 1] to [0, 1]
            X_mapped = X_mapped * (self.ub - self.lb) + self.lb  # Scale to [lb, ub]

            # Evaluate new points
            fX_next = np.array([self.f(x_) for x_ in X_mapped]).reshape(-1, 1)

            # Update data
            self.Z = np.vstack([self.Z, Z_next])
            self.X = np.vstack([self.X, X_mapped])
            self.fX = np.vstack([self.fX, fX_next])
            self.n_evals += self.batch_size

            # Track best
            min_batch = np.min(fX_next)
            if min_batch < self.best_f:
                self.best_f = float(min_batch)
                self.best_x = X_mapped[np.argmin(fX_next)].copy()
                if self.verbose:
                    print(f"New best f: {self.best_f} at evaluation {self.n_evals}")

            # Update trust region
            self._update_trust_region(fX_next)

            # Safeguard: Ensure trust region length does not fall below minimum
            if self.length < self.length_min:
                self.length = self.length_min
                if self.verbose:
                    print(f"Trust region length reached minimum: {self.length_min}")

        if self.verbose:
            print(f"Optimization completed. Best f: {self.best_f} at evaluation {self.n_evals}")

        return self.best_x, self.best_f

# ============ Benchmark Loop ============

def run_one_experiment(method_name, fid, instance, dimension, n_reps=5, budget_multiplier=10, verbose=False):
    """
    Runs one method on a single (fid, instance, dimension) problem for n_reps times.
    Returns average runtime in seconds over the n_reps.

    Parameters:
    - method_name (str): Name of the optimization method
    - fid (int): Function ID for the BBOB problem
    - instance (int): Instance number for the BBOB problem
    - dimension (int): Dimensionality of the problem
    - n_reps (int): Number of repetitions
    - budget_multiplier (int): Multiplier to determine evaluation budget
    - verbose (bool): If True, prints progress messages

    Returns:
    - float: Average runtime in seconds over the n_reps
    """
    folder_name = f"data-{method_name}-f{fid}-dim{dimension}"
    times = []
    for rep in range(1, n_reps + 1):
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
        local_hp["max_evals"] = max_evals  # Override

        # Start timing
        t0 = time.time()

        if method_name == "turbo_pca":
            optimizer = TuRBO_PCA_BO(
                f=problem,
                lb=np.array(problem.bounds.lb),
                ub=np.array(problem.bounds.ub),
                hyperparams=local_hp,
                verbose=verbose
            )
            optimizer.optimize()

        elif method_name == "turbo_only":
            optimizer = TurboOnly(
                f=problem,
                lb=np.array(problem.bounds.lb),
                ub=np.array(problem.bounds.ub),
                hyperparams=local_hp,
                verbose=verbose
            )
            optimizer.optimize()

        elif method_name == "pca_only":
            optimizer = SimplePCABO(
                f=problem,
                lb=np.array(problem.bounds.lb),
                ub=np.array(problem.bounds.ub),
                hyperparams=local_hp,
                verbose=verbose
            )
            optimizer.optimize()

        elif method_name == "baxus":
            optimizer = BAXUS(
                f=problem,
                lb=np.array(problem.bounds.lb),
                ub=np.array(problem.bounds.ub),
                hyperparams=local_hp,
                verbose=verbose
            )
            optimizer.optimize()
        else:
            raise ValueError(f"Unknown method: {method_name}")

        # Stop timing
        t1 = time.time()
        times.append(t1 - t0)

        problem.reset()

    return np.mean(times)

def main():
    """
    Main function to run benchmarking experiments across different methods, functions, instances, and dimensions.
    Saves runtime results and generates a runtime comparison plot.
    """
    # Example BBOB test set
    fids = [21]  # List of function IDs
    instances = [0, 1, 2]  # List of instances
    dimensions = [10]  # List of dimensions
    methods = ["turbo_pca", "turbo_only", "pca_only", "baxus"]
    n_reps = 5
    budget_multiplier = SHARED_PARAMS["max_evals_multiplier"]

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
                        budget_multiplier=budget_multiplier,
                        verbose=False
                    )
                    runtime_results[(m, fid, inst, dim)] = avg_time

                    elapsed = time.time() - t0_global
                    print(f"  -> Avg time over {n_reps} runs: {avg_time:.2f}s. Total elapsed: {elapsed:.2f}s.")

    print("\nAll experiments completed.")
    print("Logs are in data-<method>-f<fID>-dim<D>/ folders.\n")

    # ================== Plot the Runtime Differences ==================
    # For illustration, average across all fids, instances, dims for each method.
    method_avg_times = {}
    for m in methods:
        subset = [runtime_results[k] for k in runtime_results if k[0] == m]
        method_avg_times[m] = np.mean(subset) if len(subset) > 0 else 0.0

    plt.figure(figsize=(8, 6))
    bars = plt.bar(method_avg_times.keys(), method_avg_times.values(), color=['blue', 'orange', 'green', 'red'])
    plt.ylabel("Average Runtime (s)")
    plt.title("Runtime Comparison Across Methods (All Problems)")
    plt.tight_layout()

    # Annotate bars with their heights
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.2f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),  
                     textcoords="offset points",
                     ha='center', va='bottom')

    plt.savefig("runtime_comparison.png")
    plt.show()

if __name__ == "__main__":
    main()
