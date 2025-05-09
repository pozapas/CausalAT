"""
Balancing methods for covariate distribution matching in the latent space.
"""

import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import warnings

class Balancer(BaseEstimator, TransformerMixin):
    """
    Base class for balancing methods (Ψ component in the three-layer architecture).
    
    This component produces weights to equalize treated and control covariate distributions.
    """
    
    def __init__(self):
        self._is_fitted = False
    
    def fit(self, X, D):
        """
        Fit the balancer to the data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features (usually latent representations Z from Φ).
        D : array-like of shape (n_samples,)
            Treatment indicator (0 or 1).
            
        Returns
        -------
        self : object
            Returns self.
        """
        # Should be implemented by subclasses
        self._is_fitted = True
        return self
    
    def transform(self, X, D):
        """
        Compute balancing weights for the data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features (usually latent representations Z from Φ).
        D : array-like of shape (n_samples,)
            Treatment indicator (0 or 1).
            
        Returns
        -------
        W : array-like of shape (n_samples,)
            Balancing weights.
        """
        # Should be implemented by subclasses
        if not self._is_fitted:
            raise ValueError("Balancer not fitted. Call fit() first.")
        
        # Default implementation: return uniform weights
        return np.ones(X.shape[0])
    
    def fit_transform(self, X, D):
        """
        Fit to data and compute balancing weights.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features (usually latent representations Z from Φ).
        D : array-like of shape (n_samples,)
            Treatment indicator (0 or 1).
            
        Returns
        -------
        W : array-like of shape (n_samples,)
            Balancing weights.
        """
        return self.fit(X, D).transform(X, D)
    
    def measure_imbalance(self, X, D, W=None):
        """
        Measure the covariate imbalance between treated and control groups.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features (usually latent representations Z from Φ).
        D : array-like of shape (n_samples,)
            Treatment indicator (0 or 1).
        W : array-like of shape (n_samples,), optional
            Balancing weights. If None, use uniform weights.
            
        Returns
        -------
        imbalance : float
            A measure of covariate imbalance (lower is better).
        """
        # Default implementation: compute standardized mean differences
        if W is None:
            W = np.ones(X.shape[0])
        
        # Convert to numpy arrays if they're not already
        X = np.asarray(X)
        D = np.asarray(D)
        W = np.asarray(W)
        
        # Indices for treated and control units
        treated = (D == 1)
        control = (D == 0)
        
        # Normalize weights within each group
        W_t = W[treated] / (W[treated].sum() + 1e-8)
        W_c = W[control] / (W[control].sum() + 1e-8)
        
        # Compute weighted means
        X_t_mean = np.sum(X[treated] * W_t[:, np.newaxis], axis=0)
        X_c_mean = np.sum(X[control] * W_c[:, np.newaxis], axis=0)
        
        # Compute weighted variances
        X_t_var = np.sum(W_t[:, np.newaxis] * (X[treated] - X_t_mean)**2, axis=0)
        X_c_var = np.sum(W_c[:, np.newaxis] * (X[control] - X_c_mean)**2, axis=0)
        
        # Compute standardized mean difference
        pooled_std = np.sqrt((X_t_var + X_c_var) / 2)
        std_mean_diff = np.abs(X_t_mean - X_c_mean) / (pooled_std + 1e-8)
        
        # Return average SMD across all features
        return np.mean(std_mean_diff)


class IPWBalancer(Balancer):
    """
    Inverse Probability Weighting (IPW) balancer.
    
    Parameters
    ----------
    propensity_model : BaseEstimator
        Model to estimate propensity scores P(D=1|X).
    stabilize : bool, default=True
        Whether to stabilize weights by dividing by their mean.
    clip : float, default=10.0
        Maximum allowed weight value.
    """
    
    def __init__(
        self, 
        propensity_model: BaseEstimator,
        stabilize: bool = True,
        clip: float = 10.0
    ):
        super().__init__()
        self.propensity_model = propensity_model
        self.stabilize = stabilize
        self.clip = clip
    
    def fit(self, X, D):
        """
        Fit the propensity model to estimate P(D=1|X).
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features (usually latent representations Z from Φ).
        D : array-like of shape (n_samples,)
            Treatment indicator (0 or 1).
            
        Returns
        -------
        self : object
            Returns self.
        """
        # Convert inputs to numpy arrays if they're not already
        X = np.asarray(X)
        D = np.asarray(D)
        
        # Fit propensity model
        self.propensity_model.fit(X, D)
        
        self._is_fitted = True
        return self
    
    def transform(self, X, D):
        """
        Compute IPW weights.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features (usually latent representations Z from Φ).
        D : array-like of shape (n_samples,)
            Treatment indicator (0 or 1).
            
        Returns
        -------
        W : array-like of shape (n_samples,)
            IPW weights.
        """
        if not self._is_fitted:
            raise ValueError("Balancer not fitted. Call fit() first.")
        
        # Convert inputs to numpy arrays if they're not already
        X = np.asarray(X)
        D = np.asarray(D)
        
        # Predict propensity scores
        e_X = self.propensity_model.predict_proba(X)[:, 1]
        
        # Compute IPW weights
        W = np.where(D == 1, 1.0 / np.maximum(e_X, 1e-8), 1.0 / np.maximum(1 - e_X, 1e-8))
        
        # Stabilize weights if requested
        if self.stabilize:
            W_treated = W[D == 1]
            W_control = W[D == 0]
            
            if len(W_treated) > 0:
                W[D == 1] = W_treated / np.mean(W_treated)
            
            if len(W_control) > 0:
                W[D == 0] = W_control / np.mean(W_control)
        
        # Clip weights if requested
        if self.clip is not None:
            W = np.clip(W, 0, self.clip)
        
        return W


class KernelBalancer(Balancer):
    """
    Kernel Balancing method.
    
    Parameters
    ----------
    kernel : str, default='rbf'
        Kernel type. Options: 'rbf', 'polynomial', 'linear'.
    kernel_params : dict, optional
        Parameters for the kernel.
    lambda_reg : float, default=0.01
        Regularization parameter.
    """
    
    def __init__(
        self,
        kernel: str = 'rbf',
        kernel_params: Optional[Dict] = None,
        lambda_reg: float = 0.01
    ):
        super().__init__()
        self.kernel = kernel
        self.kernel_params = kernel_params or {}
        self.lambda_reg = lambda_reg
        self.scaler = StandardScaler()
    
    def _compute_kernel(self, X, Y=None):
        """
        Compute the kernel matrix.
        
        Parameters
        ----------
        X : array-like of shape (n_samples_X, n_features)
            First set of samples.
        Y : array-like of shape (n_samples_Y, n_features), optional
            Second set of samples. If None, use X.
            
        Returns
        -------
        K : array-like of shape (n_samples_X, n_samples_Y or n_samples_X)
            Kernel matrix.
        """
        if Y is None:
            Y = X
        
        if self.kernel == 'linear':
            return X @ Y.T
        elif self.kernel == 'polynomial':
            degree = self.kernel_params.get('degree', 3)
            coef0 = self.kernel_params.get('coef0', 1)
            return (X @ Y.T + coef0) ** degree
        elif self.kernel == 'rbf':
            gamma = self.kernel_params.get('gamma', 1.0 / X.shape[1])
            
            # Compute pairwise squared distances
            X_norm = np.sum(X**2, axis=1).reshape(-1, 1)
            Y_norm = np.sum(Y**2, axis=1).reshape(1, -1)
            dist_sq = X_norm + Y_norm - 2 * (X @ Y.T)
            
            return np.exp(-gamma * dist_sq)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def fit(self, X, D):
        """
        Fit the kernel balancer.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features (usually latent representations Z from Φ).
        D : array-like of shape (n_samples,)
            Treatment indicator (0 or 1).
            
        Returns
        -------
        self : object
            Returns self.
        """
        # Convert inputs to numpy arrays if they're not already
        X = np.asarray(X)
        D = np.asarray(D)
        
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        # Separate treated and control units
        X_t = X_scaled[D == 1]
        X_c = X_scaled[D == 0]
        
        n_t = X_t.shape[0]
        n_c = X_c.shape[0]
        
        if n_t == 0 or n_c == 0:
            raise ValueError("Both treated and control groups must have at least one unit.")
        
        # Compute kernel matrices
        K_t = self._compute_kernel(X_t)
        K_c = self._compute_kernel(X_c)
        K_tc = self._compute_kernel(X_t, X_c)
        
        # Compute target moments (mean of treated units in feature space)
        target_moments = np.mean(K_t, axis=0)
        
        # Solve for weights
        # In a real implementation, we'd solve:
        # w_c = argmin_w ||K_c @ w - target_moments||^2 + lambda * ||w||^2
        # s.t. w >= 0, sum(w) = 1
        
        # For simplicity, we'll just compute ridge regression weights here
        A = K_c + self.lambda_reg * np.eye(n_c)
        b = K_tc.T @ np.ones(n_t) / n_t
        
        # Solve the linear system
        self.control_weights = np.linalg.solve(A, b)
        
        # Normalize the weights to sum to 1
        self.control_weights = np.maximum(self.control_weights, 0)  # Ensure non-negativity
        self.control_weights /= np.sum(self.control_weights)
        
        self._is_fitted = True
        return self
    
    def transform(self, X, D):
        """
        Compute balancing weights.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features (usually latent representations Z from Φ).
        D : array-like of shape (n_samples,)
            Treatment indicator (0 or 1).
            
        Returns
        -------
        W : array-like of shape (n_samples,)
            Balancing weights.
        """
        if not self._is_fitted:
            raise ValueError("Balancer not fitted. Call fit() first.")
        
        # Convert inputs to numpy arrays if they're not already
        X = np.asarray(X)
        D = np.asarray(D)
        
        # Initialize weights
        W = np.ones(X.shape[0])
        
        # Apply control weights
        W[D == 0] = self.control_weights
        
        return W


class KernelMMD(Balancer):
    """
    Balancer that minimizes the Maximum Mean Discrepancy (MMD) between treated and control groups.
    
    Parameters
    ----------
    learning_rate : float, default=0.01
        Learning rate for gradient descent.
    n_iterations : int, default=1000
        Number of optimization iterations.
    kernel : str, default='rbf'
        Kernel type. Options: 'rbf', 'polynomial', 'linear'.
    kernel_params : dict, optional
        Parameters for the kernel.
    lambda_reg : float, default=0.01
        Regularization parameter.
    batch_size : int, default=None
        Batch size for stochastic optimization. If None, use all data.
    device : str, default='cuda' if available else 'cpu'
        Device to use for computation.
    """
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        kernel: str = 'rbf',
        kernel_params: Optional[Dict] = None,
        lambda_reg: float = 0.01,
        batch_size: Optional[int] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.kernel = kernel
        self.kernel_params = kernel_params or {}
        self.lambda_reg = lambda_reg
        self.batch_size = batch_size
        self.device = device
        self.scaler = StandardScaler()
    
    def _compute_kernel(self, X, Y=None):
        """
        Compute the kernel matrix using PyTorch.
        
        Parameters
        ----------
        X : torch.Tensor of shape (n_samples_X, n_features)
            First set of samples.
        Y : torch.Tensor of shape (n_samples_Y, n_features), optional
            Second set of samples. If None, use X.
            
        Returns
        -------
        K : torch.Tensor of shape (n_samples_X, n_samples_Y or n_samples_X)
            Kernel matrix.
        """
        if Y is None:
            Y = X
        
        if self.kernel == 'linear':
            return torch.matmul(X, Y.t())
        elif self.kernel == 'polynomial':
            degree = self.kernel_params.get('degree', 3)
            coef0 = self.kernel_params.get('coef0', 1)
            return (torch.matmul(X, Y.t()) + coef0) ** degree
        elif self.kernel == 'rbf':
            gamma = self.kernel_params.get('gamma', 1.0 / X.shape[1])
            
            # Compute pairwise squared distances
            X_norm = torch.sum(X**2, dim=1).view(-1, 1)
            Y_norm = torch.sum(Y**2, dim=1).view(1, -1)
            dist_sq = X_norm + Y_norm - 2 * torch.matmul(X, Y.t())
            
            return torch.exp(-gamma * dist_sq)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def _compute_mmd(self, X_t, X_c, W_t=None, W_c=None):
        """
        Compute the Maximum Mean Discrepancy (MMD) between treated and control groups.
        
        Parameters
        ----------
        X_t : torch.Tensor of shape (n_treated, n_features)
            Features of treated units.
        X_c : torch.Tensor of shape (n_control, n_features)
            Features of control units.
        W_t : torch.Tensor of shape (n_treated,), optional
            Weights for treated units. If None, use uniform weights.
        W_c : torch.Tensor of shape (n_control,), optional
            Weights for control units. If None, use uniform weights.
            
        Returns
        -------
        mmd : torch.Tensor
            The MMD value.
        """
        n_t = X_t.shape[0]
        n_c = X_c.shape[0]
        
        # Default to uniform weights if not provided
        if W_t is None:
            W_t = torch.ones(n_t, device=self.device) / n_t
        if W_c is None:
            W_c = torch.ones(n_c, device=self.device) / n_c
        
        # Normalize weights to sum to 1
        W_t = W_t / torch.sum(W_t)
        W_c = W_c / torch.sum(W_c)
        
        # Compute kernel matrices
        K_tt = self._compute_kernel(X_t)
        K_cc = self._compute_kernel(X_c)
        K_tc = self._compute_kernel(X_t, X_c)
        
        # Compute MMD^2 using kernel matrices and weights
        W_t_outer = torch.outer(W_t, W_t)
        W_c_outer = torch.outer(W_c, W_c)
        W_tc_outer = torch.outer(W_t, W_c)
        
        mmd_squared = (
            torch.sum(W_t_outer * K_tt) +
            torch.sum(W_c_outer * K_cc) -
            2 * torch.sum(W_tc_outer * K_tc)
        )
        
        return torch.sqrt(torch.clamp(mmd_squared, min=1e-8))
    
    def fit(self, X, D):
        """
        Fit the MMD balancer.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features (usually latent representations Z from Φ).
        D : array-like of shape (n_samples,)
            Treatment indicator (0 or 1).
            
        Returns
        -------
        self : object
            Returns self.
        """
        # Convert inputs to numpy arrays if they're not already
        X = np.asarray(X)
        D = np.asarray(D)
        
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        # Separate treated and control units
        X_t = X_scaled[D == 1]
        X_c = X_scaled[D == 0]
        
        n_t = X_t.shape[0]
        n_c = X_c.shape[0]
        
        if n_t == 0 or n_c == 0:
            raise ValueError("Both treated and control groups must have at least one unit.")
        
        # Convert to PyTorch tensors and move to device
        X_t_torch = torch.tensor(X_t, dtype=torch.float32, device=self.device)
        X_c_torch = torch.tensor(X_c, dtype=torch.float32, device=self.device)
        
        # Initialize weights for control units (we'll optimize these)
        log_W_c = torch.zeros(n_c, device=self.device, requires_grad=True)
        
        # Setup optimizer
        optimizer = torch.optim.Adam([log_W_c], lr=self.learning_rate)
        
        # Optimization loop
        for i in range(self.n_iterations):
            # Get batch indices if using mini-batch optimization
            if self.batch_size is not None and self.batch_size < min(n_t, n_c):
                t_indices = np.random.choice(n_t, self.batch_size, replace=False)
                c_indices = np.random.choice(n_c, self.batch_size, replace=False)
                X_t_batch = X_t_torch[t_indices]
                X_c_batch = X_c_torch[c_indices]
                log_W_c_batch = log_W_c[c_indices]
            else:
                X_t_batch = X_t_torch
                X_c_batch = X_c_torch
                log_W_c_batch = log_W_c
            
            # Convert log weights to weights using softmax
            W_c_batch = F.softmax(log_W_c_batch, dim=0)
            
            # Compute MMD
            mmd = self._compute_mmd(X_t_batch, X_c_batch, None, W_c_batch)
            
            # Add regularization
            reg = self.lambda_reg * torch.sum(W_c_batch**2)
            
            # Compute loss
            loss = mmd + reg
            
            # Backward pass and optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if i % 100 == 0:
                print(f"Iteration {i}, MMD: {mmd.item():.6f}, Reg: {reg.item():.6f}")
        
        # Get final weights
        self.control_weights = F.softmax(log_W_c, dim=0).cpu().detach().numpy()
        
        self._is_fitted = True
        return self
    
    def transform(self, X, D):
        """
        Compute balancing weights.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features (usually latent representations Z from Φ).
        D : array-like of shape (n_samples,)
            Treatment indicator (0 or 1).
            
        Returns
        -------
        W : array-like of shape (n_samples,)
            Balancing weights.
        """
        if not self._is_fitted:
            raise ValueError("Balancer not fitted. Call fit() first.")
        
        # Convert inputs to numpy arrays if they're not already
        X = np.asarray(X)
        D = np.asarray(D)
        
        # Initialize weights
        W = np.ones(X.shape[0])
        
        # Apply control weights
        control_indices = np.where(D == 0)[0]
        
        if len(control_indices) > 0:
            # If the number of control units has changed, we need to adapt
            if len(control_indices) != len(self.control_weights):
                warnings.warn(
                    f"Number of control units changed from {len(self.control_weights)} to {len(control_indices)}. "
                    "Using uniform weights for control units."
                )
                W[D == 0] = 1.0
            else:
                W[D == 0] = self.control_weights
        
        return W
