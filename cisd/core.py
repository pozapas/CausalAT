"""
Core implementation of the Causal-Intervention Scenario Design (CISD) framework.
"""

import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union
from sklearn.base import BaseEstimator
import warnings

class CISD(BaseEstimator):
    """
    Causal-Intervention Scenario Design framework for estimating effects under paired interventions.
    
    This class implements the CISD framework described in Section 4 of the paper, allowing
    estimation of treatment effects conditional on specified scenarios.
    
    Parameters
    ----------
    scenario_selector : Callable
        Function that maps units to their scenarios of interest.
        Should accept features X and return scenario vectors.
    propensity_model : BaseEstimator
        Model to estimate propensity scores P(D=1|X).
    outcome_model : Union[BaseEstimator, Dict[str, BaseEstimator]]
        Model(s) to estimate outcome given treatment, features and mediators.
        Can be a single model or a dict with keys '0', '1' for separate models by treatment.
    mediator_model : Optional[BaseEstimator]
        Model to estimate mediator distributions.
    random_state : Optional[int]
        Random seed for reproducibility.
    """
    
    def __init__(
        self,
        scenario_selector: Callable,
        propensity_model: BaseEstimator,
        outcome_model: Union[BaseEstimator, Dict[str, BaseEstimator]],
        mediator_model: Optional[BaseEstimator] = None,
        random_state: Optional[int] = None
    ):
        self.scenario_selector = scenario_selector
        self.propensity_model = propensity_model
        self.outcome_model = outcome_model
        self.mediator_model = mediator_model
        self.random_state = random_state
        self._is_fitted = False
    
    def fit(self, X: np.ndarray, D: np.ndarray, Y: np.ndarray, M: Optional[np.ndarray] = None):
        """
        Fit the CISD components.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features/covariates.
        D : array-like of shape (n_samples,)
            Treatment indicator (0 or 1).
        Y : array-like of shape (n_samples,)
            Outcome variable.
        M : array-like of shape (n_samples, n_mediators), optional
            Mediator variables.
            
        Returns
        -------
        self : object
            Returns self.
        """
        # Set random seed
        np.random.seed(self.random_state)
        
        # Convert inputs to numpy arrays if they're not already
        X = np.asarray(X)
        D = np.asarray(D)
        Y = np.asarray(Y)
        
        # Fit propensity score model
        self.propensity_model.fit(X, D)
        
        # Fit outcome model(s)
        if isinstance(self.outcome_model, dict):
            # Separate models for each treatment level
            X_d0 = X[D == 0]
            Y_d0 = Y[D == 0]
            X_d1 = X[D == 1]
            Y_d1 = Y[D == 1]
            
            if M is not None:
                M_d0 = M[D == 0]
                X_M_d0 = np.column_stack([X_d0, M_d0])
                M_d1 = M[D == 1]
                X_M_d1 = np.column_stack([X_d1, M_d1])
                self.outcome_model['0'].fit(X_M_d0, Y_d0)
                self.outcome_model['1'].fit(X_M_d1, Y_d1)
            else:
                self.outcome_model['0'].fit(X_d0, Y_d0)
                self.outcome_model['1'].fit(X_d1, Y_d1)
        else:
            # Single model with treatment as a feature
            if M is not None:
                X_M_D = np.column_stack([X, D.reshape(-1, 1), M])
                self.outcome_model.fit(X_M_D, Y)
            else:
                X_D = np.column_stack([X, D.reshape(-1, 1)])
                self.outcome_model.fit(X_D, Y)
        
        # Fit mediator model if provided
        if M is not None and self.mediator_model is not None:
            self.mediator_model.fit(np.column_stack([X, D.reshape(-1, 1)]), M)
        
        self._is_fitted = True
        return self
    
    def estimate(
        self, 
        X: np.ndarray, 
        D: Optional[np.ndarray] = None,
        Y: Optional[np.ndarray] = None,
        M: Optional[np.ndarray] = None,
        n_bootstrap: int = 100,
        alpha: float = 0.05
    ) -> Dict:
        """
        Estimate the CISD causal effect.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features/covariates.
        D : array-like of shape (n_samples,), optional
            Treatment indicator (0 or 1). Required if using observed data in estimation.
        Y : array-like of shape (n_samples,), optional
            Outcome variable. Required if using observed data in estimation.
        M : array-like of shape (n_samples, n_mediators), optional
            Mediator variables.
        n_bootstrap : int, default=100
            Number of bootstrap iterations for confidence intervals.
        alpha : float, default=0.05
            Significance level for confidence intervals (1-alpha)% CI.
            
        Returns
        -------
        result : dict
            Dictionary containing the estimated effect and confidence interval.
        """
        if not self._is_fitted:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        # Convert inputs to numpy arrays
        X = np.asarray(X)
        
        # Generate scenarios for each unit
        S = self.scenario_selector(X)
        
        # Generate samples for estimating the effect
        estimates = []
        
        # Main estimate
        est = self._compute_estimate(X, D, Y, M, S)
        estimates.append(est)
        
        # Bootstrap for confidence intervals
        if n_bootstrap > 0:
            bootstrap_estimates = []
            n_samples = X.shape[0]
            for _ in range(n_bootstrap):
                # Sample with replacement
                indices = np.random.choice(n_samples, n_samples, replace=True)
                X_boot = X[indices]
                D_boot = None if D is None else D[indices]
                Y_boot = None if Y is None else Y[indices]
                M_boot = None if M is None else M[indices]
                S_boot = S[indices] if S.ndim > 0 else S
                
                boot_est = self._compute_estimate(X_boot, D_boot, Y_boot, M_boot, S_boot)
                bootstrap_estimates.append(boot_est)
            
            # Calculate confidence intervals
            lower_idx = int(n_bootstrap * alpha / 2)
            upper_idx = int(n_bootstrap * (1 - alpha / 2))
            sorted_estimates = np.sort(bootstrap_estimates)
            lower_bound = sorted_estimates[lower_idx]
            upper_bound = sorted_estimates[upper_idx]
        else:
            lower_bound = np.nan
            upper_bound = np.nan
        
        return {
            'estimate': est,
            'conf_int_lower': lower_bound,
            'conf_int_upper': upper_bound,
            'alpha': alpha
        }
    
    def _compute_estimate(
        self, 
        X: np.ndarray, 
        D: Optional[np.ndarray] = None,
        Y: Optional[np.ndarray] = None,
        M: Optional[np.ndarray] = None,
        S: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute the CISD estimate using the efficient influence function approach.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features/covariates.
        D : array-like of shape (n_samples,), optional
            Treatment indicator (0 or 1).
        Y : array-like of shape (n_samples,), optional
            Outcome variable.
        M : array-like of shape (n_samples, n_mediators), optional
            Mediator variables.
        S : array-like, optional
            Scenario vectors for each unit.
            
        Returns
        -------
        estimate : float
            The CISD estimate.
        """
        # Predict propensity scores
        e_X = self.propensity_model.predict_proba(X)[:, 1]
        
        # Initialize influence functions
        n_samples = X.shape[0]
        influence_values = np.zeros(n_samples)
        
        # If we have observed data, compute the first part of the influence function
        if D is not None and Y is not None:
            # For treated units (D=1)
            treated = D == 1
            if np.any(treated):
                # Calculate weights for treated
                w1_X = self._compute_weights(X[treated], 1, S[treated] if S is not None else None)
                
                # Calculate the outcome regression term for treated
                mu1_X = self._predict_outcome(X[treated], 1, M[treated] if M is not None else None)
                
                # Compute influence contribution for treated
                influence_values[treated] = w1_X * (Y[treated] - mu1_X) / e_X[treated]
            
            # For control units (D=0)
            control = D == 0
            if np.any(control):
                # Calculate weights for control
                w0_X = self._compute_weights(X[control], 0, S[control] if S is not None else None)
                
                # Calculate the outcome regression term for control
                mu0_X = self._predict_outcome(X[control], 0, M[control] if M is not None else None)
                
                # Compute influence contribution for control
                influence_values[control] = -w0_X * (Y[control] - mu0_X) / (1 - e_X[control])
        
        # Compute the second part of the influence function (scenario-based term)
        for i in range(n_samples):
            # Get scenario for unit i
            scenario_i = S[i] if S is not None else None
            
            # Predict outcome under treatment with scenario
            mu1_X_S = self._predict_outcome(X[i:i+1], 1, scenario_i)
            
            # Predict outcome under control with scenario
            mu0_X_S = self._predict_outcome(X[i:i+1], 0, scenario_i)
            
            # Add to influence function
            influence_values[i] += (mu1_X_S - mu0_X_S)
        
        # Compute the CISD estimate as the mean of influence values
        return np.mean(influence_values)
    
    def _compute_weights(
        self, 
        X: np.ndarray, 
        d: int, 
        S: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute weights for the influence function.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features/covariates.
        d : int
            Treatment level (0 or 1).
        S : array-like, optional
            Scenario vectors.
            
        Returns
        -------
        weights : array-like
            Weights for each sample.
        """
        # Default weights are 1 (for fixed scenario)
        if S is None or self.mediator_model is None:
            return np.ones(X.shape[0])
        
        # For stochastic scenarios, compute ratio of scenario density to factual density
        # This is a simplified implementation - in practice, we'd need proper density estimation
        # p_s(M|X) / p(M|D=d,X)
        
        # For now, we return ones as placeholder
        warnings.warn("Stochastic scenario weights not fully implemented, using uniform weights.")
        return np.ones(X.shape[0])
    
    def _predict_outcome(
        self, 
        X: np.ndarray, 
        d: int, 
        M: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Predict outcomes under specific treatment and mediator values.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features/covariates.
        d : int
            Treatment level (0 or 1).
        M : array-like, optional
            Mediator values.
            
        Returns
        -------
        predictions : array-like
            Predicted outcomes.
        """
        # If we have separate models for each treatment level
        if isinstance(self.outcome_model, dict):
            model = self.outcome_model[str(d)]
            if M is not None:
                X_M = np.column_stack([X, M])
                return model.predict(X_M)
            else:
                return model.predict(X)
        else:
            # Single model with treatment as a feature
            X_d = np.column_stack([X, np.full(X.shape[0], d).reshape(-1, 1)])
            if M is not None:
                X_d_M = np.column_stack([X_d, M])
                return self.outcome_model.predict(X_d_M)
            else:
                return self.outcome_model.predict(X_d)
    
    def incremental_scenario_effect(
        self, 
        X: np.ndarray, 
        S_base: np.ndarray,
        S_new: np.ndarray,
        n_bootstrap: int = 100,
        alpha: float = 0.05
    ) -> Dict:
        """
        Estimate the incremental scenario effect as defined in Eq. (29).
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features/covariates.
        S_base : array-like
            Base scenario vectors.
        S_new : array-like
            New scenario vectors.
        n_bootstrap : int, default=100
            Number of bootstrap iterations for confidence intervals.
        alpha : float, default=0.05
            Significance level for confidence intervals (1-alpha)% CI.
            
        Returns
        -------
        result : dict
            Dictionary containing the estimated effect and confidence interval.
        """
        if not self._is_fitted:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        # Convert inputs to numpy arrays
        X = np.asarray(X)
        
        # Compute the incremental effect estimate
        estimates = []
        
        # Main estimate
        est = self._compute_incremental_effect(X, S_base, S_new)
        estimates.append(est)
        
        # Bootstrap for confidence intervals
        if n_bootstrap > 0:
            bootstrap_estimates = []
            n_samples = X.shape[0]
            for _ in range(n_bootstrap):
                # Sample with replacement
                indices = np.random.choice(n_samples, n_samples, replace=True)
                X_boot = X[indices]
                S_base_boot = S_base[indices] if S_base.ndim > 0 else S_base
                S_new_boot = S_new[indices] if S_new.ndim > 0 else S_new
                
                boot_est = self._compute_incremental_effect(X_boot, S_base_boot, S_new_boot)
                bootstrap_estimates.append(boot_est)
            
            # Calculate confidence intervals
            lower_idx = int(n_bootstrap * alpha / 2)
            upper_idx = int(n_bootstrap * (1 - alpha / 2))
            sorted_estimates = np.sort(bootstrap_estimates)
            lower_bound = sorted_estimates[lower_idx]
            upper_bound = sorted_estimates[upper_idx]
        else:
            lower_bound = np.nan
            upper_bound = np.nan
        
        return {
            'estimate': est,
            'conf_int_lower': lower_bound,
            'conf_int_upper': upper_bound,
            'alpha': alpha
        }
    
    def _compute_incremental_effect(
        self, 
        X: np.ndarray, 
        S_base: np.ndarray,
        S_new: np.ndarray
    ) -> float:
        """
        Compute the incremental scenario effect.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features/covariates.
        S_base : array-like
            Base scenario vectors.
        S_new : array-like
            New scenario vectors.
            
        Returns
        -------
        effect : float
            The incremental scenario effect.
        """
        # For each unit, compute Y(1,S_new) - Y(1,S_base)
        n_samples = X.shape[0]
        incremental_effects = np.zeros(n_samples)
        
        for i in range(n_samples):
            # Predict outcome under treatment with new scenario
            y1_new = self._predict_outcome(X[i:i+1], 1, S_new[i] if S_new.ndim > 0 else S_new)
            
            # Predict outcome under treatment with base scenario
            y1_base = self._predict_outcome(X[i:i+1], 1, S_base[i] if S_base.ndim > 0 else S_base)
            
            # Compute difference
            incremental_effects[i] = y1_new - y1_base
        
        # Return the mean effect
        return np.mean(incremental_effects)
