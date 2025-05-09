"""
Causal learning methods with orthogonal scores and influence-function corrections.
"""

import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
import warnings

class CausalLearner(BaseEstimator):
    """
    Base class for causal learners (Γ component in the three-layer architecture).
    
    This component produces orthogonal scores or influence-function corrections.
    """
    
    def __init__(self):
        self._is_fitted = False
    
    def fit(self, Z, D, Y, W=None):
        """
        Fit the causal learner.
        
        Parameters
        ----------
        Z : array-like of shape (n_samples, n_features)
            Latent features (output from Φ).
        D : array-like of shape (n_samples,)
            Treatment indicator (0 or 1).
        Y : array-like of shape (n_samples,)
            Outcome variable.
        W : array-like of shape (n_samples,), optional
            Sample weights (output from Ψ).
            
        Returns
        -------
        self : object
            Returns self.
        """
        # Should be implemented by subclasses
        self._is_fitted = True
        return self
    
    def estimate(self, Z, D=None, Y=None, W=None):
        """
        Estimate the causal effect.
        
        Parameters
        ----------
        Z : array-like of shape (n_samples, n_features)
            Latent features (output from Φ).
        D : array-like of shape (n_samples,), optional
            Treatment indicator (0 or 1).
        Y : array-like of shape (n_samples,), optional
            Outcome variable.
        W : array-like of shape (n_samples,), optional
            Sample weights (output from Ψ).
            
        Returns
        -------
        effect : dict
            Estimated causal effect and related statistics.
        """
        # Should be implemented by subclasses
        if not self._is_fitted:
            raise ValueError("Causal learner not fitted. Call fit() first.")
        
        # Default implementation: return placeholder effect
        return {'effect': 0.0, 'std_err': 0.0}
    
    def influence_function(self, Z, D, Y, W=None):
        """
        Compute the efficient influence function for each sample.
        
        Parameters
        ----------
        Z : array-like of shape (n_samples, n_features)
            Latent features (output from Φ).
        D : array-like of shape (n_samples,)
            Treatment indicator (0 or 1).
        Y : array-like of shape (n_samples,)
            Outcome variable.
        W : array-like of shape (n_samples,), optional
            Sample weights (output from Ψ).
            
        Returns
        -------
        infl : array-like of shape (n_samples,)
            Influence function values.
        """
        # Should be implemented by subclasses
        if not self._is_fitted:
            raise ValueError("Causal learner not fitted. Call fit() first.")
        
        # Default implementation: return zeros
        return np.zeros(Z.shape[0])


class DoublyRobust(CausalLearner):
    """
    Doubly robust causal estimator.
    
    Parameters
    ----------
    propensity_model : BaseEstimator
        Model to estimate propensity scores P(D=1|Z).
    outcome_models : dict or BaseEstimator
        Model(s) to estimate E[Y|Z,D]. If a dict, should have keys '0' and '1' for each treatment level.
    n_splits : int, default=5
        Number of cross-fitting splits.
    random_state : int, optional
        Random seed for reproducibility.
    """
    
    def __init__(
        self,
        propensity_model: BaseEstimator,
        outcome_models: Union[BaseEstimator, Dict[str, BaseEstimator]],
        n_splits: int = 5,
        random_state: Optional[int] = None
    ):
        super().__init__()
        self.propensity_model = propensity_model
        self.outcome_models = outcome_models
        self.n_splits = n_splits
        self.random_state = random_state
    
    def fit(self, Z, D, Y, W=None):
        """
        Fit the doubly robust estimator using cross-fitting.
        
        Parameters
        ----------
        Z : array-like of shape (n_samples, n_features)
            Latent features (output from Φ).
        D : array-like of shape (n_samples,)
            Treatment indicator (0 or 1).
        Y : array-like of shape (n_samples,)
            Outcome variable.
        W : array-like of shape (n_samples,), optional
            Sample weights (output from Ψ).
            
        Returns
        -------
        self : object
            Returns self.
        """
        # Convert inputs to numpy arrays if they're not already
        Z = np.asarray(Z)
        D = np.asarray(D)
        Y = np.asarray(Y)
        if W is not None:
            W = np.asarray(W)
        
        n_samples = Z.shape[0]
        
        # Setup cross-fitting
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        
        # Initialize storage for model predictions
        self.propensity_scores = np.zeros(n_samples)
        self.outcome_pred = np.zeros((n_samples, 2))  # For D=0 and D=1
        self.fold_indices = np.zeros(n_samples, dtype=int)
        
        # Initialize list of fitted models
        self.propensity_models_fitted = []
        self.outcome_models_fitted = []
        
        # Cross-fitting loop
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(Z)):
            self.fold_indices[test_idx] = fold_idx
            
            # Split data
            Z_train, Z_test = Z[train_idx], Z[test_idx]
            D_train, D_test = D[train_idx], D[test_idx]
            Y_train, Y_test = Y[train_idx], Y[test_idx]
            W_train = None if W is None else W[train_idx]
            
            # Fit propensity model
            prop_model = clone_estimator(self.propensity_model)
            if W_train is None:
                prop_model.fit(Z_train, D_train)
            else:
                prop_model.fit(Z_train, D_train, sample_weight=W_train)
            
            # Predict propensity scores for test set
            self.propensity_scores[test_idx] = prop_model.predict_proba(Z_test)[:, 1]
            
            # Store fitted propensity model
            self.propensity_models_fitted.append(prop_model)
            
            # Fit outcome models (either one model with D as feature or two separate models)
            if isinstance(self.outcome_models, dict):
                # Separate models for each treatment level
                outcome_models_fold = {}
                
                # Fit model for D=0
                if np.any(D_train == 0):
                    idx_0_train = (D_train == 0)
                    Z_0_train = Z_train[idx_0_train]
                    Y_0_train = Y_train[idx_0_train]
                    W_0_train = None if W_train is None else W_train[idx_0_train]
                    
                    model_0 = clone_estimator(self.outcome_models['0'])
                    if W_0_train is None:
                        model_0.fit(Z_0_train, Y_0_train)
                    else:
                        model_0.fit(Z_0_train, Y_0_train, sample_weight=W_0_train)
                    
                    outcome_models_fold['0'] = model_0
                    
                    # Predict for all test samples
                    self.outcome_pred[test_idx, 0] = model_0.predict(Z_test)
                
                # Fit model for D=1
                if np.any(D_train == 1):
                    idx_1_train = (D_train == 1)
                    Z_1_train = Z_train[idx_1_train]
                    Y_1_train = Y_train[idx_1_train]
                    W_1_train = None if W_train is None else W_train[idx_1_train]
                    
                    model_1 = clone_estimator(self.outcome_models['1'])
                    if W_1_train is None:
                        model_1.fit(Z_1_train, Y_1_train)
                    else:
                        model_1.fit(Z_1_train, Y_1_train, sample_weight=W_1_train)
                    
                    outcome_models_fold['1'] = model_1
                    
                    # Predict for all test samples
                    self.outcome_pred[test_idx, 1] = model_1.predict(Z_test)
                
                self.outcome_models_fitted.append(outcome_models_fold)
            else:
                # Single model with treatment as a feature
                Z_D_train = np.column_stack([Z_train, D_train.reshape(-1, 1)])
                model = clone_estimator(self.outcome_models)
                if W_train is None:
                    model.fit(Z_D_train, Y_train)
                else:
                    model.fit(Z_D_train, Y_train, sample_weight=W_train)
                
                # Create predictions for both D=0 and D=1
                Z_test_0 = np.column_stack([Z_test, np.zeros((Z_test.shape[0], 1))])
                Z_test_1 = np.column_stack([Z_test, np.ones((Z_test.shape[0], 1))])
                
                self.outcome_pred[test_idx, 0] = model.predict(Z_test_0)
                self.outcome_pred[test_idx, 1] = model.predict(Z_test_1)
                
                self.outcome_models_fitted.append(model)
        
        # Compute estimates and influence functions
        self._compute_estimates(Z, D, Y, W)
        
        self._is_fitted = True
        return self
    
    def _compute_estimates(self, Z, D, Y, W=None):
        """
        Compute estimates and influence functions.
        
        Parameters
        ----------
        Z : array-like of shape (n_samples, n_features)
            Latent features (output from Φ).
        D : array-like of shape (n_samples,)
            Treatment indicator (0 or 1).
        Y : array-like of shape (n_samples,)
            Outcome variable.
        W : array-like of shape (n_samples,), optional
            Sample weights (output from Ψ).
        """
        # Convert inputs to numpy arrays if they're not already
        Z = np.asarray(Z)
        D = np.asarray(D)
        Y = np.asarray(Y)
        if W is None:
            W = np.ones(Z.shape[0])
        else:
            W = np.asarray(W)
        
        n_samples = Z.shape[0]
        
        # Compute the propensity scores and outcome predictions (already done during fit)
        e_Z = self.propensity_scores
        mu_0_Z = self.outcome_pred[:, 0]
        mu_1_Z = self.outcome_pred[:, 1]
        
        # Compute efficient influence function for ATE
        self.infl_func = np.zeros(n_samples)
        
        # For treated units (D=1)
        treated = (D == 1)
        if np.any(treated):
            self.infl_func[treated] = (Y[treated] - mu_1_Z[treated]) / e_Z[treated] + mu_1_Z[treated]
        
        # For control units (D=0)
        control = (D == 0)
        if np.any(control):
            self.infl_func[control] = -(Y[control] - mu_0_Z[control]) / (1 - e_Z[control]) + mu_0_Z[control]
        
        # Compute ATE estimate
        self.ate_estimate = np.mean(mu_1_Z - mu_0_Z + self.infl_func)
        
        # Compute standard error
        self.std_err = np.std(self.infl_func) / np.sqrt(n_samples)
        
        # Compute confidence interval
        self.conf_int = [
            self.ate_estimate - 1.96 * self.std_err,
            self.ate_estimate + 1.96 * self.std_err
        ]
    
    def influence_function(self, Z, D, Y, W=None):
        """
        Compute the efficient influence function for each sample.
        
        Parameters
        ----------
        Z : array-like of shape (n_samples, n_features)
            Latent features (output from Φ).
        D : array-like of shape (n_samples,)
            Treatment indicator (0 or 1).
        Y : array-like of shape (n_samples,)
            Outcome variable.
        W : array-like of shape (n_samples,), optional
            Sample weights (output from Ψ).
            
        Returns
        -------
        infl : array-like of shape (n_samples,)
            Influence function values.
        """
        if not self._is_fitted:
            raise ValueError("Causal learner not fitted. Call fit() first.")
        
        # If this is the same data used for fitting, return stored influence functions
        if Z.shape[0] == len(self.infl_func) and np.array_equal(Z, Z):
            return self.infl_func
        
        # Otherwise, compute influence functions for new data
        # Convert inputs to numpy arrays if they're not already
        Z = np.asarray(Z)
        D = np.asarray(D)
        Y = np.asarray(Y)
        if W is None:
            W = np.ones(Z.shape[0])
        else:
            W = np.asarray(W)
        
        n_samples = Z.shape[0]
        
        # Predict propensity scores and outcomes for new data
        # For simplicity, we'll just use the last fold's models
        e_Z = self.propensity_models_fitted[-1].predict_proba(Z)[:, 1]
        
        mu_0_Z = np.zeros(n_samples)
        mu_1_Z = np.zeros(n_samples)
        
        if isinstance(self.outcome_models, dict):
            if '0' in self.outcome_models_fitted[-1]:
                mu_0_Z = self.outcome_models_fitted[-1]['0'].predict(Z)
            if '1' in self.outcome_models_fitted[-1]:
                mu_1_Z = self.outcome_models_fitted[-1]['1'].predict(Z)
        else:
            Z_0 = np.column_stack([Z, np.zeros((Z.shape[0], 1))])
            Z_1 = np.column_stack([Z, np.ones((Z.shape[0], 1))])
            
            mu_0_Z = self.outcome_models_fitted[-1].predict(Z_0)
            mu_1_Z = self.outcome_models_fitted[-1].predict(Z_1)
        
        # Compute efficient influence function for ATE
        infl_func = np.zeros(n_samples)
        
        # For treated units (D=1)
        treated = (D == 1)
        if np.any(treated):
            infl_func[treated] = (Y[treated] - mu_1_Z[treated]) / e_Z[treated] + mu_1_Z[treated]
        
        # For control units (D=0)
        control = (D == 0)
        if np.any(control):
            infl_func[control] = -(Y[control] - mu_0_Z[control]) / (1 - e_Z[control]) + mu_0_Z[control]
        
        return infl_func
    
    def estimate(self, Z, D=None, Y=None, W=None):
        """
        Estimate the causal effect.
        
        Parameters
        ----------
        Z : array-like of shape (n_samples, n_features)
            Latent features (output from Φ).
        D : array-like of shape (n_samples,), optional
            Treatment indicator (0 or 1).
        Y : array-like of shape (n_samples,), optional
            Outcome variable.
        W : array-like of shape (n_samples,), optional
            Sample weights (output from Ψ).
            
        Returns
        -------
        effect : dict
            Estimated causal effect and related statistics.
        """
        if not self._is_fitted:
            raise ValueError("Causal learner not fitted. Call fit() first.")
        
        # If D and Y are provided, re-estimate for this new data
        if D is not None and Y is not None:
            # Convert inputs to numpy arrays if they're not already
            Z = np.asarray(Z)
            D = np.asarray(D)
            Y = np.asarray(Y)
            if W is not None:
                W = np.asarray(W)
            
            n_samples = Z.shape[0]
            
            # Predict propensity scores and outcomes for new data
            # For simplicity, we'll just use the last fold's models
            e_Z = self.propensity_models_fitted[-1].predict_proba(Z)[:, 1]
            
            mu_0_Z = np.zeros(n_samples)
            mu_1_Z = np.zeros(n_samples)
            
            if isinstance(self.outcome_models, dict):
                if '0' in self.outcome_models_fitted[-1]:
                    mu_0_Z = self.outcome_models_fitted[-1]['0'].predict(Z)
                if '1' in self.outcome_models_fitted[-1]:
                    mu_1_Z = self.outcome_models_fitted[-1]['1'].predict(Z)
            else:
                Z_0 = np.column_stack([Z, np.zeros((Z.shape[0], 1))])
                Z_1 = np.column_stack([Z, np.ones((Z.shape[0], 1))])
                
                mu_0_Z = self.outcome_models_fitted[-1].predict(Z_0)
                mu_1_Z = self.outcome_models_fitted[-1].predict(Z_1)
            
            # Compute efficient influence function for ATE
            infl_func = np.zeros(n_samples)
            
            # For treated units (D=1)
            treated = (D == 1)
            if np.any(treated):
                infl_func[treated] = (Y[treated] - mu_1_Z[treated]) / e_Z[treated] + mu_1_Z[treated]
            
            # For control units (D=0)
            control = (D == 0)
            if np.any(control):
                infl_func[control] = -(Y[control] - mu_0_Z[control]) / (1 - e_Z[control]) + mu_0_Z[control]
            
            # Compute ATE estimate
            ate_estimate = np.mean(mu_1_Z - mu_0_Z + infl_func)
            
            # Compute standard error
            std_err = np.std(infl_func) / np.sqrt(n_samples)
            
            # Compute confidence interval
            conf_int = [
                ate_estimate - 1.96 * std_err,
                ate_estimate + 1.96 * std_err
            ]
            
            return {
                'ate': ate_estimate,
                'std_err': std_err,
                'conf_int': conf_int,
                'influence_function': infl_func
            }
        else:
            # Return stored estimates
            return {
                'ate': self.ate_estimate,
                'std_err': self.std_err,
                'conf_int': self.conf_int,
                'influence_function': self.infl_func
            }


class CausalForest(CausalLearner):
    """
    Causal Forest estimator for heterogeneous treatment effect estimation.
    
    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the forest.
    min_samples_leaf : int, default=5
        Minimum number of samples required to be at a leaf node.
    max_depth : int, optional
        Maximum depth of the trees. If None, nodes are expanded until all leaves are pure.
    n_splits : int, default=5
        Number of cross-fitting splits.
    honest : bool, default=True
        Whether to use honesty (split sample for structure and estimates).
    subforest_size : int, default=4
        Number of trees in each subforest for variance estimation.
    random_state : int, optional
        Random seed for reproducibility.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        min_samples_leaf: int = 5,
        max_depth: Optional[int] = None,
        n_splits: int = 5,
        honest: bool = True,
        subforest_size: int = 4,
        random_state: Optional[int] = None
    ):
        super().__init__()
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.n_splits = n_splits
        self.honest = honest
        self.subforest_size = subforest_size
        self.random_state = random_state
        
        # In a real implementation, we'd use a library like econml or causal-learn
        # Here we provide a simplified placeholder
    
    def fit(self, Z, D, Y, W=None):
        """
        Fit the causal forest.
        
        Parameters
        ----------
        Z : array-like of shape (n_samples, n_features)
            Latent features (output from Φ).
        D : array-like of shape (n_samples,)
            Treatment indicator (0 or 1).
        Y : array-like of shape (n_samples,)
            Outcome variable.
        W : array-like of shape (n_samples,), optional
            Sample weights (output from Ψ).
            
        Returns
        -------
        self : object
            Returns self.
        """
        # Convert inputs to numpy arrays if they're not already
        Z = np.asarray(Z)
        D = np.asarray(D)
        Y = np.asarray(Y)
        if W is not None:
            W = np.asarray(W)
        
        # In a real implementation, we'd fit a proper causal forest
        # Here we use a simplified approach: fit two regression forests for Y|Z,D=0 and Y|Z,D=1
        
        # Separate data by treatment
        Z_0 = Z[D == 0]
        Y_0 = Y[D == 0]
        W_0 = None if W is None else W[D == 0]
        
        Z_1 = Z[D == 1]
        Y_1 = Y[D == 1]
        W_1 = None if W is None else W[D == 1]
        
        # Fit forests
        self.forest_0 = RandomForestRegressor(
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            max_depth=self.max_depth,
            random_state=self.random_state
        )
        
        self.forest_1 = RandomForestRegressor(
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            max_depth=self.max_depth,
            random_state=self.random_state
        )
        
        if W_0 is not None:
            self.forest_0.fit(Z_0, Y_0, sample_weight=W_0)
        else:
            self.forest_0.fit(Z_0, Y_0)
            
        if W_1 is not None:
            self.forest_1.fit(Z_1, Y_1, sample_weight=W_1)
        else:
            self.forest_1.fit(Z_1, Y_1)
        
        self._is_fitted = True
        return self
    
    def estimate(self, Z, D=None, Y=None, W=None):
        """
        Estimate conditional average treatment effects.
        
        Parameters
        ----------
        Z : array-like of shape (n_samples, n_features)
            Latent features (output from Φ).
        D : array-like, optional
            Treatment indicator (not used for prediction).
        Y : array-like, optional
            Outcome variable (not used for prediction).
        W : array-like, optional
            Sample weights (not used for prediction).
            
        Returns
        -------
        effect : dict
            Estimated treatment effects and related statistics.
        """
        if not self._is_fitted:
            raise ValueError("Causal learner not fitted. Call fit() first.")
        
        # Convert inputs to numpy arrays if they're not already
        Z = np.asarray(Z)
        
        # Predict outcomes under each treatment
        Y_0_pred = self.forest_0.predict(Z)
        Y_1_pred = self.forest_1.predict(Z)
        
        # Compute CATE for each sample
        cate = Y_1_pred - Y_0_pred
        
        # Compute ATE
        ate = np.mean(cate)
        
        # In a real implementation, we'd compute proper standard errors
        # Here we use a placeholder
        std_err = np.std(cate) / np.sqrt(Z.shape[0])
        
        return {
            'cate': cate,
            'ate': ate,
            'std_err': std_err,
            'conf_int': [ate - 1.96 * std_err, ate + 1.96 * std_err]
        }


# Helper function for cloning estimators
def clone_estimator(estimator):
    """
    Clone a scikit-learn estimator.
    
    Parameters
    ----------
    estimator : BaseEstimator
        The estimator to clone.
        
    Returns
    -------
    estimator_copy : BaseEstimator
        A copy of the estimator with the same parameters.
    """
    # This is a simplified version of sklearn.base.clone
    try:
        # Use sklearn's clone if available
        from sklearn.base import clone
        return clone(estimator)
    except ImportError:
        # Simple fallback
        estimator_type = type(estimator)
        params = estimator.get_params(deep=True)
        return estimator_type(**params)
