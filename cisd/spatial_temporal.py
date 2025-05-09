"""
Utilities for handling spatial dependencies and longitudinal data in causal inference.

This module provides tools for causal inference with spatially or temporally correlated data,
which is common in active transportation research.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable
from sklearn.base import BaseEstimator, TransformerMixin
import warnings

# Check for spatial dependencies
try:
    import geopandas as gpd
    from shapely.geometry import Point
    from libpysal.weights import Queen, KNN, Kernel
    HAS_SPATIAL = True
except ImportError:
    HAS_SPATIAL = False
    warnings.warn(
        "Spatial dependencies require geopandas, shapely, and libpysal. "
        "Install with: pip install geopandas shapely libpysal"
    )

# Check for statistical modeling dependencies
try:
    from statsmodels.formula.api import ols
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    warnings.warn(
        "Statistical modeling requires statsmodels. "
        "Install with: pip install statsmodels"
    )

# Check for machine learning dependencies
try:
    from sklearn.linear_model import LinearRegression
    from sklearn.cluster import SpectralClustering
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    warnings.warn(
        "Machine learning utilities require scikit-learn. "
        "Install with: pip install scikit-learn"
    )

# Check for spatial analysis dependencies
try:
    from esda.moran import Moran_Local
    from spreg import ML_Lag, ML_Error
    HAS_SPATIAL_STATS = True
except ImportError:
    HAS_SPATIAL_STATS = False
    warnings.warn(
        "Spatial statistics require esda and spreg. "
        "Install with: pip install esda spreg"
    )

# Check for visualization dependencies
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn(
        "Visualization requires matplotlib. "
        "Install with: pip install matplotlib"
    )

class SpatialDependencyHandler(BaseEstimator, TransformerMixin):
    """
    Handle spatial dependencies in causal inference models.
    
    This class helps adjust for spatial autocorrelation in treatment effects,
    which is important for transportation infrastructure interventions that have
    spatial spillover effects.
    
    Parameters
    ----------

    weight_type : str, default='queen'
        Type of spatial weights matrix ('queen', 'knn', 'kernel', or 'distance')
    k : int, default=5
        Number of nearest neighbors for KNN weights
    kernel_bandwidth : float, optional
        Bandwidth for kernel weights
    distance_threshold : float, optional
        Distance threshold for distance-based weights
    standardize : bool, default=True
        Whether to row-standardize the spatial weights matrix
    """
    
    def __init__(
        self,
        weight_type: str = 'queen',
        k: int = 5,
        kernel_bandwidth: Optional[float] = None,
        distance_threshold: Optional[float] = None,
        standardize: bool = True
    ):
        if not HAS_SPATIAL:
            raise ImportError(
                "SpatialDependencyHandler requires geopandas, shapely, and libpysal. "
                "Install with: pip install geopandas shapely libpysal"
            )
        
        self.weight_type = weight_type
        self.k = k
        self.kernel_bandwidth = kernel_bandwidth
        self.distance_threshold = distance_threshold
        self.standardize = standardize
        self._is_fitted = False
    
    def fit(self, X, y=None):
        """
        Fit spatial weights matrix to the data.
        
        Parameters
        ----------
        X : geopandas.GeoDataFrame
            Spatial data with geometry column
        y : array-like, optional
            Not used, kept for API consistency
            
        Returns
        -------
        self : object
            Returns self.
        """
        if not isinstance(X, gpd.GeoDataFrame):
            raise ValueError("X must be a GeoDataFrame with a geometry column")
            
        # Create spatial weights matrix based on specified type
        if self.weight_type == 'queen':
            self.weights = Queen.from_dataframe(X)
        elif self.weight_type == 'knn':
            self.weights = KNN.from_dataframe(X, k=self.k)
        elif self.weight_type == 'kernel':
            bandwidth = self.kernel_bandwidth if self.kernel_bandwidth else None
            self.weights = Kernel.from_dataframe(X, fixed=False, k=self.k, bandwidth=bandwidth)
        elif self.weight_type == 'distance':
            try:
                from libpysal.weights import DistanceBand
                self.weights = DistanceBand.from_dataframe(X, threshold=self.distance_threshold)
            except ImportError:
                raise ImportError("Distance weights require libpysal to be installed")
        else:
            raise ValueError(f"Unknown weight_type: {self.weight_type}")
            
        # Row-standardize weights if requested
        if self.standardize:
            self.weights.transform = 'r'
            
        self._is_fitted = True
        return self
    
    def transform(self, X):
        """
        Create spatially lagged variables for the input features.
        
        Parameters
        ----------
        X : geopandas.GeoDataFrame
            Spatial data with geometry column and features to lag
            
        Returns
        -------
        X_with_lags : geopandas.GeoDataFrame
            Original dataframe with added spatial lag columns
        """
        if not self._is_fitted:
            raise ValueError("SpatialDependencyHandler not fitted yet")
            
        if not isinstance(X, gpd.GeoDataFrame):
            raise ValueError("X must be a GeoDataFrame with a geometry column")
            
        # Create a copy to avoid modifying the original
        X_with_lags = X.copy()
        
        # Get numeric columns to create lags for
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        # Create spatial lags for each numeric column
        for col in numeric_cols:
            lag_col_name = f"{col}_spatial_lag"
            X_with_lags[lag_col_name] = self.weights.sparse * X[col].values
            
        return X_with_lags
    
    def adjust_effect_estimates(self, effect_estimates, X):
        """
        Adjust treatment effect estimates for spatial autocorrelation.
        
        Parameters
        ----------
        effect_estimates : array-like
            Individual treatment effect estimates
        X : geopandas.GeoDataFrame
            Spatial data with geometry column
            
        Returns
        -------
        adjusted_effects : array-like
            Spatially adjusted treatment effect estimates
        """
        if not self._is_fitted:
            raise ValueError("SpatialDependencyHandler not fitted yet")
            
        if not isinstance(X, gpd.GeoDataFrame):
            raise ValueError("X must be a GeoDataFrame with a geometry column")
            
        # Calculate spatial lag of effect estimates
        effect_lag = self.weights.sparse * effect_estimates
        
        # Adjust estimates based on spatial autocorrelation
        rho = np.corrcoef(effect_estimates, effect_lag)[0, 1]
        adjusted_effects = effect_estimates - rho * effect_lag
        
        return adjusted_effects
        
    def local_morans_i(self, values):
        """
        Calculate Local Moran's I statistic for spatial autocorrelation.
        
        Parameters
        ----------
        values : array-like
            Values to calculate Local Moran's I for, typically effect estimates
            
        Returns
        -------
        lisa_values : array-like
            Local Moran's I values for each observation
        p_values : array-like
            P-values for Local Moran's I statistics
        """
        if not self._is_fitted:
            raise ValueError("SpatialDependencyHandler not fitted yet")
            
        try:
            from esda.moran import Moran_Local
        except ImportError:
            raise ImportError(
                "Local Moran's I calculation requires esda package. "
                "Install with: pip install esda"
            )
            
        # Calculate Local Moran's I
        lisa = Moran_Local(values, self.weights)
        
        return lisa.Is, lisa.p_sim
        
    def spatial_lag_model(self, X, y, exog_vars=None):
        """
        Fit a spatial lag model (SLM).
        
        Parameters
        ----------
        X : geopandas.GeoDataFrame
            Spatial data with geometry column and covariates
        y : array-like
            Dependent variable
        exog_vars : list of str, optional
            Names of exogenous variables to include in the model.
            If None, all numeric columns in X will be used.
            
        Returns
        -------
        model_results : object
            Results from the fitted spatial lag model
        """
        if not self._is_fitted:
            raise ValueError("SpatialDependencyHandler not fitted yet")
            
        try:
            from spreg import ML_Lag
        except ImportError:
            raise ImportError(
                "Spatial regression models require spreg package. "
                "Install with: pip install spreg"
            )
            
        # Determine exogenous variables
        if exog_vars is None:
            exog_vars = X.select_dtypes(include=[np.number]).columns.tolist()
            
        # Create design matrix (with constant term)
        X_model = X[exog_vars].values
        X_model = np.column_stack((np.ones(len(X_model)), X_model))
        
        # Fit spatial lag model
        model = ML_Lag(y, X_model, w=self.weights, name_y=None, 
                       name_x=['CONSTANT'] + exog_vars, name_w=None)
        
        return model
        
    def spatial_error_model(self, X, y, exog_vars=None):
        """
        Fit a spatial error model (SEM).
        
        Parameters
        ----------
        X : geopandas.GeoDataFrame
            Spatial data with geometry column and covariates
        y : array-like
            Dependent variable
        exog_vars : list of str, optional
            Names of exogenous variables to include in the model.
            If None, all numeric columns in X will be used.
            
        Returns
        -------

        model_results : object
            Results from the fitted spatial error model
        """
        if not self._is_fitted:
            raise ValueError("SpatialDependencyHandler not fitted yet")
            
        try:
            from spreg import ML_Error
        except ImportError:
            raise ImportError(
                "Spatial regression models require spreg package. "
                "Install with: pip install spreg"
            )
            
        # Determine exogenous variables
        if exog_vars is None:
            exog_vars = X.select_dtypes(include=[np.number]).columns.tolist()
            
        # Create design matrix (with constant term)
        X_model = X[exog_vars].values
        X_model = np.column_stack((np.ones(len(X_model)), X_model))
        
        # Fit spatial error model
        model = ML_Error(y, X_model, w=self.weights, name_y=None,
                        name_x=['CONSTANT'] + exog_vars, name_w=None)
        
        return model
        
    def detect_spatial_clusters(self, values, p_threshold=0.05):
        """
        Detect spatial clusters and outliers using Local Moran's I.
        
        Parameters
        ----------
        values : array-like
            Values to analyze for clusters
        p_threshold : float, default=0.05
            P-value threshold for significance
            
        Returns
        -------
        cluster_labels : array-like
            Array of cluster labels:
            1: High-High (hotspot)
            2: Low-Low (coldspot)  
            3: High-Low (outlier)
            4: Low-High (outlier)
            0: Not significant
        """
        # Calculate Local Moran's I
        lisa_values, p_values = self.local_morans_i(values)
        
        # Standardize the input values
        z_values = (values - np.mean(values)) / np.std(values)
        
        # Standardize the spatial lag of values
        lag_values = self.weights.sparse * values
        z_lag = (lag_values - np.mean(lag_values)) / np.std(lag_values)
        
        # Initialize cluster labels
        cluster_labels = np.zeros_like(values, dtype=int)
        
        # Assign cluster types for significant locations
        sig_indicators = p_values < p_threshold
        
        # High-High
        cluster_labels[(sig_indicators) & (z_values > 0) & (z_lag > 0)] = 1
        # Low-Low
        cluster_labels[(sig_indicators) & (z_values < 0) & (z_lag < 0)] = 2
        # High-Low
        cluster_labels[(sig_indicators) & (z_values > 0) & (z_lag < 0)] = 3
        # Low-High
        cluster_labels[(sig_indicators) & (z_values < 0) & (z_lag > 0)] = 4
        
        return cluster_labels


class LongitudinalDataHandler(BaseEstimator):
    """
    Handle longitudinal data in causal inference models.
    
    This class implements methods for causal inference with panel data,
    which is common in before-after studies of transportation interventions.
    
    Parameters
    ----------
    method : str, default='did'
        Method for longitudinal causal inference:
        - 'did': Difference-in-differences
        - 'fe': Fixed effects
        - 'synth': Synthetic control
        - 'staggered': Staggered adoption DiD
        - 'dynamic': Dynamic treatment effects
    aggregation : str, default='mean'
        How to aggregate repeated measures ('mean', 'median', or 'last')
    event_window : tuple, optional
        Event window for dynamic treatment effects (pre_periods, post_periods)
    """
    
    def __init__(
        self,
        method: str = 'did',
        aggregation: str = 'mean',
        event_window: Optional[Tuple[int, int]] = None
    ):
        self.method = method
        self.aggregation = aggregation
        self.event_window = event_window if event_window else (-4, 4)  # Default window
        self._is_fitted = False
    
    def fit(self, X, D, Y, time_var, id_var):
        """
        Fit the longitudinal model to the data.
        
        Parameters
        ----------
        X : array-like or DataFrame
            Covariates
        D : array-like
            Treatment indicators
        Y : array-like
            Outcome values
        time_var : str or array-like
            Time period indicators
        id_var : str or array-like
            Unit identifiers for longitudinal data
            
        Returns
        -------
        self : object
            Returns self.
        """
        # Convert inputs to DataFrames for easier handling
        if isinstance(X, pd.DataFrame):
            data = X.copy()
            if isinstance(time_var, str) and time_var in data.columns:
                time = data[time_var]
            else:
                time = time_var
                data['_time'] = time
                time_var = '_time'
                
            if isinstance(id_var, str) and id_var in data.columns:
                ids = data[id_var]
            else:
                ids = id_var
                data['_id'] = ids
                id_var = '_id'
                
            # Add treatment and outcome if not already in DataFrame
            if 'treatment' not in data.columns:
                data['treatment'] = D
            if 'outcome' not in data.columns:
                data['outcome'] = Y
        else:
            # Create DataFrame from arrays
            data = pd.DataFrame({
                '_id': id_var if hasattr(id_var, '__len__') else np.repeat(0, len(X)),
                '_time': time_var if hasattr(time_var, '__len__') else np.repeat(0, len(X)),
                'treatment': D,
                'outcome': Y
            })
            
            # Add covariates
            if X is not None and X.shape[1] > 0:
                for i in range(X.shape[1]):
                    data[f'X{i+1}'] = X[:, i]
                    
            id_var = '_id'
            time_var = '_time'
        
        # Store variables for later use
        self.data_ = data
        self.id_var_ = id_var
        self.time_var_ = time_var
        self.pre_periods_ = data.loc[data['treatment'] == 0, time_var].unique()
        self.post_periods_ = data.loc[data['treatment'] == 1, time_var].unique()
        
        # Default time periods for pre/post if not clear from data
        if len(self.pre_periods_) == 0 or len(self.post_periods_) == 0:
            all_periods = sorted(data[time_var].unique())
            midpoint = len(all_periods) // 2
            self.pre_periods_ = all_periods[:midpoint]
            self.post_periods_ = all_periods[midpoint:]
            
        # Get treatment timing for each unit
        self.treatment_timing_ = {}
        for unit_id in data[id_var].unique():
            unit_data = data[data[id_var] == unit_id]
            if 1 in unit_data['treatment'].values:
                first_treated = unit_data.loc[unit_data['treatment'] == 1, time_var].min()
                self.treatment_timing_[unit_id] = first_treated
            else:
                self.treatment_timing_[unit_id] = np.inf  # Never treated
        
        # Identify control and treated groups
        self.control_ids_ = [
            unit_id for unit_id, timing in self.treatment_timing_.items() 
            if timing == np.inf
        ]
        self.treated_ids_ = [
            unit_id for unit_id, timing in self.treatment_timing_.items() 
            if timing != np.inf
        ]
        
        # Fit model based on selected method
        if self.method == 'did':
            self._fit_difference_in_differences()
        elif self.method == 'fe':
            self._fit_fixed_effects()
        elif self.method == 'synth':
            self._fit_synthetic_control()
        elif self.method == 'staggered':
            self._fit_staggered_did()
        elif self.method == 'dynamic':
            self._fit_dynamic_did()
        else:
            raise ValueError(f"Unknown method: {self.method}")
            
        self._is_fitted = True
        return self
    
    def _fit_difference_in_differences(self):
        """Fit a difference-in-differences model."""
        data = self.data_
        id_var = self.id_var_
        time_var = self.time_var_
        
        # Calculate pre-treatment means for each unit
        pre_data = data[data[time_var].isin(self.pre_periods_)]
        if self.aggregation == 'mean':
            pre_means = pre_data.groupby(id_var)['outcome'].mean()
        elif self.aggregation == 'median':
            pre_means = pre_data.groupby(id_var)['outcome'].median()
        elif self.aggregation == 'last':
            pre_means = pre_data.groupby(id_var)['outcome'].last()
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")
            
        # Calculate post-treatment means for each unit
        post_data = data[data[time_var].isin(self.post_periods_)]
        if self.aggregation == 'mean':
            post_means = post_data.groupby(id_var)['outcome'].mean()
        elif self.aggregation == 'median':
            post_means = post_data.groupby(id_var)['outcome'].median()
        elif self.aggregation == 'last':
            post_means = post_data.groupby(id_var)['outcome'].last()
        
        # Calculate before-after differences for each unit
        unit_diffs = post_means - pre_means
        
        # Average differences by treatment group
        treated_diff = unit_diffs[self.treated_ids_].mean()
        control_diff = unit_diffs[self.control_ids_].mean()
        
        # DID estimator is the difference of differences
        self.ate_ = treated_diff - control_diff
    
    def _fit_fixed_effects(self):
        """Fit a two-way fixed effects model."""
        if not HAS_STATSMODELS:
            raise ImportError(
                "Fixed effects models require statsmodels. "
                "Install with: pip install statsmodels"
            )
        
        data = self.data_
        id_var = self.id_var_
        time_var = self.time_var_
        
        # Create dummy variables for units and time periods
        dummies_id = pd.get_dummies(data[id_var], prefix='unit', drop_first=True)
        dummies_time = pd.get_dummies(data[time_var], prefix='time', drop_first=True)
        
        # Combine with original data
        model_data = pd.concat([data, dummies_id, dummies_time], axis=1)
        
        # Construct formula with unit and time fixed effects
        fe_formula = "outcome ~ treatment"
        for col in dummies_id.columns:
            fe_formula += f" + {col}"
        for col in dummies_time.columns:
            fe_formula += f" + {col}"
            
        # Fit OLS model with fixed effects
        self.fe_model_ = ols(fe_formula, data=model_data).fit()
        
        # Extract treatment effect estimate
        self.ate_ = self.fe_model_.params['treatment']
    
    def _fit_synthetic_control(self):
        """Fit a synthetic control model."""
        # This is a simplified version of synthetic control for illustration
        # For real applications, consider using the 'synth' R package via rpy2
        # or the 'SparseSC' Python package
        
        if len(self.treated_ids_) != 1:
            warnings.warn(
                "Synthetic control is designed for a single treated unit. "
                "Using only the first treated unit."
            )
        
        # Get first treated unit
        treated_id = self.treated_ids_[0]
        
        data = self.data_
        id_var = self.id_var_
        time_var = self.time_var_
        
        # Extract data for treated unit
        treated_data = data[data[id_var] == treated_id]
        treatment_time = self.treatment_timing_[treated_id]
        
        # Extract data for control units
        control_data = data[data[id_var].isin(self.control_ids_)]
        
        # Pre-treatment periods for matching
        pre_periods = [t for t in self.pre_periods_ if t < treatment_time]
        
        # Create matrices for synthetic control
        Z1 = treated_data.loc[treated_data[time_var].isin(pre_periods), 'outcome'].values
        Z0 = control_data.pivot_table(
            index=time_var,
            columns=id_var,
            values='outcome'
        ).loc[pre_periods, self.control_ids_].values
        
        # Compute synthetic weights (simplified with OLS)
        if not HAS_SKLEARN:
            raise ImportError(
                "Synthetic control requires scikit-learn. "
                "Install with: pip install scikit-learn"
            )
            
        model = LinearRegression(fit_intercept=False, positive=True)
        model.fit(Z0, Z1)
        
        # Store weights and synthetic control
        self.sc_weights_ = model.coef_
        self.sc_model_ = model
        
        # Compute ATT using post-treatment data
        post_periods = [t for t in self.post_periods_ if t >= treatment_time]
        
        # Extract post-treatment outcomes
        Y1_post = treated_data.loc[treated_data[time_var].isin(post_periods), 'outcome'].values
        Y0_post = control_data.pivot_table(
            index=time_var,
            columns=id_var,
            values='outcome'
        ).loc[post_periods, self.control_ids_].values
        
        # Predict counterfactual using synthetic control
        Y0_synth = Y0_post @ self.sc_weights_
        
        # Compute treatment effect
        self.ate_ = np.mean(Y1_post - Y0_synth)
        
    def _fit_staggered_did(self):
        """
        Fit a staggered adoption difference-in-differences model.
        
        This method implements the approach described in Callaway and Sant'Anna (2020)
        for staggered treatment adoption settings.
        """
        data = self.data_
        id_var = self.id_var_
        time_var = self.time_var_
        
        # Group units by treatment cohort
        cohorts = {}
        for unit_id, treat_time in self.treatment_timing_.items():
            if treat_time < np.inf:  # Only include treated units
                if treat_time not in cohorts:
                    cohorts[treat_time] = []
                cohorts[treat_time].append(unit_id)
                
        # Calculate cohort-specific treatment effects
        cohort_effects = {}
        
        for cohort_time, cohort_units in cohorts.items():
            # For each cohort, calculate ATT using only never-treated as controls
            cohort_data = data[(data[id_var].isin(cohort_units)) | 
                              (data[id_var].isin(self.control_ids_))]
            
            # Split into pre and post periods for this cohort
            pre_cohort = cohort_data[cohort_data[time_var] < cohort_time]
            post_cohort = cohort_data[cohort_data[time_var] >= cohort_time]
            
            # Calculate pre-treatment means
            if self.aggregation == 'mean':
                pre_means = pre_cohort.groupby(id_var)['outcome'].mean()
            elif self.aggregation == 'median':
                pre_means = pre_cohort.groupby(id_var)['outcome'].median()
            else:
                pre_means = pre_cohort.groupby(id_var)['outcome'].last()
                
            # Calculate post-treatment means
            if self.aggregation == 'mean':
                post_means = post_cohort.groupby(id_var)['outcome'].mean()
            elif self.aggregation == 'median':
                post_means = post_cohort.groupby(id_var)['outcome'].median()
            else:
                post_means = post_cohort.groupby(id_var)['outcome'].last()
                
            # Calculate before-after differences
            unit_diffs = post_means - pre_means
            
            # DiD for this cohort
            treated_diff = unit_diffs[cohort_units].mean()
            control_diff = unit_diffs[self.control_ids_].mean()
            
            cohort_effects[cohort_time] = treated_diff - control_diff
            
        # Calculate overall treatment effect as weighted average
        total_treated = sum(len(units) for units in cohorts.values())
        self.cohort_effects_ = cohort_effects
        
        # Store cohort-specific effects and overall ATE
        self.ate_ = sum(
            effect * len(cohorts[cohort]) / total_treated
            for cohort, effect in cohort_effects.items()
        )
        
    def _fit_dynamic_did(self):
        """
        Fit a dynamic difference-in-differences model to estimate
        treatment effects over time relative to treatment.
        """
        data = self.data_
        id_var = self.id_var_
        time_var = self.time_var_
        
        # Create relative time indicators for each observation
        data = data.copy()
        data['relative_time'] = np.nan
        
        for unit_id in data[id_var].unique():
            treat_time = self.treatment_timing_[unit_id]
            if treat_time < np.inf:  # Only for treated units
                unit_data_idx = data[data[id_var] == unit_id].index
                data.loc[unit_data_idx, 'relative_time'] = (
                    data.loc[unit_data_idx, time_var] - treat_time
                )
        
        # Restrict to the event window
        min_rel_time, max_rel_time = self.event_window
        event_data = data[
            (data['relative_time'].isnull()) |  # Control units
            ((data['relative_time'] >= min_rel_time) & 
             (data['relative_time'] <= max_rel_time))  # Event window for treated
        ].copy()
        
        # Create dummy variables for relative time periods
        rel_time_dummies = {}
        for t in range(min_rel_time, max_rel_time + 1):
            if t != -1:  # Use -1 as reference period
                rel_time_dummies[f'rel_time_{t}'] = (event_data['relative_time'] == t).astype(int)
        
        # Add dummies to data
        for name, dummy in rel_time_dummies.items():
            event_data[name] = dummy
            
        # Fit model with unit and time fixed effects
        if not HAS_STATSMODELS:
            raise ImportError(
                "Dynamic DiD requires statsmodels. "
                "Install with: pip install statsmodels"
            )
            
        # Create dummy variables for units and time periods
        dummies_id = pd.get_dummies(event_data[id_var], prefix='unit', drop_first=True)
        dummies_time = pd.get_dummies(event_data[time_var], prefix='time', drop_first=True)
        
        # Combine with original data
        model_data = pd.concat([event_data, dummies_id, dummies_time], axis=1)
        
        # Construct formula with unit and time fixed effects
        fe_formula = "outcome ~ " + " + ".join(rel_time_dummies.keys())
        for col in dummies_id.columns:
            fe_formula += f" + {col}"
        for col in dummies_time.columns:
            fe_formula += f" + {col}"
            
        # Fit OLS model with fixed effects
        self.dynamic_model_ = ols(fe_formula, data=model_data).fit()
        
        # Extract dynamic treatment effects
        self.dynamic_effects_ = {
            t: self.dynamic_model_.params.get(f'rel_time_{t}', 0)
            for t in range(min_rel_time, max_rel_time + 1)
            if t != -1  # Reference period
        }
        
        # Calculate ATE as average of post-treatment effects
        post_effects = [
            effect for time, effect in self.dynamic_effects_.items()
            if time >= 0
        ]
        self.ate_ = np.mean(post_effects) if post_effects else 0
    
    def estimate_effect(self, X=None, D=None, Y=None, time_var=None, id_var=None):
        """
        Estimate causal effect using the fitted longitudinal model.
        
        Parameters
        ----------
        X : array-like or DataFrame, optional
            New covariates (not used in all methods)
        D : array-like, optional
            New treatment indicators (not used in all methods)
        Y : array-like, optional
            New outcome values (not used in all methods)
        time_var : str or array-like, optional
            New time period indicators (not used in all methods)
        id_var : str or array-like, optional
            New unit identifiers (not used in all methods)
            
        Returns
        -------
        effect : float
            Estimated causal effect
        """
        if not self._is_fitted:
            raise ValueError("LongitudinalDataHandler not fitted yet")
            
        return self.ate_
    
    def event_study_plot(self, time_range=None, figsize=(12, 8)):
        """
        Create an event study plot to visualize treatment effects over time.
        
        Parameters
        ----------
        time_range : list or tuple, optional
            Range of time periods to include in the plot (relative to treatment)
        figsize : tuple, default=(12, 8)
            Figure size
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            Event study plot
        """
        if not HAS_MATPLOTLIB:
            raise ImportError(
                "Event study plots require matplotlib. "
                "Install with: pip install matplotlib"
            )
            
        if not self._is_fitted:
            raise ValueError("LongitudinalDataHandler not fitted yet")
            
        # If dynamic effects are available, use those
        if hasattr(self, 'dynamic_effects_'):
            fig, ax = plt.subplots(figsize=figsize)
            
            # Get periods and effects
            periods = sorted(self.dynamic_effects_.keys())
            effects = [self.dynamic_effects_[p] for p in periods]
            
            # Apply time range filter if specified
            if time_range:
                mask = [(p >= time_range[0]) and (p <= time_range[1]) for p in periods]
                periods = [p for i, p in enumerate(periods) if mask[i]]
                effects = [e for i, e in enumerate(effects) if mask[i]]
                
            # Plot the effects
            ax.scatter(periods, effects, color='blue', s=40, zorder=3)
            ax.plot(periods, effects, color='blue', linestyle='-', alpha=0.5, zorder=2)
            
            # Add reference lines
            ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, 
                     label='Treatment Time')
            ax.axhline(y=0, color='black', linestyle=':', alpha=0.7)
            
            # Set labels and title
            ax.set_xlabel('Time Relative to Treatment')
            ax.set_ylabel('Treatment Effect')
            ax.set_title('Event Study: Dynamic Treatment Effects')
            ax.grid(True, alpha=0.3)
            
            return fig
        
        # Otherwise, use raw data to create event study plot
        data = self.data_
        id_var = self.id_var_
        time_var = self.time_var_
        
        # Reshape data for event study
        event_study_data = []
        
        for unit_id in data[id_var].unique():
            unit_data = data[data[id_var] == unit_id]
            treatment_time = self.treatment_timing_.get(unit_id, np.inf)
            
            if treatment_time != np.inf:
                # For treated units, calculate time relative to treatment
                for _, row in unit_data.iterrows():
                    event_time = row[time_var] - treatment_time
                    event_study_data.append({
                        'unit_id': unit_id,
                        'time': row[time_var],
                        'event_time': event_time,
                        'outcome': row['outcome'],
                        'treated': 1
                    })
        
        # Create event study dataframe
        event_df = pd.DataFrame(event_study_data)
        
        # Apply time range if specified
        if time_range:
            event_df = event_df[(event_df['event_time'] >= time_range[0]) & 
                              (event_df['event_time'] <= time_range[1])]
        
        # Calculate mean outcome by event time
        event_means = event_df.groupby('event_time')['outcome'].mean().reset_index()
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot point estimates
        ax.scatter(event_means['event_time'], event_means['outcome'], 
                 color='blue', label='Point Estimate')
        
        # Connect the dots with a line
        ax.plot(event_means['event_time'], event_means['outcome'], 
              color='blue', linestyle='-', alpha=0.5)
        
        # Add reference lines
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, 
                 label='Treatment Time')
        ax.axhline(y=event_means.loc[event_means['event_time'] < 0, 'outcome'].mean(), 
                 color='green', linestyle=':', alpha=0.7, 
                 label='Pre-Treatment Mean')
        
        # Set labels and title
        ax.set_xlabel('Time Relative to Treatment')
        ax.set_ylabel('Outcome')
        ax.set_title('Event Study: Treatment Effect Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
        
    def create_spatial_temporal_clusters(self, geo_df, outcome_col='outcome', n_clusters=5):
        """
        Create spatial-temporal clusters from panel data with geospatial information.
        
        This method combines spatial proximity with temporal trends to identify areas
        with similar treatment response patterns.
        
        Parameters
        ----------

        geo_df : geopandas.GeoDataFrame
            GeoDataFrame with geometries for spatial units
        outcome_col : str, default='outcome'
            Name of column containing outcomes
        n_clusters : int, default=5
            Number of clusters to create
            
        Returns
        -------
        clustered_df : geopandas.GeoDataFrame
            GeoDataFrame with added cluster labels
        """
        if not HAS_SPATIAL:
            raise ImportError(
                "Spatial-temporal clustering requires geopandas. "
                "Install with: pip install geopandas"
            )
            
        if not HAS_SKLEARN:
            raise ImportError(
                "Clustering requires scikit-learn. "
                "Install with: pip install scikit-learn"
            )
        
        data = self.data_
        id_var = self.id_var_
        time_var = self.time_var_
        
        # Create temporal profiles for each unit
        temporal_profiles = data.pivot_table(
            index=id_var,
            columns=time_var,
            values=outcome_col,
            aggfunc='mean'
        )
        
        # Fill missing values with column means
        temporal_profiles = temporal_profiles.fillna(temporal_profiles.mean())
        
        # Create spatial weights matrix
        try:
            from libpysal.weights import Queen
        except ImportError:
            raise ImportError(
                "Spatial weights require libpysal. "
                "Install with: pip install libpysal"
            )
        
        # Ensure geo_df has same ids as temporal_profiles
        geo_df = geo_df.loc[geo_df.index.isin(temporal_profiles.index)]
        temporal_profiles = temporal_profiles.loc[temporal_profiles.index.isin(geo_df.index)]
        
        # Align indices
        geo_df = geo_df.loc[temporal_profiles.index]
        
        # Create weights matrix
        w = Queen.from_dataframe(geo_df)
        w.transform = 'r'
        
        # Create adjacency matrix
        adj_matrix = w.to_pandas_adjacency()
        
        # Use spectral clustering to find clusters
        clustering = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            random_state=42,
            assign_labels='discretize'
        )
        
        # Fit clustering using adjacency matrix
        clusters = clustering.fit_predict(adj_matrix)
        
        # Add clusters to geo_df
        result = geo_df.copy()
        result['cluster'] = clusters
        
        # Calculate cluster means
        cluster_means = {}
        for cluster in range(n_clusters):
            cluster_units = result.index[result['cluster'] == cluster]
            cluster_means[cluster] = temporal_profiles.loc[cluster_units].mean()
            
        self.cluster_means_ = pd.DataFrame(cluster_means).T
        self.clusters_ = clusters
        
        return result


def create_spatial_panel_data(
    geo_df: 'gpd.GeoDataFrame',
    n_periods: int = 5,
    treatment_time: int = 2,
    treatment_share: float = 0.3,
    spatial_correlation: float = 0.5,
    temporal_correlation: float = 0.7,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Create a synthetic spatial panel dataset for testing spatio-temporal methods.
    
    Parameters
    ----------
    geo_df : geopandas.GeoDataFrame
        Spatial data with geometry column
    n_periods : int, default=5
        Number of time periods
    treatment_time : int, default=2
        Time period when treatment starts (for treated units)
    treatment_share : float, default=0.3
        Share of units that receive treatment
    spatial_correlation : float, default=0.5
        Strength of spatial correlation in outcome
    temporal_correlation : float, default=0.7
        Strength of temporal correlation in outcome
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    panel_df : pandas.DataFrame
        Synthetic panel dataset with spatial and temporal structure
    """
    if not HAS_SPATIAL:
        raise ImportError(
            "create_spatial_panel_data requires geopandas, shapely, and libpysal. "
            "Install with: pip install geopandas shapely libpysal"
        )
        
    if seed is not None:
        np.random.seed(seed)
        
    # Create spatial weights matrix
    w = Queen.from_dataframe(geo_df)
    w.transform = 'r'
    
    # Assign treatment to some units
    n_units = len(geo_df)
    n_treated = int(n_units * treatment_share)
    treatment_assignment = np.zeros(n_units)
    treatment_assignment[:n_treated] = 1
    np.random.shuffle(treatment_assignment)
    
    # Create initial random effects for each unit
    unit_effects = np.random.normal(0, 1, n_units)
    
    # Create panel data
    panel_data = []
    
    # For each time period
    y_prev = np.random.normal(0, 1, n_units)  # Initial outcome values
    
    for t in range(n_periods):
        # Create spatially correlated shock
        spatial_shock = spatial_correlation * (w.sparse * y_prev)
        
        # Create temporally correlated component
        temporal_component = temporal_correlation * y_prev
        
        # Treatment effect (only for treated units after treatment time)
        treatment = np.zeros(n_units)
        if t >= treatment_time:
            treatment = treatment_assignment
            
        # Treatment effect (2.0 plus spillover effects)
        treatment_effect = 2.0 * treatment
        
        # Spatially correlated treatment effect spillovers
        treatment_spillover = 1.0 * (w.sparse * treatment)
        
        # Calculate outcome
        y = (
            unit_effects +                    # Unit fixed effects
            0.5 * t +                         # Time trend
            spatial_shock +                   # Spatial correlation
            temporal_component +              # Temporal correlation
            treatment_effect +                # Direct treatment effect
            treatment_spillover +             # Treatment spillover
            np.random.normal(0, 0.5, n_units) # Random noise
        )
        
        # Update previous outcome for next period
        y_prev = y
        
        # Build this period's data
        for i in range(n_units):
            unit_id = geo_df.index[i] if hasattr(geo_df.index, '__getitem__') else i
            panel_data.append({
                'unit_id': unit_id,
                'time': t,
                'treatment': treatment[i],
                'outcome': y[i],
                'x1': np.random.normal(0, 1),  # Random covariates
                'x2': np.random.normal(0, 1)
            })
    
    # Create DataFrame
    panel_df = pd.DataFrame(panel_data)
    
    return panel_df
