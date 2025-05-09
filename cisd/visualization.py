"""
Visualization utilities for causal inference diagnostics and results exploration.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd

def plot_covariate_balance(
    X_treated: np.ndarray,
    X_control: np.ndarray,
    feature_names: Optional[List[str]] = None,
    before_weighting: Optional[Dict[str, np.ndarray]] = None,
    figsize: Tuple[int, int] = (12, 8),
    title: str = "Covariate Balance"
):
    """
    Plot standardized mean differences (SMD) between treated and control groups.
    
    Parameters
    ----------
    X_treated : np.ndarray
        Covariates for treated units
    X_control : np.ndarray
        Covariates for control units
    feature_names : List[str], optional
        Names of features for axis labels
    before_weighting : Dict[str, np.ndarray], optional
        Dictionary with 'treated' and 'control' keys containing pre-weighting covariates
        to show balance improvement after weighting
    figsize : Tuple[int, int], default=(12, 8)
        Figure size
    title : str, default="Covariate Balance"
        Plot title
    """
    if feature_names is None:
        feature_names = [f"X{i+1}" for i in range(X_treated.shape[1])]
    
    # Calculate standardized mean difference
    def calc_smd(x1, x0):
        diff = np.mean(x1, axis=0) - np.mean(x0, axis=0)
        pooled_std = np.sqrt((np.var(x1, axis=0) + np.var(x0, axis=0)) / 2)
        # Handle zero std
        pooled_std = np.where(pooled_std == 0, 1, pooled_std)
        return diff / pooled_std
    
    smd_after = calc_smd(X_treated, X_control)
    
    plt.figure(figsize=figsize)
    
    # If before_weighting data is provided, calculate SMD before weighting
    if before_weighting is not None:
        smd_before = calc_smd(
            before_weighting['treated'], 
            before_weighting['control']
        )
        
        # Create a dataframe for plotting
        smd_df = pd.DataFrame({
            'Feature': feature_names * 2,
            'SMD': np.concatenate([smd_before, smd_after]),
            'Balance': ['Before'] * len(smd_before) + ['After'] * len(smd_after)
        })
        
        # Plot SMD before and after
        sns.barplot(
            data=smd_df, 
            x='SMD', 
            y='Feature',
            hue='Balance',
            palette=['#ff7f0e', '#1f77b4']
        )
        
        # Add reference lines
        plt.axvline(x=0.1, color='red', linestyle='--', alpha=0.5)
        plt.axvline(x=-0.1, color='red', linestyle='--', alpha=0.5)
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    else:
        # Plot SMD after weighting only
        plt.barh(feature_names, smd_after, color='#1f77b4')
        
        # Add reference lines
        plt.axvline(x=0.1, color='red', linestyle='--', alpha=0.5)
        plt.axvline(x=-0.1, color='red', linestyle='--', alpha=0.5)
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    plt.title(title)
    plt.xlabel('Standardized Mean Difference')
    plt.ylabel('Feature')
    plt.tight_layout()
    
    return plt.gcf()

def plot_propensity_scores(
    propensities_treated: np.ndarray,
    propensities_control: np.ndarray,
    figsize: Tuple[int, int] = (10, 6),
    title: str = "Propensity Score Distribution"
):
    """
    Plot propensity score distributions for treated and control groups.
    
    Parameters
    ----------
    propensities_treated : np.ndarray
        Propensity scores for treated units
    propensities_control : np.ndarray
        Propensity scores for control units
    figsize : Tuple[int, int], default=(10, 6)
        Figure size
    title : str, default="Propensity Score Distribution"
        Plot title
    """
    plt.figure(figsize=figsize)
    
    # Create dataframe for easier plotting
    treated_df = pd.DataFrame({
        'Propensity': propensities_treated,
        'Group': 'Treated'
    })
    control_df = pd.DataFrame({
        'Propensity': propensities_control,
        'Group': 'Control'
    })
    df = pd.concat([treated_df, control_df])
    
    # Plot distributions
    sns.histplot(
        data=df,
        x='Propensity',
        hue='Group',
        element='step',
        stat='density',
        common_norm=False,
        alpha=0.5
    )
    
    plt.title(title)
    plt.xlabel('Propensity Score')
    plt.ylabel('Density')
    plt.legend(title='')
    plt.tight_layout()
    
    return plt.gcf()

def plot_treatment_effect_heterogeneity(
    X: np.ndarray,
    treatment_effects: np.ndarray,
    feature_names: Optional[List[str]] = None,
    n_top_features: int = 3,
    figsize: Tuple[int, int] = (15, 10)
):
    """
    Plot treatment effect heterogeneity with respect to important covariates.
    
    Parameters
    ----------
    X : np.ndarray
        Covariate matrix
    treatment_effects : np.ndarray
        Individual treatment effect estimates
    feature_names : List[str], optional
        Names of features
    n_top_features : int, default=3
        Number of top features to plot
    figsize : Tuple[int, int], default=(15, 10)
        Figure size
    """
    if feature_names is None:
        feature_names = [f"X{i+1}" for i in range(X.shape[1])]
    
    # Create dataframe
    df = pd.DataFrame(X, columns=feature_names)
    df['treatment_effect'] = treatment_effects
    
    # Calculate correlation with treatment effect
    correlations = np.array([
        np.abs(np.corrcoef(X[:, i], treatment_effects)[0, 1])
        for i in range(X.shape[1])
    ])
    
    # Get top features
    top_feature_idx = np.argsort(-correlations)[:n_top_features]
    top_features = [feature_names[i] for i in top_feature_idx]
    
    plt.figure(figsize=figsize)
    
    # Create subplot for each top feature
    for i, feature in enumerate(top_features):
        plt.subplot(1, n_top_features, i+1)
        
        # Scatter plot
        sns.regplot(
            data=df,
            x=feature,
            y='treatment_effect',
            scatter_kws={'alpha': 0.5},
            line_kws={'color': 'red'}
        )
        
        plt.title(f"{feature}\nCorrelation: {correlations[top_feature_idx[i]]:.2f}")
        plt.ylabel('Treatment Effect' if i == 0 else '')
    
    plt.tight_layout()
    plt.suptitle('Treatment Effect Heterogeneity', fontsize=16, y=1.05)
    
    return plt.gcf()

def plot_latent_space(
    Z: np.ndarray,
    D: np.ndarray,
    method: str = 'pca',
    title: str = "Latent Space Visualization",
    figsize: Tuple[int, int] = (10, 8),
    random_state: Optional[int] = None
):
    """
    Visualize latent space representations colored by treatment groups.
    
    Parameters
    ----------
    Z : np.ndarray
        Latent representations
    D : np.ndarray
        Treatment indicators
    method : str, default='pca'
        Dimensionality reduction method ('pca' or 'tsne')
    title : str, default="Latent Space Visualization"
        Plot title
    figsize : Tuple[int, int], default=(10, 8)
        Figure size
    random_state : int, optional
        Random seed for reproducibility
    """
    plt.figure(figsize=figsize)
    
    # Apply dimensionality reduction
    if Z.shape[1] <= 2:
        # No reduction needed
        Z_2d = Z[:, :2]
    else:
        if method.lower() == 'tsne':
            model = TSNE(n_components=2, random_state=random_state)
            Z_2d = model.fit_transform(Z)
        else:  # Default to PCA
            model = PCA(n_components=2, random_state=random_state)
            Z_2d = model.fit_transform(Z)
    
    # Create dataframe for plotting
    df = pd.DataFrame({
        'Z1': Z_2d[:, 0],
        'Z2': Z_2d[:, 1] if Z_2d.shape[1] > 1 else np.zeros(Z_2d.shape[0]),
        'Treatment': D.astype(int)
    })
    
    # Plot scatter
    sns.scatterplot(
        data=df,
        x='Z1',
        y='Z2',
        hue='Treatment',
        palette=['#1f77b4', '#ff7f0e'],
        alpha=0.7
    )
    
    plt.title(title)
    plt.xlabel(f"{method.upper()} Component 1")
    plt.ylabel(f"{method.upper()} Component 2")
    plt.legend(title='Treatment')
    plt.tight_layout()
    
    return plt.gcf()

def diagnostic_plots(
    X_treated: np.ndarray,
    X_control: np.ndarray,
    Z_treated: Optional[np.ndarray] = None,
    Z_control: Optional[np.ndarray] = None,
    propensity_treated: Optional[np.ndarray] = None,
    propensity_control: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
    before_weighting: Optional[Dict[str, np.ndarray]] = None,
    figsize: Tuple[int, int] = (18, 12)
):
    """
    Create a comprehensive set of diagnostic plots for causal inference.
    
    Parameters
    ----------
    X_treated : np.ndarray
        Covariates for treated units
    X_control : np.ndarray
        Covariates for control units
    Z_treated : np.ndarray, optional
        Latent representations for treated units
    Z_control : np.ndarray, optional
        Latent representations for control units
    propensity_treated : np.ndarray, optional
        Propensity scores for treated units
    propensity_control : np.ndarray, optional
        Propensity scores for control units
    feature_names : List[str], optional
        Names of features for axis labels
    before_weighting : Dict[str, np.ndarray], optional
        Dictionary with 'treated' and 'control' keys containing pre-weighting covariates
    figsize : Tuple[int, int], default=(18, 12)
        Figure size
    """
    plt.figure(figsize=figsize)
    
    # Define number of plots based on available data
    n_plots = 1  # Start with balance plot
    if propensity_treated is not None and propensity_control is not None:
        n_plots += 1
    if Z_treated is not None and Z_control is not None:
        n_plots += 1
    
    # Covariate balance plot
    plt.subplot(1, n_plots, 1)
    plot_balance = plot_covariate_balance(
        X_treated, 
        X_control, 
        feature_names, 
        before_weighting
    )
    plt.close(plot_balance)
    
    current_plot = 2
    
    # Propensity score plot
    if propensity_treated is not None and propensity_control is not None:
        plt.subplot(1, n_plots, current_plot)
        plot_prop = plot_propensity_scores(propensity_treated, propensity_control)
        plt.close(plot_prop)
        current_plot += 1
    
    # Latent space plot
    if Z_treated is not None and Z_control is not None:
        plt.subplot(1, n_plots, current_plot)
        Z = np.vstack([Z_treated, Z_control])
        D = np.concatenate([np.ones(len(Z_treated)), np.zeros(len(Z_control))])
        plot_latent = plot_latent_space(Z, D, method='pca')
        plt.close(plot_latent)
    
    plt.tight_layout()
    plt.suptitle('Causal Inference Diagnostic Plots', fontsize=16, y=1.05)
    
    return plt.gcf()

def plot_scenario_effects(
    scenarios: Dict[str, np.ndarray],
    effects: Dict[str, np.ndarray],
    figsize: Tuple[int, int] = (12, 8)
):
    """
    Plot treatment effects under different scenarios.
    
    Parameters
    ----------
    scenarios : Dict[str, np.ndarray]
        Dictionary with scenario names as keys and scenario vectors as values
    effects : Dict[str, np.ndarray]
        Dictionary with scenario names as keys and effect estimates as values
    figsize : Tuple[int, int], default=(12, 8)
        Figure size
    """
    # Calculate mean and CI for each scenario
    scenario_names = list(scenarios.keys())
    mean_effects = np.array([effects[s].mean() for s in scenario_names])
    ci_lower = np.array([np.percentile(effects[s], 2.5) for s in scenario_names])
    ci_upper = np.array([np.percentile(effects[s], 97.5) for s in scenario_names])
    
    # Sort by mean effect
    sort_idx = np.argsort(mean_effects)
    scenario_names = [scenario_names[i] for i in sort_idx]
    mean_effects = mean_effects[sort_idx]
    ci_lower = ci_lower[sort_idx]
    ci_upper = ci_upper[sort_idx]
    
    plt.figure(figsize=figsize)
    
    # Plot mean effects with CI error bars
    plt.errorbar(
        mean_effects,
        range(len(scenario_names)),
        xerr=np.vstack([mean_effects - ci_lower, ci_upper - mean_effects]),
        fmt='o',
        capsize=5
    )
    
    plt.yticks(range(len(scenario_names)), scenario_names)
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    plt.title('Treatment Effects Under Different Scenarios')
    plt.xlabel('Average Treatment Effect')
    plt.ylabel('Scenario')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()

def plot_mediation_effects(
    direct_effect: float,
    indirect_effects: Dict[str, float],
    ci_direct: Optional[Tuple[float, float]] = None,
    ci_indirect: Optional[Dict[str, Tuple[float, float]]] = None,
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Plot direct and indirect (mediation) effects.
    
    Parameters
    ----------
    direct_effect : float
        Estimate of direct effect
    indirect_effects : Dict[str, float]
        Dictionary with mediator names as keys and indirect effect estimates as values
    ci_direct : Tuple[float, float], optional
        Confidence interval for direct effect (lower, upper)
    ci_indirect : Dict[str, Tuple[float, float]], optional
        Dictionary with mediator names as keys and CI tuples (lower, upper) as values
    figsize : Tuple[int, int], default=(10, 6)
        Figure size
    """
    plt.figure(figsize=figsize)
    
    # Prepare data for plotting
    effect_names = ['Direct Effect'] + list(indirect_effects.keys())
    effect_values = [direct_effect] + list(indirect_effects.values())
    
    # Calculate total effect
    total_effect = direct_effect + sum(indirect_effects.values())
    effect_names.append('Total Effect')
    effect_values.append(total_effect)
    
    # Calculate error bars if CI is provided
    if ci_direct is not None and ci_indirect is not None:
        ci_all = [ci_direct] + [ci_indirect[name] for name in indirect_effects]
        
        # Calculate CI for total effect
        total_ci_lower = ci_direct[0] + sum(ci[0] for ci in ci_indirect.values())
        total_ci_upper = ci_direct[1] + sum(ci[1] for ci in ci_indirect.values())
        ci_all.append((total_ci_lower, total_ci_upper))
        
        # Calculate error bar heights
        yerr_lower = np.array([val - ci[0] for val, ci in zip(effect_values, ci_all)])
        yerr_upper = np.array([ci[1] - val for val, ci in zip(effect_values, ci_all)])
        yerr = np.vstack([yerr_lower, yerr_upper])
    else:
        yerr = None
    
    # Create color map
    colors = ['#1f77b4'] + ['#ff7f0e'] * len(indirect_effects) + ['#2ca02c']
    
    # Plot the effects
    bars = plt.barh(effect_names, effect_values, color=colors, alpha=0.7)
    
    # Add error bars if CI is provided
    if yerr is not None:
        for i, bar in enumerate(bars):
            plt.errorbar(
                bar.get_width(), bar.get_y() + bar.get_height()/2,
                xerr=[[yerr[0, i]], [yerr[1, i]]],
                fmt='o',
                color='black',
                capsize=5
            )
    
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    plt.title('Direct and Indirect Treatment Effects')
    plt.xlabel('Effect Magnitude')
    plt.tight_layout()
    
    return plt.gcf()

def plot_spatial_effects(
    geo_df,
    effect_col: str,
    plot_type: str = 'choropleth',
    title: str = 'Spatial Distribution of Treatment Effects',
    cmap: str = 'RdBu_r',
    figsize: Tuple[int, int] = (12, 10),
    legend_title: str = 'Effect Size',
    cluster_labels: Optional[np.ndarray] = None,
    time_periods: Optional[List] = None
):
    """
    Plot spatial distribution of treatment effects on a map.
    
    Parameters
    ----------
    geo_df : geopandas.GeoDataFrame
        GeoDataFrame with geometries and effect estimates
    effect_col : str
        Column name in geo_df containing effect estimates
    plot_type : str, default='choropleth'
        Type of plot to create:
        - 'choropleth': Standard choropleth map
        - 'cluster': Spatial cluster map (requires cluster_labels)
        - 'local_moran': Local Moran's I cluster map
        - 'bubble': Bubble map with size proportional to effect
        - 'time_series': Multiple maps for each time period (requires time_periods)
    title : str, default='Spatial Distribution of Treatment Effects'
        Plot title
    cmap : str, default='RdBu_r'
        Colormap name
    figsize : Tuple[int, int], default=(12, 10)
        Figure size
    legend_title : str, default='Effect Size'
        Title for the colorbar legend
    cluster_labels : array-like, optional
        Cluster labels for each geometry (required for 'cluster' plot)
    time_periods : List, optional
        List of time periods to plot (required for 'time_series' plot)
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Spatial effects plot
    """
    # Check for required dependencies
    try:
        import geopandas as gpd
        import matplotlib.pyplot as plt
        import matplotlib.colors as colors
    except ImportError:
        raise ImportError(
            "Spatial visualization requires geopandas and matplotlib. "
            "Install with: pip install geopandas matplotlib"
        )
    
    # Standard choropleth map
    if plot_type == 'choropleth':
        fig, ax = plt.subplots(figsize=figsize)
        
        # Find range for symmetric colormap
        vmax = max(abs(geo_df[effect_col].max()), abs(geo_df[effect_col].min()))
        vmin = -vmax
        
        # Plot the effects
        geo_df.plot(
            column=effect_col,
            cmap=cmap,
            linewidth=0.8,
            ax=ax,
            edgecolor='0.8',
            vmin=vmin,
            vmax=vmax,
            legend=True,
            legend_kwds={'label': legend_title}
        )
        
        # Try to add contextual base map if possible
        try:
            import contextily as ctx
            ctx.add_basemap(
                ax,
                crs=geo_df.crs,
                source=ctx.providers.CartoDB.Positron
            )
        except (ImportError, Exception) as e:
            print(f"Note: Contextily not available. Install with: pip install contextily")
        
        ax.set_title(title, fontsize=14)
        ax.set_axis_off()
        plt.tight_layout()
        
    # Spatial cluster map
    elif plot_type == 'cluster':
        if cluster_labels is None:
            raise ValueError("cluster_labels is required for 'cluster' plot_type")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create a copy with cluster labels
        cluster_map = geo_df.copy()
        cluster_map['cluster'] = cluster_labels
        
        # Create categorical colormap
        n_clusters = len(np.unique(cluster_labels))
        cluster_cmap = plt.cm.get_cmap('tab10', n_clusters)
        
        # Plot clusters
        cluster_map.plot(
            column='cluster',
            categorical=True,
            cmap=cluster_cmap,
            linewidth=0.8,
            ax=ax,
            edgecolor='0.8',
            legend=True,
            legend_kwds={'title': 'Cluster'}
        )
        
        # Try to add contextual base map
        try:
            import contextily as ctx
            ctx.add_basemap(
                ax,
                crs=geo_df.crs,
                source=ctx.providers.CartoDB.Positron
            )
        except (ImportError, Exception) as e:
            print(f"Note: Contextily not available. Install with: pip install contextily")
        
        ax.set_title(title, fontsize=14)
        ax.set_axis_off()
        plt.tight_layout()
        
    # Local Moran's I cluster map
    elif plot_type == 'local_moran':
        try:
            from esda.moran import Moran_Local
            from libpysal.weights import Queen
        except ImportError:
            raise ImportError(
                "Local Moran's I plot requires esda and libpysal packages. "
                "Install with: pip install esda libpysal"
            )
        
        # Create weights matrix
        w = Queen.from_dataframe(geo_df)
        w.transform = 'r'
        
        # Calculate Local Moran's I
        lisa = Moran_Local(geo_df[effect_col].values, w)
        
        # Assign significance and cluster types
        sig = 0.05
        labels = np.zeros(len(geo_df))
        
        # High-High
        labels[(lisa.p_sim < sig) & (lisa.q == 1)] = 1
        # Low-Low
        labels[(lisa.p_sim < sig) & (lisa.q == 2)] = 2
        # High-Low
        labels[(lisa.p_sim < sig) & (lisa.q == 3)] = 3
        # Low-High
        labels[(lisa.p_sim < sig) & (lisa.q == 4)] = 4
        
        # Create choropleth with LISA clusters
        fig, ax = plt.subplots(figsize=figsize)
        
        # Add labels to dataframe
        lisa_map = geo_df.copy()
        lisa_map['lisa_cluster'] = labels
        
        # Set up colors for LISA clusters
        lisa_colors = {
            0: 'white',
            1: 'red',
            2: 'blue',
            3: 'pink',
            4: 'lightblue'
        }
        
        # Create custom colormap
        cmap_lisa = colors.ListedColormap([lisa_colors[i] for i in range(5)])
        categories = ['Not Significant', 'High-High', 'Low-Low', 'High-Low', 'Low-High']
        
        # Plot LISA clusters
        lisa_map.plot(
            column='lisa_cluster',
            categorical=True,
            cmap=cmap_lisa,
            linewidth=0.8,
            edgecolor='0.8',
            ax=ax,
            legend=True,
            legend_kwds={'labels': categories}
        )
        
        ax.set_title('Local Moran\'s I Cluster Map', fontsize=14)
        ax.set_axis_off()
        plt.tight_layout()
        
    # Bubble map
    elif plot_type == 'bubble':
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create centroids for bubble placement
        centroids = geo_df.copy()
        centroids['centroid'] = geo_df.geometry.centroid
        centroids = centroids.set_geometry('centroid')
        
        # Plot base geometries
        geo_df.plot(
            color='lightgrey',
            ax=ax,
            edgecolor='grey',
            linewidth=0.5
        )
        
        # Scale effects for bubble sizes
        min_size = 20
        max_size = 500
        abs_effects = np.abs(centroids[effect_col])
        if abs_effects.max() > abs_effects.min():
            scaled_size = min_size + (abs_effects - abs_effects.min()) / \
                          (abs_effects.max() - abs_effects.min()) * (max_size - min_size)
        else:
            scaled_size = np.ones_like(abs_effects) * min_size
            
        # Create bubble colors based on sign of effect
        bubble_colors = np.where(centroids[effect_col] > 0, 'red', 'blue')
        
        # Plot bubbles
        centroids.plot(
            ax=ax,
            markersize=scaled_size,
            color=bubble_colors,
            alpha=0.7
        )
        
        # Try to add contextual base map
        try:
            import contextily as ctx
            ctx.add_basemap(
                ax,
                crs=geo_df.crs,
                source=ctx.providers.CartoDB.Positron
            )
        except (ImportError, Exception) as e:
            print(f"Note: Contextily not available. Install with: pip install contextily")
            
        # Add legend for positive and negative effects
        try:
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                    markersize=10, label='Positive Effect'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                    markersize=10, label='Negative Effect'),
            ]
            ax.legend(handles=legend_elements, loc='lower right')
        except ImportError:
            pass
        
        ax.set_title(title, fontsize=14)
        ax.set_axis_off()
        plt.tight_layout()
        
    # Time series of maps
    elif plot_type == 'time_series':
        if time_periods is None or len(time_periods) == 0:
            raise ValueError("time_periods is required for 'time_series' plot_type")
        
        n_periods = len(time_periods)
        n_cols = min(3, n_periods)
        n_rows = (n_periods + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], figsize[1] * n_rows / 2))
        
        if n_periods == 1:
            axes = np.array([axes])
        
        # Find global min/max for consistent color scale
        vmax = max(abs(geo_df[f"{effect_col}_{period}"].max()) 
                  for period in time_periods if f"{effect_col}_{period}" in geo_df.columns)
        vmin = -vmax
        
        for i, period in enumerate(time_periods):
            row = i // n_cols
            col = i % n_cols
            
            # Get current axis
            if n_rows > 1 and n_cols > 1:
                ax = axes[row, col]
            elif n_rows > 1 or n_cols > 1:
                ax = axes[i]
            else:
                ax = axes
                
            # Get column name for this period
            period_col = f"{effect_col}_{period}"
            
            # Check if column exists
            if period_col in geo_df.columns:
                # Plot this time period
                geo_df.plot(
                    column=period_col,
                    cmap=cmap,
                    linewidth=0.5,
                    ax=ax,
                    edgecolor='0.8',
                    vmin=vmin,
                    vmax=vmax
                )
                
                ax.set_title(f"Period: {period}", fontsize=12)
                ax.set_axis_off()
            else:
                ax.set_visible(False)
                
        # Add a colorbar to the figure
        try:
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            
            # Find the last visible axis
            last_ax = None
            for i in range(n_rows * n_cols):
                row = i // n_cols
                col = i % n_cols
                
                if n_rows > 1 and n_cols > 1:
                    current_ax = axes[row, col]
                elif n_rows > 1 or n_cols > 1:
                    current_ax = axes[i]
                else:
                    current_ax = axes
                    
                if current_ax.get_visible():
                    last_ax = current_ax
                    
            if last_ax:
                divider = make_axes_locatable(last_ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
                sm._A = []
                cbar = fig.colorbar(sm, cax=cax)
                cbar.set_label(legend_title)
        except ImportError:
            print("Note: mpl_toolkits is required for advanced colorbar placement.")
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
    else:
        raise ValueError(f"Unknown plot_type: {plot_type}")
    
    return fig
