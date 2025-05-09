"""
Synthetic dataset generators for active transportation research.
This module provides functions to generate synthetic data for illustrating
and testing the CISD framework and AI-augmented causal inference methods.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from sklearn.model_selection import train_test_split

def generate_synthetic_active_transportation_data(
    n_samples: int = 1000,
    seed: Optional[int] = None,
    treatment_effect: float = 2.0,
    confounding_strength: float = 1.0,
    mediator_effect: float = 1.5,
    add_noise: bool = True
) -> Dict[str, np.ndarray]:
    """
    Generate synthetic data for active transportation research.
    
    The data generation process follows:
    1. Generate covariates X (socioeconomics, demographics, built environment)
    2. Generate treatment D (infrastructure intervention) with confounding
    3. Generate mediators M (mode choice, physical activity) affected by treatment
    4. Generate outcome Y (well-being) affected by treatment and mediators
    
    Parameters
    ----------
    n_samples : int, default=1000
        Number of samples to generate
    seed : int, optional
        Random seed for reproducibility
    treatment_effect : float, default=2.0
        Direct causal effect of treatment on outcome
    confounding_strength : float, default=1.0
        Strength of confounding bias (0 = no confounding)
    mediator_effect : float, default=1.5
        Effect of mediators on outcome
    add_noise : bool, default=True
        Whether to add random noise to variables
        
    Returns
    -------
    data : Dict[str, np.ndarray]
        Dictionary with keys 'X', 'D', 'M', 'Y', 'Y_0', 'Y_1' containing:
        - X: Covariates (socioeconomics, demographics, built environment)
        - D: Treatment indicator (infrastructure intervention)
        - M: Mediator variables (mode choice, physical activity, etc.)
        - Y: Observed outcome (well-being)
        - Y_0: Potential outcome under control
        - Y_1: Potential outcome under treatment
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate covariates X (5 features)
    # X1: Socioeconomic status (-2 to 2)
    # X2: Distance to work/school (0 to 10 miles)
    # X3: Age (standardized)
    # X4: Gender (binary)
    # X5: Neighborhood walkability score (0 to 10)
    X1 = np.random.normal(0, 1, n_samples)
    X2 = np.random.gamma(2, 2, n_samples)  # Skewed distance distribution
    X3 = np.random.normal(0, 1, n_samples)
    X4 = np.random.binomial(1, 0.5, n_samples)
    X5 = np.random.beta(2, 2, n_samples) * 10  # Walkability follows beta distribution
    
    X = np.column_stack([X1, X2, X3, X4, X5])
    
    # Generate treatment D with confounding from X
    # Higher SES (X1), shorter distances (X2), and higher walkability (X5) 
    # make treatment more likely
    propensity = 1 / (1 + np.exp(-(0.5*X1 - 0.3*X2 + 0.1*X3 + 0.2*X5)))
    D = np.random.binomial(1, propensity)
    
    # Generate mediators M (3 mediators)
    # M1: Travel mode choice (probability of active transportation)
    # M2: Physical activity level
    # M3: Perceived safety
    
    # Active travel more likely with treatment, shorter distance, higher walkability
    M1_prob = 1 / (1 + np.exp(-(0.3 + 1.0*D - 0.5*X2 + 0.4*X5)))
    M1 = np.random.binomial(1, M1_prob)
    
    # Physical activity increases with active travel and treatment
    M2 = 2 + 1.5*M1 + 0.8*D + 0.2*X5
    if add_noise:
        M2 += np.random.normal(0, 0.5, n_samples)
    
    # Perceived safety increases with treatment and walkability
    M3 = 3 + 1.2*D + 0.3*X5 - 0.2*X2
    if add_noise:
        M3 += np.random.normal(0, 0.5, n_samples)
    
    M = np.column_stack([M1, M2, M3])
    
    # Generate potential outcomes
    # Baseline well-being affected by SES and age
    baseline = 5 + 0.5*X1 + 0.1*X3
    
    # Treatment effect (direct plus through mediators)
    # Confounding bias term: X affects both D and Y
    confounding = confounding_strength * (0.3*X1 - 0.1*X2 + 0.2*X5)
    
    # Potential outcomes
    Y_0 = baseline + confounding
    Y_1 = baseline + treatment_effect + confounding
    
    # Add mediator effects to both potential outcomes
    # Note: M is different under D=0 vs D=1, but we only observe one
    M_effect = mediator_effect * (0.8*M1 + 0.4*M2/10 + 0.3*M3/10)
    Y_0 += M_effect
    Y_1 += M_effect
    
    # Add noise to potential outcomes
    if add_noise:
        noise = np.random.normal(0, 1, n_samples)
        Y_0 += noise
        Y_1 += noise
    
    # Observed outcome based on treatment assignment
    Y = np.where(D == 1, Y_1, Y_0)
    
    # Return all components
    return {
        'X': X,
        'D': D,
        'M': M,
        'Y': Y,
        'Y_0': Y_0,
        'Y_1': Y_1,
    }

def generate_street_image_dataset(
    n_samples: int = 500,
    seed: Optional[int] = None,
    image_size: Tuple[int, int] = (32, 32),
    n_channels: int = 3
) -> Dict[str, np.ndarray]:
    """
    Generate synthetic street image data for testing representation learning.
    
    This function creates a simplified street image dataset where:
    - Images with bike lanes have more green pixels
    - Images with sidewalks have more gray pixels
    - Treatment (infrastructure improvement) adds distinct features
    
    Parameters
    ----------
    n_samples : int, default=500
        Number of samples to generate
    seed : int, optional
        Random seed for reproducibility
    image_size : tuple, default=(32, 32)
        Size of generated images (height, width)
    n_channels : int, default=3
        Number of channels in images (RGB=3)
        
    Returns
    -------
    data : Dict[str, np.ndarray]
        Dictionary with keys 'images', 'D', 'Y', 'features'
        - images: Array of synthetic street images (n_samples, height, width, channels)
        - D: Treatment indicator (infrastructure intervention)
        - Y: Outcome variable (walkability score)
        - features: Underlying features used to generate images
    """
    if seed is not None:
        np.random.seed(seed)
    
    height, width = image_size
    
    # Generate base features
    features = np.random.normal(0, 1, (n_samples, 5))
    
    # Treatment assignment (infrastructure improvement)
    propensity = 1 / (1 + np.exp(-(features[:, 0] + features[:, 2])))
    D = np.random.binomial(1, propensity)
    
    # Create blank images
    images = np.zeros((n_samples, height, width, n_channels))
    
    # Fill images based on features and treatment
    for i in range(n_samples):
        # Base image: roads (dark gray)
        images[i, :, :, :] = 0.2 + np.random.normal(0, 0.05, (height, width, n_channels))
        
        # Add sidewalks (light gray) for some locations
        has_sidewalk = features[i, 1] > 0 or D[i] == 1
        if has_sidewalk:
            # Add sidewalks on the sides
            sidewalk_width = max(2, int(width * 0.15))
            intensity = 0.6 + 0.2 * (D[i])  # Better sidewalks if treated
            
            # Left sidewalk
            images[i, :, :sidewalk_width, :] = intensity + np.random.normal(0, 0.05, 
                                                                (height, sidewalk_width, n_channels))
            
            # Right sidewalk
            images[i, :, -sidewalk_width:, :] = intensity + np.random.normal(0, 0.05, 
                                                                (height, sidewalk_width, n_channels))
        
        # Add bike lanes (green tint) for some locations
        has_bike_lane = features[i, 2] > 0.5 or D[i] == 1
        if has_bike_lane:
            # Add bike lanes on both sides
            bike_lane_width = max(1, int(width * 0.1))
            intensity = 0.1 + 0.3 * (D[i])  # Better bike lanes if treated
            
            # Left bike lane
            images[i, :, sidewalk_width:sidewalk_width+bike_lane_width, 1] += intensity
            
            # Right bike lane
            images[i, :, -(sidewalk_width+bike_lane_width):-sidewalk_width, 1] += intensity
        
        # Add trees (green blobs) more likely in treated areas
        if D[i] == 1 or features[i, 3] > 0.5:
            n_trees = np.random.poisson(2 + 3 * D[i])
            for _ in range(n_trees):
                tree_x = np.random.randint(0, width)
                tree_y = np.random.randint(0, height)
                tree_radius = np.random.randint(2, 5)
                
                # Create green blob
                for y in range(max(0, tree_y - tree_radius), min(height, tree_y + tree_radius)):
                    for x in range(max(0, tree_x - tree_radius), min(width, tree_x + tree_radius)):
                        if ((x - tree_x) ** 2 + (y - tree_y) ** 2) <= tree_radius ** 2:
                            images[i, y, x, 0] = 0.2  # Red channel (low)
                            images[i, y, x, 1] = 0.8  # Green channel (high)
                            images[i, y, x, 2] = 0.2  # Blue channel (low)
        
        # Add crosswalks (white stripes) more likely in treated areas
        if D[i] == 1 or features[i, 4] > 0.7:
            # Add crosswalk in the middle
            crosswalk_y = height // 2
            stripe_width = 2
            n_stripes = 5
            
            for s in range(n_stripes):
                start_x = width // (n_stripes + 1) * (s + 1) - stripe_width // 2
                end_x = start_x + stripe_width
                images[i, crosswalk_y-3:crosswalk_y+3, start_x:end_x, :] = 0.9  # White stripes
    
    # Generate outcome (walkability score) based on features and treatment
    Y = 3 + 2 * D + features[:, 1] + features[:, 3] + np.random.normal(0, 0.5, n_samples)
    
    # Ensure pixel values are in valid range [0, 1]
    images = np.clip(images, 0, 1)
    
    return {
        'images': images,
        'D': D,
        'Y': Y,
        'features': features
    }

def generate_gps_mobility_data(
    n_users: int = 100,
    days_per_user: int = 7,
    seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Generate synthetic GPS mobility data for active transportation research.
    
    Simulates GPS traces showing travel patterns before and after an intervention.
    
    Parameters
    ----------
    n_users : int, default=100
        Number of users in the dataset
    days_per_user : int, default=7
        Number of days of data per user
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    data : Dict[str, np.ndarray]
        Dictionary containing:
        - user_id: User identifiers
        - day: Day index (0 to days_per_user-1)
        - mode: Transportation mode (0=car, 1=walk, 2=bike, 3=public)
        - distance: Distance traveled in km
        - duration: Travel duration in minutes
        - speed: Average speed in km/h
        - calories: Estimated calories burned
        - D: Treatment indicator (neighborhood infrastructure improvement)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Constants and parameters
    modes = [0, 1, 2, 3]  # car, walk, bike, public transport
    mode_names = ['car', 'walk', 'bike', 'public']
    
    # Average distance parameters by mode (in km)
    distance_means = {0: 8.0, 1: 1.5, 2: 5.0, 3: 10.0}
    distance_stds = {0: 3.0, 1: 0.7, 2: 2.0, 3: 4.0}
    
    # Average speed parameters by mode (in km/h)
    speed_means = {0: 35.0, 1: 4.5, 2: 15.0, 3: 20.0}
    speed_stds = {0: 7.0, 1: 1.0, 2: 3.0, 3: 5.0}
    
    # Calories burned per minute by mode (rough approximation)
    calorie_rates = {0: 2.0, 1: 5.0, 2: 8.0, 3: 2.5}
    
    # Generate user properties (fixed across days)
    n_total_records = n_users * days_per_user
    user_ids = np.repeat(np.arange(n_users), days_per_user)
    days = np.tile(np.arange(days_per_user), n_users)
    
    # User propensities toward active travel (individual fixed effect)
    user_active_propensity = np.random.normal(0, 1, n_users)
    
    # Assign treatment to some users (neighborhood improvement)
    user_treatments = np.random.binomial(1, 0.5, n_users)
    
    # Expand user properties to all records
    treatments = np.repeat(user_treatments, days_per_user)
    active_propensities = np.repeat(user_active_propensity, days_per_user)
    
    # Generate mode choices
    # Higher likelihood of active modes (1=walk, 2=bike) with treatment
    # Base probabilities (car, walk, bike, public)
    base_probs = np.array([0.6, 0.15, 0.1, 0.15])
    
    # Adjust probabilities based on treatment and user propensity
    modes = []
    for i in range(n_total_records):
        user_propensity = active_propensities[i]
        treatment = treatments[i]
        day_of_week = days[i] % 7  # 0-6 (Mon-Sun)
        
        # Adjust for weekends
        is_weekend = day_of_week >= 5
        
        # Treatment effect: increases active transportation probability
        treatment_effect = 0.2 * treatment
        
        # Propensity effect: individual preference for active transportation
        propensity_effect = 0.1 * user_propensity
        
        # Weekend effect: more leisure trips, more active transportation
        weekend_effect = 0.1 if is_weekend else 0
        
        # Compute adjusted probabilities
        p = base_probs.copy()
        
        # Reduce car probability, increase active modes
        p[0] -= (treatment_effect + propensity_effect + weekend_effect)
        p[1] += (treatment_effect + propensity_effect + weekend_effect) * 0.5  # Half to walking
        p[2] += (treatment_effect + propensity_effect + weekend_effect) * 0.5  # Half to biking
        
        # Ensure probabilities are valid
        p = np.maximum(p, 0.05)  # Ensure minimum probability
        p = p / p.sum()  # Normalize to sum to 1
        
        # Choose mode
        mode = np.random.choice(len(p), p=p)
        modes.append(mode)
    
    modes = np.array(modes)
    
    # Generate trip features based on mode
    distances = np.zeros(n_total_records)
    speeds = np.zeros(n_total_records)
    
    for mode_id in [0, 1, 2, 3]:
        mask = modes == mode_id
        n_mask = mask.sum()
        
        if n_mask > 0:
            # Generate distances for this mode
            distances[mask] = np.maximum(0.1, np.random.normal(
                distance_means[mode_id], distance_stds[mode_id], n_mask
            ))
            
            # Generate speeds for this mode
            speeds[mask] = np.maximum(0.1, np.random.normal(
                speed_means[mode_id], speed_stds[mode_id], n_mask
            ))
    
    # Calculate duration (hours)
    durations = distances / speeds
    
    # Convert to minutes
    durations = durations * 60
    
    # Calculate calories burned
    calories = np.zeros(n_total_records)
    for mode_id in [0, 1, 2, 3]:
        mask = modes == mode_id
        calories[mask] = durations[mask] * calorie_rates[mode_id]
    
    # Create structured output
    data = {
        'user_id': user_ids,
        'day': days,
        'mode': modes,
        'distance': distances,
        'duration': durations,
        'speed': speeds,
        'calories': calories,
        'D': treatments,
        'mode_name': [mode_names[m] for m in modes]
    }
    
    return data

def generate_synthetic_neighborhood_data(
    n_neighborhoods: int = 100,
    n_time_periods: int = 10,
    treatment_period: int = 5,
    treatment_share: float = 0.3,
    spatial_correlation: float = 0.4,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate synthetic neighborhood-level panel data for active transportation.
    
    Creates a balanced panel dataset with neighborhoods observed over multiple time periods,
    with some neighborhoods receiving an active transportation infrastructure treatment.
    Incorporates spatial correlation between neighborhoods and temporal trends.
    
    Parameters
    ----------
    n_neighborhoods : int, default=100
        Number of neighborhoods in the dataset
    n_time_periods : int, default=10
        Number of time periods
    treatment_period : int, default=5
        Time period when treatment begins (0-indexed)
    treatment_share : float, default=0.3
        Share of neighborhoods receiving treatment
    spatial_correlation : float, default=0.4
        Strength of spatial correlation between neighborhoods (0-1)
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    df : pandas.DataFrame
        DataFrame containing:
        - neighborhood_id: Neighborhood identifier
        - period: Time period (0 to n_time_periods-1)
        - X1-X5: Neighborhood characteristics
        - treatment: Treatment indicator
        - active_transportation_rate: Outcome variable
        - x_coord, y_coord: Spatial coordinates
    """
    if seed is not None:
        np.random.seed(seed)
        
    # Generate neighborhood characteristics (time-invariant)
    X1 = np.random.normal(0, 1, n_neighborhoods)  # Socioeconomic status
    X2 = np.random.gamma(2, 1.5, n_neighborhoods)  # Population density
    X3 = np.random.beta(2, 5, n_neighborhoods) * 10  # Distance to downtown
    X4 = np.random.normal(5, 2, n_neighborhoods)  # Baseline walkability
    X5 = np.random.normal(0, 1, n_neighborhoods)  # Climate favorability
    
    # Generate spatial coordinates for neighborhoods
    # Creating a rough grid layout with some randomness
    grid_size = int(np.ceil(np.sqrt(n_neighborhoods)))
    x_coords = np.zeros(n_neighborhoods)
    y_coords = np.zeros(n_neighborhoods)
    
    for i in range(n_neighborhoods):
        grid_x = i % grid_size
        grid_y = i // grid_size
        # Add some random jitter to avoid perfect grid
        x_coords[i] = grid_x + np.random.uniform(-0.3, 0.3)
        y_coords[i] = grid_y + np.random.uniform(-0.3, 0.3)
    
    # Normalize coordinates to 0-1 range
    x_coords = (x_coords - x_coords.min()) / (x_coords.max() - x_coords.min())
    y_coords = (y_coords - y_coords.min()) / (y_coords.max() - y_coords.min())
    
    # Assign treatment status to neighborhoods (some neighborhoods get treatment)
    # Treatment is more likely in areas with higher socioeconomic status and walkability
    treatment_propensity = 1 / (1 + np.exp(-(0.5*X1 + 0.5*X4)))
    neighborhood_treated = np.random.binomial(1, treatment_propensity)
    
    # Ensure we have the correct proportion of treated neighborhoods
    n_treated_target = int(treatment_share * n_neighborhoods)
    if np.sum(neighborhood_treated) != n_treated_target:
        # Adjust by randomly selecting neighborhoods to change
        current_treated = np.sum(neighborhood_treated)
        if current_treated > n_treated_target:
            # Remove treatment from some neighborhoods
            excess = current_treated - n_treated_target
            treated_indices = np.where(neighborhood_treated == 1)[0]
            to_remove = np.random.choice(treated_indices, size=excess, replace=False)
            neighborhood_treated[to_remove] = 0
        else:
            # Add treatment to some neighborhoods
            shortfall = n_treated_target - current_treated
            control_indices = np.where(neighborhood_treated == 0)[0]
            to_add = np.random.choice(control_indices, size=shortfall, replace=False)
            neighborhood_treated[to_add] = 1
    
    # Generate a spatial weights matrix based on distance between neighborhoods
    distance_matrix = np.zeros((n_neighborhoods, n_neighborhoods))
    for i in range(n_neighborhoods):
        for j in range(n_neighborhoods):
            if i != j:
                # Euclidean distance between neighborhoods
                dist = np.sqrt((x_coords[i] - x_coords[j])**2 + (y_coords[i] - y_coords[j])**2)
                # Convert to similarity (closer = more similar)
                distance_matrix[i, j] = np.exp(-5 * dist)
    
    # Row-normalize the weights matrix
    row_sums = distance_matrix.sum(axis=1)
    distance_matrix = distance_matrix / row_sums[:, np.newaxis]
    
    # Create neighborhood-specific time trends and random effects
    time_trends = np.random.normal(0.1, 0.05, n_neighborhoods)  # Trend slopes
    neighborhood_effects = np.random.normal(0, 1, n_neighborhoods)  # Random effects
    
    # Empty array to store active transportation rates
    active_transport_rates = np.zeros((n_neighborhoods, n_time_periods))
    
    # Generate outcomes for each time period with spatial correlation
    for t in range(n_time_periods):
        # Start with baseline rates depending on neighborhood characteristics
        baseline = 15 + 1.0*X1 - 0.5*X2 - 0.8*X3 + 2.0*X4 + 1.0*X5 + neighborhood_effects
        
        # Add time trend
        baseline += t * time_trends
        
        # Add seasonal effect (sinusoidal pattern with period 4)
        seasonal = 2 * np.sin(t * np.pi / 2)
        baseline += seasonal
        
        # Treatment effect (only after treatment period for treated neighborhoods)
        treatment_indicator = (t >= treatment_period) * neighborhood_treated
        treatment_effect = 5 * treatment_indicator * (1 - np.exp(-0.5 * (t - treatment_period + 1)))
        
        # Add treatment effect to baseline
        rates = baseline + treatment_effect
        
        # Add spatially correlated noise
        if t > 0:
            # Use previous period's rates for spatial correlation
            spatial_component = spatial_correlation * (distance_matrix @ active_transport_rates[:, t-1])
            rates += spatial_component
        
        # Add random noise
        rates += np.random.normal(0, 1, n_neighborhoods)
        
        # Store rates
        active_transport_rates[:, t] = rates
    
    # Create panel dataset
    neighborhoods = np.repeat(np.arange(n_neighborhoods), n_time_periods)
    periods = np.tile(np.arange(n_time_periods), n_neighborhoods)
    
    # Expand neighborhood characteristics to panel
    X1_panel = np.repeat(X1, n_time_periods)
    X2_panel = np.repeat(X2, n_time_periods)
    X3_panel = np.repeat(X3, n_time_periods)
    X4_panel = np.repeat(X4, n_time_periods)
    X5_panel = np.repeat(X5, n_time_periods)
    
    # Expand spatial coordinates to panel
    x_coords_panel = np.repeat(x_coords, n_time_periods)
    y_coords_panel = np.repeat(y_coords, n_time_periods)
    
    # Create treatment indicators for the panel
    treatment_panel = np.zeros(n_neighborhoods * n_time_periods)
    for i in range(n_neighborhoods):
        for t in range(n_time_periods):
            idx = i * n_time_periods + t
            if t >= treatment_period and neighborhood_treated[i]:
                treatment_panel[idx] = 1
    
    # Flatten the rates array
    rates_panel = active_transport_rates.flatten('F')  # Column-major flattening
    
    # Create DataFrame
    df = pd.DataFrame({
        'neighborhood_id': neighborhoods,
        'period': periods,
        'X1': X1_panel,  # Socioeconomic status
        'X2': X2_panel,  # Population density
        'X3': X3_panel,  # Distance to downtown
        'X4': X4_panel,  # Baseline walkability
        'X5': X5_panel,  # Climate favorability
        'treatment': treatment_panel,
        'active_transportation_rate': rates_panel,
        'x_coord': x_coords_panel,
        'y_coord': y_coords_panel
    })
    
    return df

def generate_synthetic_infrastructure_network(
    n_nodes: int = 50,
    edge_density: float = 0.15,
    treatment_share: float = 0.3,
    seed: Optional[int] = None
) -> Dict:
    """
    Generate synthetic infrastructure network data for active transportation.
    
    Creates a network representing intersections and road segments,
    with some segments receiving infrastructure improvements.
    
    Parameters
    ----------
    n_nodes : int, default=50
        Number of nodes (intersections) in the network
    edge_density : float, default=0.15
        Density of edges in the network (0-1)
    treatment_share : float, default=0.3
        Share of edges receiving treatment
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    data : Dict
        Dictionary containing:
        - nodes: DataFrame of nodes with coordinates and attributes
        - edges: DataFrame of edges with attributes
        - treatment_edges: List of treated edge indices
    """
    if seed is not None:
        np.random.seed(seed)
    
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("This function requires networkx. Install with: pip install networkx")
    
    # Generate a geometric random graph (nodes connected if close in space)
    # First, generate random points in 2D space
    pos = {i: (np.random.uniform(0, 1), np.random.uniform(0, 1)) for i in range(n_nodes)}
    
    # Create graph with edges between nearby points
    G = nx.random_geometric_graph(n_nodes, np.sqrt(edge_density), pos=pos)
    
    # Ensure the graph is connected
    if not nx.is_connected(G):
        # Find the largest connected component
        largest_cc = max(nx.connected_components(G), key=len)
        
        # Connect other components to the largest one
        components = list(nx.connected_components(G))
        for component in components:
            if component != largest_cc:
                # Find the closest pair of nodes between components
                min_dist = float('inf')