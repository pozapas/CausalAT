"""
Generate synthetic neighborhood-level spatial and temporal datasets for active transportation research.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union

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
                min_pair = None
                
                for node1 in component:
                    for node2 in largest_cc:
                        dist = np.sqrt((pos[node1][0] - pos[node2][0])**2 + 
                                      (pos[node1][1] - pos[node2][1])**2)
                        if dist < min_dist:
                            min_dist = dist
                            min_pair = (node1, node2)
                
                # Add edge between the closest nodes
                G.add_edge(min_pair[0], min_pair[1])
    
    # Generate node attributes
    node_attributes = {}
    for i in range(n_nodes):
        # Generate node-level attributes
        node_attributes[i] = {
            'population_density': np.random.gamma(2, 2),  # People per area
            'intersection_type': np.random.choice(['signalized', 'unsignalized', 'roundabout'], 
                                                p=[0.3, 0.6, 0.1]),
            'n_lanes': np.random.choice([1, 2, 3, 4], p=[0.1, 0.4, 0.3, 0.2]),
        }
    
    # Add node attributes to graph
    nx.set_node_attributes(G, node_attributes)
    
    # Generate edge attributes and select treatment edges
    edge_attributes = {}
    n_edges = G.number_of_edges()
    n_treated_edges = int(treatment_share * n_edges)
    
    # Generate edge indices and shuffle them
    edge_indices = list(range(n_edges))
    np.random.shuffle(edge_indices)
    
    # Select treatment edges
    treated_edges = edge_indices[:n_treated_edges]
    
    # Convert to list of edges
    edges_list = list(G.edges())
    treatment_edges = [edges_list[i] for i in treated_edges]
    
    # Set edge attributes
    for i, (u, v) in enumerate(G.edges()):
        # Distance between nodes
        dist = np.sqrt((pos[u][0] - pos[v][0])**2 + (pos[u][1] - pos[v][1])**2)
        
        # Is this a treated edge?
        is_treated = i in treated_edges
        
        # Edge attributes
        edge_attributes[(u, v)] = {
            'length': dist * 1000,  # Convert to meters
            'speed_limit': np.random.choice([20, 30, 40, 50], p=[0.2, 0.4, 0.3, 0.1]),
            'road_type': np.random.choice(['residential', 'primary', 'secondary', 'tertiary'],
                                         p=[0.4, 0.2, 0.2, 0.2]),
            'bike_lane': 1 if is_treated else np.random.binomial(1, 0.1),
            'sidewalk': 1 if is_treated else np.random.binomial(1, 0.5),
            'treatment': 1 if is_treated else 0,
            'traffic_volume': np.random.gamma(5, 100) * (1 - 0.2 * is_treated)  # Treatment reduces volume
        }
    
    # Add edge attributes to graph
    nx.set_edge_attributes(G, edge_attributes)
    
    # Convert nodes to DataFrame
    nodes_data = []
    for node, data in G.nodes(data=True):
        node_data = {
            'node_id': node,
            'x': pos[node][0],
            'y': pos[node][1]
        }
        node_data.update(data)
        nodes_data.append(node_data)
    
    nodes_df = pd.DataFrame(nodes_data)
    
    # Convert edges to DataFrame
    edges_data = []
    for u, v, data in G.edges(data=True):
        edge_data = {
            'source': u,
            'target': v
        }
        edge_data.update(data)
        edges_data.append(edge_data)
    
    edges_df = pd.DataFrame(edges_data)
    
    return {
        'nodes': nodes_df,
        'edges': edges_df,
        'treatment_edges': treatment_edges
    }

if __name__ == "__main__":
    # Example usage
    neighborhood_data = generate_synthetic_neighborhood_data(
        n_neighborhoods=50,
        n_time_periods=8,
        treatment_period=4
    )
    print(f"Generated neighborhood data shape: {neighborhood_data.shape}")
    
    try:
        network_data = generate_synthetic_infrastructure_network(n_nodes=30)
        print(f"Generated network with {len(network_data['nodes'])} nodes and {len(network_data['edges'])} edges")
    except ImportError:
        print("Networkx not installed, skipping network generation")
