"""
CISD: Causal-Intervention Scenario Design framework for active transportation research.

This package implements the CISD framework and AI-augmented causal inference approaches
described in the paper "Causality in Active Transportation: Exploring Travel Behavior and Well-being".
"""

from .core import CISD
from .representation import RepresentationLearner, StreetviewEncoder, GPSEncoder, ZoningEncoder, TextEncoder, MultiModalEncoder
from .balancing import Balancer, KernelMMD, IPWBalancer
from .causal import CausalLearner, DoublyRobust
from .ai_pipeline import ThreeLayerArchitecture, ActiveBERTDML

# Import visualization utilities
from .visualization import (
    plot_covariate_balance,
    plot_propensity_scores,
    plot_treatment_effect_heterogeneity,
    plot_latent_space,
    diagnostic_plots,
    plot_scenario_effects,
    plot_mediation_effects,
    plot_spatial_effects
)

# Import dataset utilities
from .datasets import (
    generate_synthetic_active_transportation_data,
    generate_street_image_dataset,
    generate_gps_mobility_data,
    save_datasets
)

# Import spatial dataset utilities
from .spatial_neighborhood_generator import (
    generate_synthetic_neighborhood_data,
    generate_synthetic_infrastructure_network
)

# Import spatial and temporal utilities
try:
    from .spatial_temporal import (
        SpatialDependencyHandler,
        LongitudinalDataHandler,
        create_spatial_panel_data
    )
except ImportError:
    import warnings
    warnings.warn(
        "Some spatial-temporal dependencies could not be imported. "
        "Install required packages with: pip install geopandas libpysal esda spreg statsmodels scikit-learn"
    )

__version__ = '0.1.0'
