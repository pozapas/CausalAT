# CISD: Causal-Intervention Scenario Design for Active Transportation

This package implements the Causal-Intervention Scenario Design (CISD) framework and AI-augmented causal inference approach for active transportation research. CISD combines traditional causal inference methods with advanced AI techniques to analyze active transportation data and estimate the causal effects of infrastructure interventions while accounting for spatial dependencies and longitudinal trends.

## Overview

CISD treats policy analysis as a two-stage act:
1. Choose an explicit scenario vector that bundles mediating and moderating features
2. Apply a treatment indicator to the population, estimating what would happen if the same individuals experienced different treatments while scenario elements remain pinned to user-specified references

The framework combines traditional causal inference methods with advanced AI techniques to handle high-dimensional data common in transportation research, such as street-view images, GPS traces, and textual data.

## Key Features

- **CISD Framework**: Implementation of the canonical CISD estimand with support for stochastic scenarios
- **Three-Layer Architecture**: Representation learning (Φ), balancing (Ψ), and causal learning (Γ) components
- **Multimodal Data Processing**: Support for street imagery, GPS-accelerometer traces, zoning data, and text
- **Causal Machine Learning**: Doubly robust estimators with semiparametric efficiency
- **Visualization Tools**: Diagnostic plots for balance checking and effect heterogeneity
- **Spatial Analysis**: Tools for handling spatial dependencies and geospatial data in causal inference
- **Longitudinal Data**: Support for panel data analysis with difference-in-differences, fixed effects, synthetic control, and staggered adoption methods

## Installation

### Quick Installation

For development installation (editable mode):
```bash
pip install -e .
```

For users who want to install from GitHub directly:
```bash
pip install git+https://github.com/pozapas/CausalAT.git
```

### Installation with Optional Dependencies

For spatial and longitudinal data analysis features:
```bash
pip install -e ".[spatial]"
```

For visualization capabilities:
```bash
pip install -e ".[viz]"
```

For full functionality including all optional dependencies:
```bash
pip install -e ".[all]"
```

### Step-by-step Installation Guide

1. Clone the repository:
   ```bash
   git clone https://github.com/pozapas/CausalAT.git
   cd CausalAT
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. Install the package with desired dependencies:
   ```bash
   pip install -e ".[all]"
   ```

4. Verify installation:
   ```bash
   python -c "import cisd; print(cisd.__version__)"
   ```

### Dependencies

The package has different levels of dependencies:

* **Core dependencies**: 
  - NumPy, Pandas, SciPy (data handling and computation)
  - scikit-learn (machine learning algorithms)
  - TensorFlow/PyTorch (neural networks for representation learning)
  - NetworkX (basic network analysis)

* **Spatial dependencies**: 
  - GeoPandas (spatial data handling)
  - Shapely (geometric objects)
  - libpysal (spatial weights)
  - ESDA (exploratory spatial data analysis)
  - spreg (spatial regression models)
  - statsmodels (econometric models)

* **Visualization**: 
  - Matplotlib, Seaborn (basic plotting)
  - Contextily (basemaps for spatial visualization)
  - Folium (interactive maps)

* **Network analysis**: 
  - NetworkX (graph operations)
  - OSMnx (OpenStreetMap network data)
  - igraph (optional, for faster network algorithms)

## Usage

See the notebooks directory for tutorials on implementing the CISD framework with different types of transportation data:

- `cisd_framework_tutorial.ipynb`: Introduction to the CISD framework and concepts
- `ai_augmented_causal_inference.ipynb`: Using AI for causal inference in transportation research
- `spatial_temporal_analysis_tutorial.ipynb`: Analyzing spatial-temporal data for transportation infrastructure effects
- `network_analysis_tutorial.ipynb`: Working with transportation network data and infrastructure interventions

## Documentation

- `cisd.core`: Core implementation of the CISD framework
- `cisd.representation`: Neural encoders for multimodal transportation data
- `cisd.balancing`: Balancing methods for covariate distribution matching
- `cisd.causal`: Causal estimators with influence function corrections
- `cisd.ai_pipeline`: End-to-end AI pipelines for causal inference
- `cisd.spatial_temporal`: Classes and utilities for handling spatial and longitudinal data
- `cisd.spatial_neighborhood_generator`: Functions for generating synthetic spatial and network data
- `cisd.visualization`: Visualization tools including spatial effect mapping and diagnostic plots

## Citation

As the paper is not yet published, please cite this GitHub repository if you use this package in your research:
```
@software{causalat_cisd,
  title={CISD: Causal-Intervention Scenario Design for Active Transportation Research},
  author={Rafe, Amir},
  url={https://github.com/pozapas/CausalAT},
  year={2025},
  month={May}
}
```

## Contributing

Contributions to enhance the CISD package are welcome. Please feel free to submit a pull request or open an issue to discuss potential improvements.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT
