<div align="center">

# 🚶‍♀️ CISD: Causal-Intervention Scenario Design 🚴‍♂️

### *Advanced Causal Inference Framework for Active Transportation Research*

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub issues](https://img.shields.io/github/issues/pozapas/CausalAT)](https://github.com/pozapas/CausalAT/issues)
[![GitHub stars](https://img.shields.io/github/stars/pozapas/CausalAT)](https://github.com/pozapas/CausalAT/stargazers)

*Bridging the gap between causality science and active transportation research*

</div>

> **📌 Note:** For the Census Tract Network Analysis Tool, please check the [Census_Tract_Network branch](https://github.com/pozapas/CausalAT/tree/Census_Tract_Network).

</div>

## 📋 Overview

**CISD** is a cutting-edge framework that revolutionizes how we understand cause and effect in active transportation systems. By leveraging both traditional causal inference and advanced AI techniques, CISD enables researchers and policymakers to:

- **Quantify** the true impact of infrastructure interventions
- **Account for** complex spatial dependencies and longitudinal trends
- **Process** multimodal data sources including imagery, GPS traces, and textual data
- **Generate** robust evidence for evidence-based policy decisions

### The CISD Approach

CISD treats policy analysis as a principled two-stage process:

1. **Define Scenario Vector** → Bundle mediating and moderating features into explicit scenarios
2. **Apply Treatment Indicator** → Estimate counterfactual outcomes when individuals experience different treatments while scenario elements remain fixed

## ✨ Key Features

<table>
  <tr>
    <td width="50%">
      <h3>🧠 Advanced Framework</h3>
      <ul>
        <li><b>CISD Estimand</b>: Implementation of the canonical CISD estimand with support for stochastic scenarios</li>
        <li><b>Three-Layer Architecture</b>: Representation learning (Φ), balancing (Ψ), and causal learning (Γ) components</li>
        <li><b>Causal Machine Learning</b>: Doubly robust estimators with semiparametric efficiency guarantees</li>
      </ul>
    </td>
    <td width="50%">
      <h3>🔍 Data Processing</h3>
      <ul>
        <li><b>Multimodal Analysis</b>: Street imagery, GPS-accelerometer traces, zoning data, and textual data</li>
        <li><b>Spatial Analysis</b>: Specialized tools for handling spatial dependencies and geospatial data</li>
        <li><b>Longitudinal Methods</b>: Difference-in-differences, fixed effects, synthetic control, and staggered adoption</li>
      </ul>
    </td>
  </tr>
  <tr>
    <td width="50%">
      <h3>📊 Visualization</h3>
      <ul>
        <li><b>Diagnostic Plots</b>: Balance checking and effect heterogeneity visualization</li>
        <li><b>Interactive Maps</b>: Spatial effect mapping with rich contextual layers</li>
        <li><b>Model Interpretability</b>: Tools to understand complex causal relationships</li>
      </ul>
    </td>
    <td width="50%">
      <h3>🌐 Extensible Design</h3>
      <ul>
        <li><b>Modular Components</b>: Easily extend or replace individual components</li>
        <li><b>Integration Support</b>: Works with popular ML frameworks and GIS systems</li>
        <li><b>Research Ready</b>: Designed to facilitate reproducible transportation research</li>
      </ul>
    </td>
  </tr>
</table>

## 🚀 Installation

### Choose your installation path

</div>

### 💨 Quick Installation

```bash
# Development mode (editable)
pip install -e .

# Direct installation from GitHub
pip install git+https://github.com/pozapas/CausalAT.git
```

### 🧩 Installation with Optional Dependencies

```bash
# Spatial analysis features
pip install -e ".[spatial]"

# Visualization capabilities
pip install -e ".[viz]"

# Full functionality (all dependencies)
pip install -e ".[all]"
```

### 📝 Step-by-Step Guide

<details>
<summary><b>Expand for detailed installation instructions</b></summary>

1. **Clone the repository**
   ```bash
   git clone https://github.com/pozapas/CausalAT.git
   cd CausalAT
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   venv\Scripts\activate   # Windows
   source venv/bin/activate  # Linux/MacOS
   ```

3. **Install with desired dependencies**
   ```bash
   pip install -e ".[all]"
   ```

4. **Verify installation**
   ```bash
   python -c "import cisd; print(cisd.__version__)"
   ```
</details>

### 📦 Dependencies

<details>
<summary><b>Core dependencies</b> - Essential libraries for basic functionality</summary>

- **Data Handling:** NumPy, Pandas, SciPy
- **Machine Learning:** scikit-learn
- **Deep Learning:** TensorFlow/PyTorch
- **Network Analysis:** NetworkX

</details>

<details>
<summary><b>Spatial dependencies</b> - For geospatial analysis</summary>

- **Spatial Data:** GeoPandas, Shapely
- **Spatial Statistics:** libpysal, ESDA, spreg
- **Econometric Models:** statsmodels

</details>

<details>
<summary><b>Visualization dependencies</b> - For data visualization</summary>

- **Basic Plotting:** Matplotlib, Seaborn
- **Geospatial Maps:** Contextily, Folium
- **Interactive Charts:** Plotly (for interactive dashboards)

</details>

<details>
<summary><b>Network analysis dependencies</b> - For transportation network modeling</summary>

- **Graph Operations:** NetworkX
- **OSM Integration:** OSMnx (OpenStreetMap data)
- **Advanced Algorithms:** igraph (optional, performance-optimized)

</details>

## 💡 Usage

<div align="center">

### Interactive Tutorials & Examples

Explore our comprehensive tutorial notebooks to get started with CISD.

</div>

| Tutorial | Description | Topics |
|---------|-------------|--------|
| [📘 CISD Framework](notebooks/cisd_framework_tutorial.ipynb) | Introduction to the CISD framework and concepts | Core concepts, estimands, scenario design |
| [🤖 AI-Augmented Inference](notebooks/ai_augmented_causal_inference.ipynb) | Using AI for causal inference in transportation research | Neural networks, representation learning, multimodal data |
| [🗺️ Spatial-Temporal Analysis](notebooks/spatial_temporal_analysis_tutorial.ipynb) | Analyzing spatial-temporal effects of infrastructure interventions | GIS integration, spatial autocorrelation, panel data models |
| [🔀 Network Analysis](notebooks/network_analysis_tutorial.ipynb) | Working with transportation network data | Graph theory, network metrics, flow modeling |

> **Quick Start:** Begin with the CISD Framework tutorial to understand the fundamental concepts before diving into specialized application areas.

## 📚 Documentation

<div align="center">

### Package Structure & Module Reference

</div>

```
cisd/
├── core                      # Core implementation of CISD framework
├── representation            # Neural encoders for multimodal data
├── balancing                 # Covariate distribution matching methods
├── causal                    # Causal estimators with efficiency guarantees
├── ai_pipeline               # End-to-end AI workflows for causal inference
├── spatial_temporal          # Spatial and longitudinal data utilities
├── spatial_neighborhood_generator  # Synthetic spatial data generation
└── visualization             # Diagnostic and effect visualization tools
```

<details>
<summary><b>Module Details</b></summary>

- **`cisd.core`**: Framework fundamentals, estimand definitions, scenario modeling
- **`cisd.representation`**: Feature embedding for images, GPS traces, text using neural networks
- **`cisd.balancing`**: Propensity modeling, entropy balancing, distribution matching algorithms
- **`cisd.causal`**: Doubly-robust estimators, influence functions, sensitivity analysis
- **`cisd.ai_pipeline`**: End-to-end workflows connecting all components
- **`cisd.spatial_temporal`**: Spatial weights, autocorrelation tests, panel data models
- **`cisd.spatial_neighborhood_generator`**: Synthetic data for testing and benchmarking
- **`cisd.visualization`**: Interactive plots, spatial effect maps, balance diagnostics

</details>

## 📄 Citation

If you use CISD in your research, please cite our work:

```bibtex
@software{causalat_cisd,
  title     = {CISD: Causal-Intervention Scenario Design for Active Transportation Research},
  author    = {Rafe, Amir},
  url       = {https://github.com/pozapas/CausalAT},
  year      = {2025},
  month     = {May},
  publisher = {GitHub},
  version   = {1.0.0}
}
```

> **Note:** A formal paper describing the methodology is forthcoming. This citation will be updated when published.

## 👥 Contributing
  
We welcome contributions from researchers, practitioners, and developers!

</div>

**Ways to contribute:**
- 🐛 Report bugs and issues
- 💡 Suggest new features or enhancements
- 🧪 Add test cases
- 📝 Improve documentation
- 🔧 Submit pull requests

### Contribution Workflow

1. **Fork** the repository
2. **Clone** your fork: `git clone https://github.com/YOUR-USERNAME/CausalAT.git`
3. **Create** a feature branch: `git checkout -b feature/amazing-feature`
4. **Develop** your contribution
5. **Commit** your changes: `git commit -m 'Add some amazing feature'`
6. **Push** to your branch: `git push origin feature/amazing-feature`
7. **Submit** a Pull Request

We strive to maintain high-quality, well-documented code that follows best practices for scientific computing.

## 📜 License
  
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
  
</div>

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---
  <p>
    <i>Transforming transportation policy analysis with causal science</i>
  </p>
</div>
