# Census Tract Network Analysis Tool

This project creates a stylized network representation of census tracts using OpenStreetMap statistics for any US city. Salt Lake City, Utah is provided as an example implementation, but the code can be easily adapted to analyze other cities. It's part of the [CausalAT](https://github.com/pozapas/CausalAT) repository focused on causal analysis of active transportation.

[![GitHub Repository](https://img.shields.io/badge/GitHub-CausalAT-blue?logo=github)](https://github.com/pozapas/CausalAT/tree/slc_network)

## Project Overview

This project generates a network representation of a city where:
- Nodes represent census tracts (approximately 200)
- Edges represent adjacency between census tracts
- Node attributes include OpenStreetMap statistics (road lengths, types of roads, etc.)

The resulting network visualization shows census tract connectivity with node sizes representing road network density.

## Requirements

This project requires the following Python libraries:
- geopandas
- osmnx
- networkx
- matplotlib
- contextily
- cenpy
- pandas
- numpy
- shapely

You can install them using pip:
```
pip install geopandas osmnx networkx matplotlib contextily cenpy pandas numpy shapely
```

## Usage

Open and run the Jupyter notebook `salt_lake_city_network.ipynb`. The notebook contains detailed steps for:

1. Downloading census tract data for Salt Lake City
2. Fetching OpenStreetMap data for the road network
3. Creating a network representation of census tracts
4. Calculating OpenStreetMap statistics for each tract
5. Visualizing the network
6. Exporting the network for further analysis

## Outputs

The notebook generates the following outputs:
- `salt_lake_city_network.png` - Basic visualization of the network
- `salt_lake_city_network_with_attributes.png` - Visualization with node sizes representing road lengths
- `salt_lake_city_tract_network.graphml` - Network in GraphML format for use in other tools
- `slc_tract_nodes.csv` - Node data including OSM statistics
- `slc_tract_edges.csv` - Edge data representing tract adjacency

## Using with Other Cities

This project is designed to be easily adaptable to any US city:

1. **Change the Census parameters**:
   - Modify the `state_fips` and `county_fips` variables for your target location
   - You can find FIPS codes for US states and counties at the [Census Bureau website](https://www.census.gov/library/reference/code-lists/ansi.html)

2. **Change the OpenStreetMap location**:
   - Modify the `place_name` variable with your target city name
   - Format: 'City Name, State, Country' (e.g., 'Boston, Massachusetts, USA')

All file names and visualizations will automatically use your chosen city name. The methodology remains the same regardless of which US city you analyze.

The visualization and analysis techniques can be further customized based on specific research questions, such as analyzing walkability, accessibility, or infrastructure distribution.

## Repository Structure

- `salt_lake_city_network.ipynb` - Main Jupyter notebook with complete analysis workflow
- `visualize_network.ipynb` - Simplified notebook for loading and visualizing existing network data
- `run_notebook.bat` - Windows batch file to install dependencies and launch the notebook
- `README.md` - This documentation file

## Contributing

Contributions to this project are welcome! To contribute:

1. Fork the repository on GitHub
2. Make your changes and test them
3. Submit a pull request with a clear description of your changes

## Citation

If you use this code in your research, please cite:

```
@misc{pozapas2025slcnetwork,
  author = {Amir Rafe},
  title = {Salt Lake City Census Tract Network},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/pozapas/CausalAT}},
}
```
