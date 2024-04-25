
# Genetic Algorithm Visualization Dashboard

This dashboard visualizes various metrics and results from a genetic algorithm analysis. It allows users to interactively explore data through a web interface built with Streamlit.

## Features
- Load and display genetic algorithm checkpoints.
- Plot fitness over generations, heatmaps of clustered populations, and PCA scatter plots.
- Visualize dendrograms and fitness distributions.
- Interactive selection of checkpoint directories and genetic generations for detailed analysis.

## Prerequisites
- Python 3.x
- pip

## Installation

Clone the repository with GitHub CLI:
```bash
git clone https://github.com/matthewmazurek/ga-dashboard
cd ga-dashboard
```

Create and activate a virtual environment:
- **Windows:**
  ```bash
  python -m venv env
  .\env\Scripts\activate
  ```
- **macOS and Linux:**
  ```bash
  python3 -m venv env
  source env/bin/activate
  ```

Install the required packages:
```bash
pip install -r requirements.txt
```

## Running the Dashboard
To run the dashboard, execute:
```bash
streamlit run app.py [checkpoint_directory]
```
If a checkpoint directory is not provided, then the dashboard will look for checkpoints in the home directory.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
