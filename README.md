# Hydrogent

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

Computational screening platform for doped metal-oxide nanoparticles for H2 storage. Python tools for descriptor analysis, structure visualization, thermodynamic screening, and interpretable ML.

## Architecture

```
Python Library (3.11+)
  ├── csv_tools     — CDFT descriptors, correlations, shifts
  ├── xyz_tools     — 3D geometry, Mulliken charges
  ├── thermo_tools  — Langmuir isotherm, van't Hoff, T50
  └── ml_tools      — GP, symbolic regression, active learning
```

## Features

- **Interpretable ML** — Symbolic regression (PySR) discovers human-readable formulas; Gaussian Process provides uncertainty-quantified predictions
- **Active learning** — Suggests most informative next experiment via maximum GP uncertainty
- **Thermodynamic screening** — Langmuir isotherm, van't Hoff analysis, T50 desorption midpoint, DOE window compliance
- **CDFT descriptors** — Works with conceptual DFT descriptors (electronegativity, hardness, electrophilicity, etc.)

## Quick Start

```bash
pip install -r src/requirements.txt
```

## Data

The built-in `doped-tio2` project in `data/` contains CDFT descriptors and XYZ geometries for doped TiO2 nanoparticles with and without adsorbed H2.

## ML Capabilities

| Method | Purpose | Best For |
|--------|---------|----------|
| Symbolic Regression | Discover E_ads = f(descriptors) | Interpretable relationships |
| Gaussian Process | Predict E_ads with uncertainty | Small datasets (n < 20) |
| Active Learning | Rank untested dopants | Experiment planning |
| Feature Importance | Identify key descriptors | Understanding what drives adsorption |

## Tech Stack

- **Core**: Python, pandas, NumPy, SciPy
- **ML**: scikit-learn, PySR (optional)
- **Data**: CDFT descriptors from Gaussian 16, XYZ geometries

## Testing

```bash
pytest src/tests/ -v
```

## License

MIT
