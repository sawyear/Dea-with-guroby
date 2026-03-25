# gurobi-dea

Modular Data Envelopment Analysis (DEA) toolkit powered by [Gurobi](https://www.gurobi.com/).

All models support **undesirable outputs**.

## Models

| Model | Class | Description |
|-------|-------|-------------|
| SBM | `SBM` | Slacks-Based Measure (Tone, 2001) |
| Super-SBM | `SuperSBM` | Super-efficiency SBM (Tone, 2002) |
| EBM | `EBM` | Epsilon-Based Measure (Tone & Tsutsui, 2010) |
| Additive | `Additive` | Additive model (Charnes et al., 1985) |
| Network SBM | `NetworkSBM` | Network DEA up to 3 stages (Tone & Tsutsui, 2009) |
| Network EBM | `NetworkEBM` | Network EBM up to 3 stages |
| Dynamic SBM | `DynamicSBM` | Dynamic DEA with carry-over (Tone & Tsutsui, 2010) |

## Installation

```bash
pip install -e .
```

Requires a valid [Gurobi license](https://www.gurobi.com/downloads/).

## Quick Start

```python
import pandas as pd
from gurobi_dea import SBM

df = pd.DataFrame({
    "dmu": ["A", "B", "C", "D"],
    "x1":  [10, 20, 30, 40],
    "y1":  [5, 10, 15, 12],
    "b1":  [3, 2, 4, 5],
})

model = SBM(
    inputs=["x1"],
    desirable_outputs=["y1"],
    undesirable_outputs=["b1"],
    data=df,
)
results = model.solve(scale="c")  # CRS
print(results[["dmu", "TE"]])
```

## API

Every model follows the same pattern:

```python
model = ModelClass(
    inputs=["x1", "x2"],
    desirable_outputs=["y1"],
    undesirable_outputs=["b1"],
    data=df,               # DataFrame with a 'dmu' column
)
results = model.solve(scale="c")  # "c" = CRS, "v" = VRS
```

`results` is a DataFrame with columns: `dmu`, `TE` (efficiency score), and slack values for each variable.

### Network models

```python
from gurobi_dea import NetworkSBM

model = NetworkSBM(
    inputs=all_input_cols,
    desirable_outputs=all_output_cols,
    undesirable_outputs=all_undesirable_cols,
    data=df,
    stages=[
        {"x": ["x1"], "y": ["y1"], "z": ["z1"]},
        {"x": ["x2"], "y": ["y2"], "z": ["z2"]},
    ],
    links=[{"vars": ["w12"]}],  # stage 1 -> stage 2
)
results = model.solve(scale="c")
```

### Dynamic models

```python
from gurobi_dea import DynamicSBM

model = DynamicSBM(
    inputs=["x1"],
    desirable_outputs=["y1"],
    undesirable_outputs=["b1"],
    carry_overs=["k1"],
    data=panel_df,  # must have 'dmu' and 'year' columns
)
results = model.solve(scale="c")
```

## Project Structure

```
src/gurobi_dea/
├── __init__.py        # Public API
├── base.py            # Abstract base class
├── utils.py           # Affinity matrix, S-correlation
└── models/
    ├── sbm.py         # SBM + Super-SBM
    ├── ebm.py         # EBM
    ├── additive.py    # Additive
    ├── network.py     # Network SBM/EBM
    └── dynamic.py     # Dynamic SBM
```

## Dependencies

- Python >= 3.9
- gurobipy >= 10.0
- numpy >= 1.22
- pandas >= 1.4
- scipy >= 1.8

## License

GPL-3.0
