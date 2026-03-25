# gurobi-dea

Modular Data Envelopment Analysis (DEA) toolkit powered by [Gurobi](https://www.gurobi.com/).

All models support **undesirable outputs**.

## Models

### Static (Cross-Sectional)

| Model | Class | Description | Reference |
|-------|-------|-------------|-----------|
| CCR | `CCR` | CRS radial model | Charnes, Cooper & Rhodes (1978, EJOR) |
| BCC | `BCC` | VRS radial model | Banker, Charnes & Cooper (1984, MS) |
| SBM | `SBM` | Slacks-Based Measure | Tone (2001, EJOR) |
| Super-SBM | `SuperSBM` | Super-efficiency SBM | Tone (2002, EJOR) |
| EBM | `EBM` | Epsilon-Based Measure | Tone & Tsutsui (2010, EJOR) |
| Additive | `Additive` | Additive model | Charnes et al. (1985) |
| DDF | `DDF` | Directional Distance Function | Chambers, Chung & Fare (1996, JET) |

### Structural

| Model | Class | Description | Reference |
|-------|-------|-------------|-----------|
| Network SBM | `NetworkSBM` | Multi-stage network (up to 3) | Tone & Tsutsui (2009, EJOR) |
| Network EBM | `NetworkEBM` | Multi-stage network EBM | Tone & Tsutsui (2009) |
| Dynamic SBM | `DynamicSBM` | Panel with carry-over | Tone & Tsutsui (2010, Omega) |

### Productivity Indices

| Model | Class | Description | Reference |
|-------|-------|-------------|-----------|
| Malmquist | `Malmquist` | TFP = EC × TC | Fare et al. (1994, AER) |
| Malmquist-Luenberger | `MalmquistLuenberger` | Environmental TFP with DDF | Chung, Fare & Grosskopf (1997, JEM) |

## Installation

```bash
pip install -e .
```

Requires a valid [Gurobi license](https://www.gurobi.com/downloads/).

## Quick Start

```python
import pandas as pd
from gurobi_dea import SBM, CCR, Malmquist

# --- Cross-sectional efficiency ---
df = pd.DataFrame({
    "dmu": ["A", "B", "C", "D"],
    "x1":  [10, 20, 30, 40],
    "y1":  [5, 10, 15, 12],
    "b1":  [3, 2, 4, 5],
})

# Radial (CCR)
ccr = CCR(inputs=["x1"], desirable_outputs=["y1"],
          undesirable_outputs=["b1"], data=df)
print(ccr.solve(orientation="input"))

# Non-radial (SBM)
sbm = SBM(inputs=["x1"], desirable_outputs=["y1"],
          undesirable_outputs=["b1"], data=df)
print(sbm.solve(scale="c"))

# --- Productivity index (panel data) ---
panel = pd.DataFrame({
    "dmu":  ["A","A","B","B","C","C"],
    "year": [2020,2021,2020,2021,2020,2021],
    "x1":   [10, 9, 20, 18, 30, 28],
    "y1":   [5, 6, 10, 12, 15, 16],
    "b1":   [3, 2, 2, 1.5, 4, 3],
})

mi = Malmquist(inputs=["x1"], desirable_outputs=["y1"],
               undesirable_outputs=["b1"], data=panel)
print(mi.compute(orientation="input"))
```

## API

Every static model follows the same pattern:

```python
model = ModelClass(
    inputs=["x1", "x2"],
    desirable_outputs=["y1"],
    undesirable_outputs=["b1"],
    data=df,               # DataFrame with a 'dmu' column
)
results = model.solve(scale="c")  # "c" = CRS, "v" = VRS
```

CCR/BCC also accept `orientation="input"` or `"output"`.

DDF accepts `direction="observed"`, `"mean"`, or `"unit"`.

Productivity indices use `model.compute()` on panel data (requires `dmu` + `year` columns).

## Project Structure

```
├── backup/                    # Original legacy code
│   ├── Gurobi_dea.py
│   └── dynamic_dea.py
├── src/gurobi_dea/
│   ├── __init__.py            # Public API (12 classes)
│   ├── base.py                # Abstract base class
│   ├── utils.py               # Affinity matrix, S-correlation
│   └── models/
│       ├── radial.py          # CCR, BCC
│       ├── sbm.py             # SBM, Super-SBM
│       ├── ebm.py             # EBM
│       ├── additive.py        # Additive
│       ├── ddf.py             # Directional Distance Function
│       ├── network.py         # Network SBM/EBM
│       ├── dynamic.py         # Dynamic SBM
│       ├── malmquist.py       # Malmquist TFP Index
│       └── malmquist_luenberger.py  # ML Index
├── examples/
│   └── quickstart.py
├── pyproject.toml
└── LICENSE
```

## Roadmap

- [x] Phase 1: CCR/BCC, DDF, Malmquist, Malmquist-Luenberger
- [ ] Phase 2: Meta-frontier, Dynamic Network SBM, Cross-efficiency
- [ ] Phase 3: Bootstrap DEA, Window DEA, Cost/Revenue efficiency

## Dependencies

- Python >= 3.9
- gurobipy >= 10.0
- numpy >= 1.22
- pandas >= 1.4
- scipy >= 1.8

## License

GPL-3.0
