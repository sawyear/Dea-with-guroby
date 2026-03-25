"""Quick-start example for gurobi-dea.

Generates a small synthetic dataset and runs SBM, Super-SBM, EBM, and Additive.
Requires: gurobipy (with valid license), numpy, pandas, scipy.
"""

import numpy as np
import pandas as pd

from gurobi_dea import SBM, SuperSBM, EBM, Additive

# --- Generate synthetic data ---
np.random.seed(42)
n = 10
df = pd.DataFrame({
    "dmu": [f"DMU{i+1}" for i in range(n)],
    "x1": np.random.uniform(10, 50, n),
    "x2": np.random.uniform(5, 30, n),
    "y1": np.random.uniform(20, 80, n),
    "b1": np.random.uniform(1, 10, n),   # undesirable output
})

inputs = ["x1", "x2"]
desirable = ["y1"]
undesirable = ["b1"]

# --- SBM (CRS) ---
print("=" * 50)
print("SBM (CRS)")
print("=" * 50)
sbm = SBM(inputs=inputs, desirable_outputs=desirable,
          undesirable_outputs=undesirable, data=df)
res_sbm = sbm.solve(scale="c")
print(res_sbm[["dmu", "TE"]].to_string(index=False))

# --- Super-SBM ---
print("\n" + "=" * 50)
print("Super-SBM")
print("=" * 50)
ssbm = SuperSBM(inputs=inputs, desirable_outputs=desirable,
                undesirable_outputs=undesirable, data=df)
res_ssbm = ssbm.solve()
print(res_ssbm[["dmu", "TE"]].to_string(index=False))

# --- EBM (CRS) ---
print("\n" + "=" * 50)
print("EBM (CRS)")
print("=" * 50)
ebm = EBM(inputs=inputs, desirable_outputs=desirable,
          undesirable_outputs=undesirable, data=df)
res_ebm = ebm.solve(scale="c")
print(res_ebm[["dmu", "TE"]].to_string(index=False))

# --- Additive (VRS) ---
print("\n" + "=" * 50)
print("Additive (VRS)")
print("=" * 50)
add = Additive(inputs=inputs, desirable_outputs=desirable,
               undesirable_outputs=undesirable, data=df)
res_add = add.solve(scale="v")
print(res_add[["dmu", "TE"]].to_string(index=False))
