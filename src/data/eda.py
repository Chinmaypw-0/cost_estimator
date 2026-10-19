from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("data/processed/unified.csv")
Path("reports/figures").mkdir(parents=True, exist_ok=True)

print(df.groupby("dataset")[["size_value", "effort_hours"]].describe().T.to_string())
print("\nduplicate rows:", int(df.duplicated().sum()))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, (unit, g) in zip(axes, df.groupby("size_unit")):
    for name, d in g.groupby("dataset"):
        ax.scatter(d["size_value"], d["effort_hours"], label=name, s=18)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"size ({unit})")
    ax.set_ylabel("effort (person-hours)")
    ax.set_title(f"Size vs effort: {unit} family")
    ax.legend()
fig.tight_layout()
fig.savefig("reports/figures/size_vs_effort.png", dpi=150)

plt.figure(figsize=(6, 4))
plt.hist(np.log10(df["effort_hours"]), bins=30)
plt.xlabel("log10(effort hours)")
plt.ylabel("projects")
plt.tight_layout()
plt.savefig("reports/figures/effort_hist.png", dpi=150)
print("saved figures to reports/figures/")