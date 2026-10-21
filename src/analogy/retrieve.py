import numpy as np
import pandas as pd

def similar_projects(size, unit, k=3):
    df = pd.read_csv("data/processed/unified.csv")
    g = df[df["size_unit"] == unit].copy()
    g["distance"] = (np.log(g["size_value"]) - np.log(size)).abs()
    cols = ["dataset", "size_value", "effort_hours", "duration_months"]
    return g.nsmallest(k, "distance")[cols].round(1)

if __name__ == "__main__":
    print("3 projects closest to 300 function points:")
    print(similar_projects(300, "fp").to_string(index=False))
    print("\n3 projects closest to 50 KLOC:")
    print(similar_projects(50, "kloc").to_string(index=False))