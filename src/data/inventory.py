from src.data.load import RAW, load_file

lines = []
for f in sorted(list(RAW.rglob("*.csv")) + list(RAW.rglob("*.arff"))):
    df = load_file(f)
    lines.append(f"\n=== {f.relative_to(RAW)} ===")
    lines.append(f"rows={len(df)}, cols={df.shape[1]}")
    lines.append("columns/dtypes:\n" + df.dtypes.to_string())
    lines.append("missing per column:\n" + df.isna().sum().to_string())
    lines.append("head:\n" + df.head(3).to_string())

out = "\n".join(lines)
print(out)
open("reports/dataset_inventory.txt", "w", encoding="utf-8").write(out)