from pathlib import Path
import pandas as pd
from scipy.io import arff

RAW = Path("data/raw")

def load_file(path):
    path = Path(path)
    if path.suffix.lower() == ".arff":
        data, _ = arff.loadarff(path)
        df = pd.DataFrame(data)
        for c in df.columns:
            df[c] = df[c].apply(lambda v: v.decode() if isinstance(v, bytes) else v)
        return df
    return pd.read_csv(path)