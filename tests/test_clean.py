from src.data.clean import build_unified

def test_unified():
    df = build_unified()
    counts = df.groupby("dataset").size().to_dict()
    assert counts["desharnais"] == 77
    assert counts["nasa93"] == 93
    assert counts["cocomo811"] == 63
    assert (df["effort_hours"] > 0).all()
    assert (df["size_value"] > 0).all()
    assert set(df["size_unit"]) == {"kloc", "fp"}