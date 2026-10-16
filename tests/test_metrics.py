from src.metrics import mmre, pred, mae

def test_pred25():
    # 100 vs 120 -> MRE ~0.167 (inside 25%); 100 vs 200 -> 0.5 (outside)
    assert pred([120, 200], [100, 100]) == 50.0

def test_mae():
    assert mae([10, 20], [12, 18]) == 2.0