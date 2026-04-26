import pandas as pd
from src.preprocessing import clean_dataset
def test_clean_dataset_fills_missing_and_sorts():
    df = pd.DataFrame({"timestamp":["2025-01-02","2025-01-01"],"humidity_pct":[50.0,None],"risk_level":[1,1],"failure_risk":[1,1]})
    cleaned = clean_dataset(df)
    assert cleaned["humidity_pct"].isna().sum() == 0
    assert str(cleaned.iloc[0]["timestamp"]).startswith("2025-01-01")
