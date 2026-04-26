import numpy as np
import pandas as pd
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["humidity_temp_index"] = out["humidity_pct"] * out["temperature_c"] / 100.0
    out["rain_humidity_score"] = out["rainfall_mm"] + out["humidity_pct"] / 10.0
    out["environment_stress_score"] = (out["humidity_pct"] + out["pollution_proxy"] + out["salinity_proxy"]) / 3.0
    out["pressure_deviation"] = (out["pressure_hpa"] - 1013.0).abs()
    out["vibration_exposure_index"] = out["vibration_mm_s"] * np.log1p(out["exposure_hours"])
    out["high_humidity_flag"] = (out["humidity_pct"] >= 85).astype(int)
    out["high_salinity_flag"] = (out["salinity_proxy"] >= 55).astype(int)
    out["thermal_stress_flag"] = ((out["temperature_c"] >= 30) | (out["temperature_c"] <= 0)).astype(int)
    return out
