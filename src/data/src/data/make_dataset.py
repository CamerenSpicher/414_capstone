#Make the clean dataset

import os
import numpy as np
import pandas as pd

RAW  = "data/raw/ACS_5-Year_Housing_Characteristics_DC_Census_Tract.csv"
PROC = "data/processed/dc_acs_cleaned_with_features.csv"
OUT_DIR = "outputs"

os.makedirs("data/processed", exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

def safe_div(num, den):
    """Robust division that avoids divide-by-zero and propagates NaN sensibly."""
    num = pd.to_numeric(num, errors="coerce")
    den = pd.to_numeric(den, errors="coerce")
    return np.where((den == 0) | pd.isna(den), np.nan, num / den)

def main():
    df = pd.read_csv(RAW)

    # Harmonize geometry column names if present
    df.rename(columns={"SHAPE_AREA": "SHAPEAREA", "SHAPE_LEN": "SHAPELEN"}, inplace=True)

    # Trim to valid identifiers and de-duplicate by GEOID
    if {"GEOID", "NAME"}.issubset(df.columns):
        df = df.dropna(subset=["GEOID", "NAME"]).drop_duplicates("GEOID")

    # ---- Feature engineering (DP04 profile) ----
    df["renter_share"]         = safe_div(df.get("DP04_0047E"), df.get("DP04_0045E"))
    df["owner_share"]          = safe_div(df.get("DP04_0046E"), df.get("DP04_0045E"))
    df["recent_movers_share"]  = safe_div(df.get("DP04_0051E"), df.get("DP04_0050E"))
    df["overcrowded_share"]    = safe_div(df.get("DP04_0079E"), df.get("DP04_0076E"))
    df["rent_burden35_share"]  = safe_div(df.get("DP04_0142E"), df.get("DP04_0136E"))
    df["median_rent"]          = pd.to_numeric(df.get("DP04_0134E"), errors="coerce")
    df["vacancy_rate_rental"]  = pd.to_numeric(df.get("DP04_0005E"), errors="coerce")
    df["median_rooms"]         = pd.to_numeric(df.get("DP04_0037E"), errors="coerce")

    # Optional: bedroom shares if base exists
    if "DP04_0038E" in df.columns:
        base = pd.to_numeric(df["DP04_0038E"], errors="coerce")
        for code, newcol in [
            ("DP04_0039E", "bed_0_share"),
            ("DP04_0040E", "bed_1_share"),
            ("DP04_0041E", "bed_2_share"),
            ("DP04_0042E", "bed_3_share"),
            ("DP04_0043E", "bed_4_share"),
            ("DP04_0044E", "bed_5p_share"),
        ]:
            if code in df.columns:
                df[newcol] = safe_div(pd.to_numeric(df[code], errors="coerce"), base)

    # Keep compact analysis set (IDs + engineered features)
    keep_id = [c for c in ["GEOID","NAME","TRACTCE","STATEFP","COUNTYFP","GEOIDFQ","NAMELSAD","INTPTLAT","INTPTLON"] if c in df.columns]
    keep_feats = [
        "median_rent","vacancy_rate_rental","median_rooms",
        "renter_share","owner_share","recent_movers_share","overcrowded_share","rent_burden35_share",
        "bed_0_share","bed_1_share","bed_2_share","bed_3_share","bed_4_share","bed_5p_share"
    ]
    keep = keep_id + [c for c in keep_feats if c in df.columns]
    df_out = df[keep].copy()
    df_out.to_csv(PROC, index=False)
    print(f"[make_dataset] saved {PROC} (rows={len(df_out)})")

    # Variable inventory and summary stats (non-visual, for docs)
    key = {"median_rent","vacancy_rate_rental","renter_share","recent_movers_share","overcrowded_share","rent_burden35_share","median_rooms"}
    inv = pd.DataFrame([{
        "variable": col,
        "dtype": str(df[col].dtype),
        "missing_pct": round(df[col].isna().mean()*100, 2),
        "relevance_to_problem": "★" if col in key else ""
    } for col in df.columns]).sort_values(["relevance_to_problem","variable"], ascending=[False,True])
    inv.to_csv(os.path.join(OUT_DIR, "variable_inventory.csv"), index=False)

    summ_cols = [c for c in ["median_rent","vacancy_rate_rental","median_rooms","renter_share","owner_share","recent_movers_share","overcrowded_share","rent_burden35_share"] if c in df_out.columns]
    df_out[summ_cols].describe().T.to_csv(os.path.join(OUT_DIR, "summary_stats.csv"))

if __name__ == "__main__":
    main()

