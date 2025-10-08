import pandas as pd
from pathlib import Path

# === Path to your postflop manifest ===
MANIFEST_PATH = Path("data/artifacts/rangenet_postflop_flop_manifest.parquet")

# Expected BvS menu mapping (after your alias fix)
EXPECTED_MENU_ID = "srp_hu.PFR_IP"
EXPECTED_SIZES = [0.33, 0.66]

def main():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    df = pd.read_parquet(MANIFEST_PATH)
    print(f"Loaded manifest: {len(df):,} rows")

    # --- Relaxed detection of Blind-vs-Steal scenarios ---
    # Some manifests normalize ctx to "SRP" but keep the scenario name with "BLIND_VS_STEAL"
    mask_ctx = df["ctx"].str.upper().isin(["BLIND_VS_STEAL"])
    mask_name = df.get("scenario", df.get("name", "")).astype(str).str.contains("BLIND_VS_STEAL", case=False, na=False)
    bvs_rows = df[mask_ctx | mask_name]

    print(f"Found {len(bvs_rows):,} Blind-vs-Steal rows (by ctx or scenario match)")

    if bvs_rows.empty:
        print("⚠️ No BvS rows found — check your manifest builder config or YAML scenario names.")
        print("💡 Tip: The builder may have normalized BLIND_VS_STEAL → SRP in ctx.")
        return

    # --- Check menu IDs and bet sizes ---
    bad_menu = bvs_rows[bvs_rows["bet_sizing_id"] != EXPECTED_MENU_ID]
    bad_sizes = bvs_rows[bvs_rows["bet_sizes"].apply(lambda s: s != EXPECTED_SIZES)]

    if bad_menu.empty and bad_sizes.empty:
        print("✅ All BvS rows use the correct bet_sizing_id and bet_sizes menu!")
    else:
        print("⚠️ Some BvS rows have incorrect menus or sizes:")
        if not bad_menu.empty:
            print(f"   - {len(bad_menu)} rows with wrong menu_id (expected {EXPECTED_MENU_ID})")
        if not bad_sizes.empty:
            print(f"   - {len(bad_sizes)} rows with wrong bet_sizes (expected {EXPECTED_SIZES})")
        print("💡 Suggestion: re-run manifest builder for only the BvS scenarios.")

    # --- Quick peek ---
    print("\nUnique bet_sizing_id values in BvS rows:")
    print(bvs_rows["bet_sizing_id"].value_counts())

    print("\nSample BvS rows:")
    print(bvs_rows[["ctx", "scenario", "ip_actor_flop", "oop_actor_flop", "bet_sizing_id", "bet_sizes"]].head(10).to_string(index=False))

if __name__ == "__main__":
    main()