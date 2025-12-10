import pandas as pd
import os

RAW_DATA_PATH = "data/raw/moneypuck_full_2008_2025.csv"
PROCESSED_DATA_PATH = "data/processed/moneypuck_modern.csv"

def main():
    print("Starting MoneyPuck data processing pipeline...")

    # ✅ Check that raw data exists
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(
            f"Raw data file not found at {RAW_DATA_PATH}.\n"
            "Please download the MoneyPuck CSV and place it in data/raw/ before running this script."
        )

    # ✅ Load raw data
    print("Loading raw MoneyPuck data...")
    df = pd.read_csv(RAW_DATA_PATH)

    print(f"Raw dataset shape: {df.shape}")

    # ✅ Apply required filters
    print("Filtering to modern team-level, all-situation data (2017–present)...")
    df_modern = df[
        (df["situation"] == "all") &
        (df["position"] == "Team Level") &
        (df["season"] >= 2017)
    ].copy()

    # ✅ Validate one row per team per game
    dup_check = df_modern.groupby(["gameId", "team"]).size().value_counts()
    if not (len(dup_check) == 1 and dup_check.index[0] == 1):
        raise ValueError("Duplicate team-game rows detected after filtering. Check filtering logic.")

    print(f"Filtered dataset shape: {df_modern.shape}")

    # ✅ Create processed directory if it doesn't exist
    os.makedirs("data/processed", exist_ok=True)

    # ✅ Save processed dataset
    df_modern.to_csv(PROCESSED_DATA_PATH, index=False)

    print(f"Modern dataset successfully saved to: {PROCESSED_DATA_PATH}")
    print("✅ Data pipeline completed successfully.")

if __name__ == "__main__":
    main()

