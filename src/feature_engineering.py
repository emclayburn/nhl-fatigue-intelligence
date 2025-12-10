import pandas as pd

INPUT_PATH = "data/processed/moneypuck_modern.csv"
OUTPUT_PATH = "data/processed/moneypuck_features_final.csv"

def main():
    print("Starting feature engineering...")

    df = pd.read_csv(INPUT_PATH)

    df["gameDate"] = pd.to_datetime(df["gameDate"], format="%Y%m%d")

    df = df.sort_values(["team", "gameDate"]).reset_index(drop=True)

    df["days_rest"] = df.groupby("team")["gameDate"].diff().dt.days - 1
    df["days_rest"] = df["days_rest"].fillna(3)

    df["back_to_back"] = (df["days_rest"] == 0).astype(int)

    df["rolling_goals_5"] = (
        df.groupby("team")["goalsFor"]
        .shift(1)
        .rolling(5)
        .mean()
    )

    df["rolling_xg_5"] = (
        df.groupby("team")["xGoalsFor"]
        .shift(1)
        .rolling(5)
        .mean()
    )

    df["rolling_goals_5"] = df.groupby("team")["rolling_goals_5"]\
        .transform(lambda x: x.fillna(x.expanding().mean()))

    df["rolling_xg_5"] = df.groupby("team")["rolling_xg_5"]\
        .transform(lambda x: x.fillna(x.expanding().mean()))

    df["fatigue_penalty"] = 1 - 0.1 * df["back_to_back"]

    df["fatigue_adj_goals"] = df["rolling_goals_5"] * df["fatigue_penalty"]
    df["fatigue_adj_xg"] = df["rolling_xg_5"] * df["fatigue_penalty"]

    df["opp_rolling_xg_against_5"] = (
        df.groupby("opposingTeam")["xGoalsAgainst"]
        .shift(1)
        .rolling(5)
        .mean()
    )

    df["opp_rolling_xg_against_5"] = (
        df.groupby("opposingTeam")["opp_rolling_xg_against_5"]
        .transform(lambda x: x.fillna(x.expanding().mean()))
    )

    df["opp_days_rest"] = (
        df.groupby("opposingTeam")["gameDate"].diff().dt.days - 1
    ).fillna(3)

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Features saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
