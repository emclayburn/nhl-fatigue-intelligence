import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score

DATA_PATH = "data/processed/moneypuck_features_final.csv"

def main():
    print("Starting modeling pipeline...")

    df = pd.read_csv(DATA_PATH)

    df["win"] = (df["goalsFor"] > df["goalsAgainst"]).astype(int)

    features = [
        "days_rest",
        "back_to_back",
        "rolling_goals_5",
        "rolling_xg_5",
        "fatigue_adj_goals",
        "fatigue_adj_xg",
        "opp_rolling_xg_against_5",
        "opp_days_rest"
    ]

    X = df[features]
    y_cls = df["win"]
    y_reg = df["goalsFor"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cls, test_size=0.2, random_state=42
    )

    clf = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("model", LogisticRegression(max_iter=1000))
    ])

    clf.fit(X_train, y_train)

    win_pred = clf.predict(X_test)
    win_proba = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, win_pred)
    auc = roc_auc_score(y_test, win_proba)

    print(f"✅ Win Model Accuracy: {acc:.3f}")
    print(f"✅ Win Model AUC: {auc:.3f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )

    reg = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("model", LinearRegression())
    ])

    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"✅ Goals RMSE: {rmse:.3f}")
    print(f"✅ Goals R²: {r2:.3f}")

if __name__ == "__main__":
    main()
