# 🏒 NHL Fatigue Intelligence

An end-to-end data science pipeline that analyzes how scheduling fatigue (rest days and back-to-backs) impacts NHL team performance and win probability.

This project demonstrates a full, reproducible machine learning workflow including:
- Automated data processing  
- Time-aware feature engineering  
- Fatigue-adjusted performance modeling  
- Win probability prediction  
- Model evaluation and comparison  

---

## 📌 Project Motivation

NHL teams frequently play:
- Back-to-back games  
- Road-heavy schedule stretches  
- Compressed travel sequences  

These scheduling effects are widely believed to impact performance, but the true **quantitative effect of fatigue** is difficult to isolate.

This project answers:

**How much predictive signal does fatigue actually provide for goals scored and win probability?**

---

## 📂 Data Source

- MoneyPuck NHL Team-Level Data  
- Seasons: **2017–Present**  
- Granularity: **One row per team per game**  
- Scope includes:
  - Goals  
  - Expected goals (xG)  
  - Shots  
  - Possession metrics  
  - Game dates  
  - Home/Away  
  - Team & opponent identifiers  

Due to file size and source restrictions, raw CSV files are not stored in this repository.  
All datasets are regenerated locally using the automated pipeline.

---

## ⚙️ Full Pipeline Overview

### 1. Data Processing

Run:

```bash
python src/data_loader.py
```

This script:

- Loads raw MoneyPuck data

- Filters to:

 - situation == "all"

 - position == "Team Level"

 - season >= 2017

- Saves a clean modern modeling dataset

---

### 2. Feature Engineering

Run:

```bash
python src/feature_engineering.py
```

| Feature                    | Description                                      |
| -------------------------- | ------------------------------------------------ |
| `days_rest`                | Number of days since the team last played        |
| `back_to_back`             | Binary indicator for zero rest days              |
| `rolling_goals_5`          | 5-game rolling average of goals scored           |
| `rolling_xg_5`             | 5-game rolling average of expected goals (xG)    |
| `fatigue_adj_goals`        | Rolling goals adjusted downward on back-to-backs |
| `fatigue_adj_xg`           | Rolling xG adjusted downward on back-to-backs    |
| `opp_rolling_xg_against_5` | 5-game rolling average of opponent defensive xG  |
| `opp_days_rest`            | Opponent rest days entering the game             |

All rolling features are:

- Grouped by team

- Shifted to prevent data leakage

- Filled using expanding means for early-season games

---

### 3. Modeling and Evaluation

Run:

```bash
python src/modeling.py
```

Models used:

- Linear Regression -> Predict goals scored

- Logistic Regression -> Predict win probability

Model setup includes:

- Feature imputation using SimpleImputer

- Train/test split (80/20)

- Evaluation using:

 - RMSE and R^2 for regression

  - Accuracy and AUC for classification

---

## 📊 Results

### Regression: Predicting Goals

| Metric | Value |
| ------ | ----- |
| RMSE   | ~1.73 |
| R²     | ~0.01 |

Goal scoring at the single-game level is highly noisy. Fatigue alone explains only a small portion of variance, which is a realistic outcome for hockey analytics.

### Win Probability Model (Fatigue + Opponent Context)

| Metric   | Value  |
| -------- | ------ |
| Accuracy | ~0.553 |
| AUC      | ~0.565 |

The model performs measurably better than random (AUC > 0.50), showing that:

- Fatigue provides real but limited predictive signal

- Opponent context slightly improves discrimination

- Scheduling effects matter most in combination with overall team strength

--- 

## 🔬 Key Findings

- Fatigue alone is not sufficient to strongly predict single-game outcomes  
- Back-to-backs reduce rolling offensive performance  
- Adding opponent defensive form improves win model discrimination  
- Scheduling effects provide marginal predictive value, not dominant control  

Overall conclusion:

Fatigue matters — but only as one component of a much larger performance system.

---

## ✅ Technical Skills Demonstrated

- End-to-end reproducible ML pipeline  
- Time-series feature engineering  
- Data leakage prevention  
- Rolling window statistics  
- Interaction features  
- Pipeline-based model training  
- Regression and classification modeling  
- Model evaluation with multiple metrics  
- Git hygiene with large-file exclusion  

---

## 🚀 How to Run This Project Locally

```bash
git clone https://github.com/emclayburn/nhl-fatigue-intelligence.git
cd nhl-fatigue-intelligence
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Place MoneyPuck CSV in data/raw/
python src/data_loader.py
python src/feature_engineering.py
python src/modeling.py
```

--- 

## 📈 Future Work

- Add home/away interaction effects  
- Team identity encoding  
- Goalie performance features  
- Travel distance modeling  
- Special teams (power play / penalty kill) effects  
- Playoff-specific fatigue effects  
- Interactive Streamlit dashboard for win probability visualization  

---

## 👤 Author

**Ethan Clayburn**  
Statistics & Data Science  
Sports Analytics Focus  