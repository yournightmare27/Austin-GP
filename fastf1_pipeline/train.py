# train.py
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from colorama import Fore, Style, init
import os
import joblib
import fastf1

from data_builder import build_au_dataset

init(autoreset=True)

# FastF1 downloads session data once and caches it locally
os.makedirs("cache_folder", exist_ok=True)
fastf1.Cache.enable_cache("cache_folder")


def train_and_eval():
    # Years: adjust as available in FastF1 for you
    train_years = [2018, 2019, 2021, 2022, 2023]  # 2020 had no Austin GP
    test_year = 2024

    print(f"\n{Fore.CYAN}🏎️ Building Austin GP dataset...{Style.RESET_ALL}")
    df = build_au_dataset(train_years + [test_year])
    assert not df.empty, "No data built."

    # Basic feature set (all known pre-race)
    feats = ["Q_best_lap_s", "Q_avg_lap_s", "Q_laps_count", "Q_delta_to_pole_s", "Grid"]

    # Keep rows with valid data
    df = df[df["Q_best_lap_s"].notna()].copy()
    df["Grid"] = pd.to_numeric(df["Grid"], errors="coerce").fillna(20)

    # Split by year (no leakage)
    df_train = df[df["Year"].isin(train_years)].copy()
    df_test = df[df["Year"] == test_year].copy()

    X_tr, y_tr = df_train[feats], df_train["FinishPosition"]
    X_te, y_te = df_test[feats], df_test["FinishPosition"]

    # Train Gradient Boosting Regressor
    print(f"{Fore.YELLOW}Training Gradient Boosting model...{Style.RESET_ALL}")
    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_tr, y_tr)

    # Evaluate
    pred_pos = model.predict(X_te)
    mae = mean_absolute_error(y_te, pred_pos)
    print(f"\n{Fore.CYAN}{test_year} MAE (finish position): {mae:.2f}{Style.RESET_ALL}\n")

    # Rank by predicted finishing position
    df_test = df_test.assign(PredictedPos=pred_pos).sort_values("PredictedPos").reset_index(drop=True)

    # Define podium symbols 🏆🥈🥉
    podium = ["🏆", "🥈", "🥉"]
    colors = [Fore.YELLOW, Fore.LIGHTWHITE_EX, Fore.LIGHTRED_EX]

    print(f"{Fore.MAGENTA}Predicted Top-3 ({test_year} Austin GP):{Style.RESET_ALL}\n")
    for i in range(3):
        row = df_test.iloc[i]
        medal = podium[i]
        color = colors[i]
        driver = f"{row['Driver']}"
        team = f"{row['Team']}"
        pred = f"{row['PredictedPos']:.3f}"
        print(f"{color}{medal} {i+1}. {driver} ({team}) — Predicted Pos: {pred}{Style.RESET_ALL}")

    print("\n" + "-" * 55)
    print(f"{Fore.GREEN}Actual Top-3 ({test_year} Austin GP):{Style.RESET_ALL}\n")

    # Actual 2024 Austin GP results
    actual = [
        ("Charles Leclerc", "Ferrari"),
        ("Carlos Sainz", "Ferrari"),
        ("Max Verstappen", "Red Bull")
    ]
    for i, (driver, team) in enumerate(actual):
        medal = podium[i]
        color = colors[i]
        print(f"{color}{medal} {i+1}. {driver} ({team}){Style.RESET_ALL}")

    print("-" * 55)

    # Save the trained model here, inside the function
    joblib.dump(model, f"../models/model_austin_{test_year}.pkl")
    print(f"💾 Model saved as models/model_austin_{test_year}.pkl")

    return model, feats, test_year


if __name__ == "__main__":
    train_and_eval()
