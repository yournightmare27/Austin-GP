# predict_2025.py
import os
import joblib
import fastf1
import pandas as pd
from colorama import Fore, Style, init
from data_builder import quali_features

init(autoreset=True)

# FastF1 downloads session data once and caches it locally
os.makedirs("cache_folder", exist_ok=True)
fastf1.Cache.enable_cache("cache_folder")

# Load saved model
model = joblib.load("../models/model_austin_2024.pkl")   # trained from previous run
feats = ["Q_best_lap_s", "Q_avg_lap_s", "Q_laps_count", "Q_delta_to_pole_s", "Grid"]

year = 2025
event = "Austin"

print(f"{Fore.CYAN}🏎️ Fetching latest qualifying data for {year} {event}...{Style.RESET_ALL}")
df = quali_features(year, event)

# Clean and predict
df = df.dropna(subset=feats).copy()
df["Grid"] = pd.to_numeric(df["Grid"], errors="coerce").fillna(20)

df["PredictedPos"] = model.predict(df[feats])
df = df.sort_values("PredictedPos").reset_index(drop=True)

# Podium display 🏆🥈🥉
podium = ["🏆", "🥈", "🥉"]
colors = [Fore.YELLOW, Fore.LIGHTWHITE_EX, Fore.LIGHTRED_EX]

print(f"\n{Fore.MAGENTA}Predicted Podium — {year} {event} GP{Style.RESET_ALL}\n")
for i in range(3):
    row = df.iloc[i]
    print(f"{colors[i]}{podium[i]} {i+1}. {row['Driver']} ({row['Team']}) — Predicted Pos: {row['PredictedPos']:.3f}{Style.RESET_ALL}")

print("\n" + "-"*50)
winner = df.iloc[0]["Driver"]
team = df.iloc[0]["Team"]
print(f"{Fore.GREEN}🏁 Predicted Winner: {winner} ({team}){Style.RESET_ALL}")
print("-"*50)
