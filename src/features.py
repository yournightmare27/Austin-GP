"""Build pre-race features from the Ergast F1 dataset (Kaggle CSV export).

Every feature is known before the race starts: qualifying pace, how far a driver
got in qualifying, starting grid slot, and recent driver/team form.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
COTA_CIRCUIT_ID = 69  # Circuit of the Americas, Austin

ORIGINAL_FEATURES = ["best_q_s", "delta_to_pole_s", "q_sessions", "grid"]
SIMPLE_FEATURES = ["pct_off_pole", "q_sessions", "grid"]
FORM_FEATURES = SIMPLE_FEATURES + ["driver_form", "team_form"]


def _read(name: str, data_dir: Path) -> pd.DataFrame:
    path = data_dir / name
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. See data/README.md for how to get the CSVs.")
    return pd.read_csv(path, na_values="\\N")


def _lap_seconds(t):
    if pd.isna(t) or t == "":
        return np.nan
    try:
        minutes, seconds = str(t).split(":")
        return int(minutes) * 60 + float(seconds)
    except ValueError:
        return np.nan


def build_dataset(start_year: int = 2014, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """One row per driver per race, with pre-race features and the finishing position."""
    quali = _read("qualifying.csv", data_dir)
    results = _read("results.csv", data_dir)
    races = _read("races.csv", data_dir)
    drivers = _read("drivers.csv", data_dir)
    constructors = _read("constructors.csv", data_dir)

    for col in ["q1", "q2", "q3"]:
        quali[f"{col}_s"] = quali[col].map(_lap_seconds)
    quali["best_q_s"] = quali[["q1_s", "q2_s", "q3_s"]].min(axis=1)
    quali["q_sessions"] = quali[["q1_s", "q2_s", "q3_s"]].notna().sum(axis=1)

    df = results[["raceId", "driverId", "constructorId", "grid", "positionOrder", "points"]].merge(
        quali[["raceId", "driverId", "best_q_s", "q_sessions"]], on=["raceId", "driverId"], how="left")
    df = df.merge(races[["raceId", "year", "round", "circuitId", "name", "date"]], on="raceId")
    df = df[df["year"] >= start_year].copy()
    df["date"] = pd.to_datetime(df["date"])

    # Grid 0 means a pit-lane start: treat it as the back of the grid
    df["grid"] = df["grid"].replace(0, 20)

    # Qualifying pace relative to pole; drivers without a time get the slowest value of that race
    pole = df.groupby("raceId")["best_q_s"].transform("min")
    df["delta_to_pole_s"] = df["best_q_s"] - pole
    df["pct_off_pole"] = (df["best_q_s"] / pole - 1) * 100
    for col in ["best_q_s", "delta_to_pole_s", "pct_off_pole"]:
        df[col] = df[col].fillna(df.groupby("raceId")[col].transform("max"))
    df["q_sessions"] = df["q_sessions"].fillna(0)

    # Recent form, using only races before this one
    df = df.sort_values(["date", "positionOrder"]).reset_index(drop=True)
    df["driver_form"] = df.groupby("driverId")["positionOrder"].transform(
        lambda s: s.shift().rolling(5, min_periods=1).mean()).fillna(15)
    team = (df.groupby(["raceId", "constructorId", "date"])["points"].sum()
              .reset_index().sort_values("date"))
    team["team_form"] = team.groupby("constructorId")["points"].transform(
        lambda s: s.shift().rolling(5, min_periods=1).mean())
    df = df.merge(team[["raceId", "constructorId", "team_form"]], on=["raceId", "constructorId"], how="left")
    df["team_form"] = df["team_form"].fillna(0)

    df = df.merge(drivers[["driverId", "code", "forename", "surname"]], on="driverId", how="left")
    df = df.merge(constructors[["constructorId", "name"]].rename(columns={"name": "team"}),
                  on="constructorId", how="left")
    df["driver"] = df["forename"] + " " + df["surname"]
    return df.dropna(subset=["best_q_s"]).reset_index(drop=True)
