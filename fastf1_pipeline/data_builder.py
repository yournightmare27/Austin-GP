# data_builder.py: build qualifying features and race results for one Grand Prix from FastF1
import fastf1
import pandas as pd
import numpy as np

def load_session(year: int, event: str, kind: str):
    s = fastf1.get_session(year, event, kind)
    s.load(laps=True, telemetry=False, weather=True)
    return s

def _safe_mode(series: pd.Series):
    s = series.dropna()
    return s.mode().iloc[0] if not s.empty else np.nan

def get_grid_from_race(year: int, event: str = "Austin") -> pd.DataFrame:
    """Fallback: race results table contains the final GridPosition (pre-race)."""
    r = load_session(year, event, "Race")
    res = r.results.rename(columns={"Abbreviation": "Driver"})
    return res[["Driver", "GridPosition"]].rename(columns={"GridPosition": "Grid"})

def quali_features(year: int, event: str = "Austin") -> pd.DataFrame:
    q = load_session(year, event, "Qualifying")
    laps = q.laps
    laps = laps[laps["IsAccurate"] == True].copy()

    g = laps.groupby("Driver")
    df = pd.DataFrame({
        "Driver": g.size().index,
        "Team": g["Team"].agg(_safe_mode).values,
        "Q_best_lap_s": g["LapTime"].min().dt.total_seconds().values,
        "Q_avg_lap_s": g["LapTime"].mean().dt.total_seconds().values,
        "Q_laps_count": g.size().values
    })

    # Delta to pole
    pole = df["Q_best_lap_s"].min()
    df["Q_delta_to_pole_s"] = df["Q_best_lap_s"] - pole

    # Try to get Grid from qualifying results first
    try:
        res_q = q.results.rename(columns={"Abbreviation": "Driver",
                                          "GridPosition": "Grid"})
        df = df.merge(res_q[["Driver", "Grid"]], on="Driver", how="left")
    except Exception:
        df["Grid"] = np.nan

    # Fallback: get Grid from the Race results (historical data)
    if df["Grid"].isna().any():
        try:
            grid_r = get_grid_from_race(year, event)
            df = df.merge(grid_r, on="Driver", how="left", suffixes=("", "_r"))
            df["Grid"] = df["Grid"].fillna(df["Grid_r"])
            df.drop(columns=["Grid_r"], inplace=True)
        except Exception as e:
            print(f"[WARN] {year} grid fallback failed: {e}")

    df["Grid"] = pd.to_numeric(df["Grid"], errors="coerce")
    df["Year"] = year
    df["Event"] = event

    # Keep rows with at least a valid best lap
    return df.dropna(subset=["Q_best_lap_s"]).reset_index(drop=True)

def race_target(year: int, event: str = "Austin") -> pd.DataFrame:
    r = load_session(year, event, "Race")
    res = r.results.rename(columns={"Abbreviation": "Driver"}).copy()
    res["FinishPosition"] = pd.to_numeric(res["Position"], errors="coerce").fillna(99)
    return res[["Driver", "FinishPosition"]].assign(Year=year, Event=event)

def build_au_dataset(years: list[int]) -> pd.DataFrame:
    rows = []
    for y in years:
        try:
            q = quali_features(y, "Austin")
            t = race_target(y, "Austin")
            df = q.merge(t, on=["Driver", "Year", "Event"], how="inner")
            rows.append(df)
        except Exception as e:
            print(f"[WARN] {y} skipped: {e}")
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

