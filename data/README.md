# Data

The main notebook uses the Ergast F1 database as CSV files. They are not stored in this repository.

1. Download the Kaggle dataset **Formula 1 World Championship (1950–2024)** by Rohan Rao:
   https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020
2. Unzip it and put these files in this `data/` folder:
   `circuits.csv`, `constructors.csv`, `drivers.csv`, `qualifying.csv`, `races.csv`, `results.csv`
   (the exploration notebooks also read `lap_times.csv`, `pit_stops.csv`, `seasons.csv` and `status.csv`).

The live pipeline in `fastf1_pipeline/` does not need these files: it downloads session data from the FastF1 API and caches it in `fastf1_pipeline/cache_folder/`.
