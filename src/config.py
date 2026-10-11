"""Project-wide settings (column names, valid ranges, file paths, parameters)."""
import os

# ---- Column names AFTER standardisation (see data_loading.standardize_columns) ----
TARGET = "Fertilizer_Name"
NUMERIC_COLS = ["Temperature", "Humidity", "Moisture", "Nitrogen", "Potassium", "Phosphorus"]
CATEGORICAL_COLS = ["Soil_Type", "Crop_Type"]
NUTRIENT_COLS = ["Nitrogen", "Phosphorus", "Potassium"]

# Spelling fixes applied after whitespace -> underscore conversion
COLUMN_FIXES = {"Temparature": "Temperature", "Phosphorous": "Phosphorus"}

# Physically plausible limits. (None = no limit on that side.)
# A zero nutrient value is VALID (a fertilizer can legitimately need no P, for
# example), so the lower bound for nutrients is 0 and zeros are never treated
# as missing.
VALID_RANGES = {
    "Temperature": (-50, 60),
    "Humidity": (0, 100),
    "Moisture": (0, 100),
    "Nitrogen": (0, None),
    "Phosphorus": (0, None),
    "Potassium": (0, None),
}

# ---- Paths (relative to the project folder; main.py changes into that folder) ----
DATA_PATH = os.path.join("data", "fertilizer.csv")
PROCESSED_PATH = os.path.join("data", "processed", "fertilizer_cleaned.csv")
FIGURES_DIR = os.path.join("results", "figures")
REPORT_PATH = os.path.join("results", "analysis_report.txt")

# ---- Parameters ----
RANDOM_SEED = 42
TEST_FRACTION = 0.2
ALPHA = 0.05                      # significance level for all hypothesis tests
K_CANDIDATES = [1, 3, 5, 7, 9]    # neighbour counts tried for the classifier
REPEATED_SPLITS = 10              # repeated hold-out runs (robustness check)
