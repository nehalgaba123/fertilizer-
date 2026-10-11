"""Step 1 - Data input and inspection (the raw file is never modified)."""
import os

import pandas as pd

from src import config
from src.report import log, subheading


def load_csv(path):
    """Read the CSV with Pandas; raise clear errors (exception handling)."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Download 'Fertilizer Prediction' from\n"
            "  https://www.kaggle.com/datasets/gdabhishek/fertilizer-prediction\n"
            "and save the CSV as data/fertilizer.csv (see README.md)."
        )
    try:
        df = pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"'{path}' could not be read as a CSV file: {exc}") from exc
    if df.empty:
        raise ValueError(f"'{path}' contains no rows.")
    return df


def standardize_columns(df):
    """Return a COPY with tidy column names: spaces -> '_', known typos fixed
    (Temparature -> Temperature, Phosphorous -> Phosphorus)."""
    new_names = []
    for col in df.columns:
        name = "_".join(str(col).strip().split())
        new_names.append(config.COLUMN_FIXES.get(name, name))
    out = df.copy()
    out.columns = new_names
    return out


def check_required_columns(df):
    expected = config.NUMERIC_COLS + config.CATEGORICAL_COLS + [config.TARGET]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(
            f"Expected column(s) {missing} not found after standardisation. "
            f"Columns present: {list(df.columns)}"
        )


def find_invalid_numeric(df):
    """Count, per numeric column, values that are non-numeric text or outside
    config.VALID_RANGES. Missing values (NaN) are NOT counted here."""
    rows = []
    for col in config.NUMERIC_COLS:
        if col not in df.columns:
            continue
        converted = pd.to_numeric(df[col], errors="coerce")
        non_numeric = int((df[col].notna() & converted.isna()).sum())
        lo, hi = config.VALID_RANGES[col]
        below = int((converted < lo).sum()) if lo is not None else 0
        above = int((converted > hi).sum()) if hi is not None else 0
        rows.append({"column": col, "non_numeric_text": non_numeric,
                     "below_valid_min": below, "above_valid_max": above,
                     "total_invalid": non_numeric + below + above})
    return pd.DataFrame(rows).set_index("column")


def inspect_dataset(df):
    """Print the inspection checklist from the project brief."""
    subheading("First 5 rows")
    log(df.head().to_string())
    subheading("Last 5 rows")
    log(df.tail().to_string())

    subheading("Dimensions and columns")
    log(f"Rows: {df.shape[0]}   Columns: {df.shape[1]}")
    log(f"Column names: {list(df.columns)}")

    subheading("Data types / DataFrame information")
    info = pd.DataFrame({"dtype": df.dtypes.astype(str),
                         "non_null": df.notna().sum(),
                         "missing": df.isna().sum(),
                         "unique": df.nunique()})
    log(info.to_string())

    subheading("Descriptive statistics (numeric columns)")
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if c not in num_cols]
    log(df[num_cols].describe().T.round(2).to_string())
    log(f"\nNumerical columns  : {num_cols}")
    log(f"Categorical columns: {cat_cols}")

    subheading("Unique categories")
    for col in cat_cols:
        values = sorted(df[col].dropna().astype(str).unique())
        log(f"{col}: {len(values)} -> {values}")

    subheading("Target column")
    if config.TARGET in df.columns:
        log(f"Target = '{config.TARGET}'")
        counts = df[config.TARGET].value_counts(dropna=False)
        dist = pd.DataFrame({"count": counts, "percent": (counts / len(df) * 100).round(1)})
        log(dist.to_string())
    else:
        log(f"WARNING: target column '{config.TARGET}' not found")

    subheading("Missing values, duplicates, invalid numbers")
    log(f"Total missing cells : {int(df.isna().sum().sum())}")
    log(f"Duplicate rows      : {int(df.duplicated().sum())}")
    log("Invalid numeric values (non-numeric text / outside plausible range):")
    log(find_invalid_numeric(df).to_string())
