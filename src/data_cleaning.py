"""Step 2 - Data cleaning. Works on a COPY; the raw DataFrame is untouched.

Order of operations (and why):
 1. Trim text, turn empty strings into missing.
 2. Convert numeric columns; values that are text or physically impossible
    become missing (so they are handled like any other missing value).
 3. Drop rows with a missing TARGET (we cannot learn from / score them).
 4. Drop exact duplicate rows (an identical record adds no information and
    would let the same row land in both train and test sets).
 5. Impute missing numeric values with the MEDIAN (robust to outliers).
 6. Fill missing categorical features with the label 'Unknown'
    (we do not guess a soil/crop type).
Zeros in nutrient columns are valid and are never treated as missing.
Outliers are only REPORTED (IQR rule), never deleted automatically.
"""
import os

import numpy as np
import pandas as pd

from src import config
from src.data_loading import find_invalid_numeric
from src.report import log, subheading


def iqr_outliers(df, cols):
    """Tukey fences: flag values < Q1-1.5*IQR or > Q3+1.5*IQR."""
    rows = []
    for col in cols:
        x = df[col].to_numpy(dtype=float)
        q1, q3 = np.percentile(x, [25, 75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((x < low) | (x > high)).sum())
        rows.append({"column": col, "Q1": q1, "Q3": q3, "IQR": iqr,
                     "lower_fence": low, "upper_fence": high,
                     "n_outliers": n_out, "percent": round(100 * n_out / len(x), 1)})
    return pd.DataFrame(rows).set_index("column")


def clean_data(df_std):
    """Return (cleaned_df, summary_dict)."""
    df = df_std.copy()
    summary = {
        "rows_before": len(df),
        "missing_before": df.isna().sum(),
        "invalid_before": find_invalid_numeric(df),
        "duplicates_before": int(df.duplicated().sum()),
    }

    # 1. text columns: trim whitespace, empty string -> missing
    for col in config.CATEGORICAL_COLS + [config.TARGET]:
        cleaned = df[col].astype("string").str.strip()
        cleaned = cleaned.mask(cleaned == "", pd.NA)
        df[col] = cleaned.astype(object).where(cleaned.notna(), np.nan)

    # 2. numeric conversion + impossible values -> NaN
    for col in config.NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        lo, hi = config.VALID_RANGES[col]
        bad = pd.Series(False, index=df.index)
        if lo is not None:
            bad |= df[col] < lo
        if hi is not None:
            bad |= df[col] > hi
        df.loc[bad, col] = np.nan

    # 3. missing target
    summary["missing_target_rows"] = int(df[config.TARGET].isna().sum())
    df = df[df[config.TARGET].notna()]

    # 4. duplicates
    summary["duplicates_removed"] = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)

    # 5. numeric imputation (median)
    imputed = {}
    for col in config.NUMERIC_COLS:
        n_missing = int(df[col].isna().sum())
        if n_missing:
            median = df[col].median()
            if np.isnan(median):
                raise ValueError(f"Column '{col}' has no valid values to impute from.")
            df[col] = df[col].fillna(median)
            imputed[col] = (n_missing, float(median))
        if (df[col] % 1 == 0).all():          # keep whole numbers as integers
            df[col] = df[col].astype(int)
    summary["numeric_imputed"] = imputed

    # 6. categorical imputation
    cat_filled = {}
    for col in config.CATEGORICAL_COLS:
        n_missing = int(df[col].isna().sum())
        if n_missing:
            df[col] = df[col].fillna("Unknown")
            cat_filled[col] = n_missing
    summary["categorical_filled"] = cat_filled

    summary["rows_after"] = len(df)
    summary["missing_after"] = df.isna().sum()
    summary["outliers"] = iqr_outliers(df, config.NUMERIC_COLS)
    return df, summary


def print_cleaning_summary(summary):
    subheading("Missing values per column: before vs after")
    table = pd.DataFrame({"before": summary["missing_before"],
                          "after": summary["missing_after"]})
    log(table.to_string())

    subheading("Cleaning actions")
    log(f"Rows before cleaning           : {summary['rows_before']}")
    log(f"Rows with missing target dropped: {summary['missing_target_rows']}")
    log(f"Exact duplicate rows removed    : {summary['duplicates_removed']}")
    log(f"Rows after cleaning            : {summary['rows_after']}")
    if summary["numeric_imputed"]:
        for col, (n, med) in summary["numeric_imputed"].items():
            log(f"Imputed {n} missing/invalid value(s) in {col} with median = {med:g}")
    else:
        log("No numeric imputation was needed.")
    if summary["categorical_filled"]:
        for col, n in summary["categorical_filled"].items():
            log(f"Filled {n} missing value(s) in {col} with 'Unknown'")
    else:
        log("No categorical imputation was needed.")
    log("Note: zero values in nutrient columns were kept (zero is a valid amount).")

    subheading("Invalid numeric values found in the raw data")
    log(summary["invalid_before"].to_string())

    subheading("Potential outliers (IQR rule) - reported, NOT removed")
    log(summary["outliers"].round(2).to_string())
    log("Outliers can be genuine field conditions, so they are kept.")

    subheading("Before / after summary")
    log(f"{'':22}{'before':>10}{'after':>10}")
    log(f"{'rows':22}{summary['rows_before']:>10}{summary['rows_after']:>10}")
    log(f"{'missing cells':22}{int(summary['missing_before'].sum()):>10}"
        f"{int(summary['missing_after'].sum()):>10}")
    log(f"{'duplicate rows':22}{summary['duplicates_before']:>10}{0:>10}")
    log(f"{'invalid numeric cells':22}{int(summary['invalid_before']['total_invalid'].sum()):>10}{0:>10}")


def save_cleaned(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    log(f"Cleaned dataset saved to: {path}")
