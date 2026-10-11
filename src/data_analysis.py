"""Step 3 - Data manipulation and exploratory data analysis (Pandas / NumPy)."""
import numpy as np
import pandas as pd

from src import config
from src.report import log, subheading


def manipulation_demo(df):
    """Show the Pandas operations from the syllabus. Each block says WHY."""
    T = config.TARGET
    work = df.copy()                       # never modify the cleaned data

    subheading("1. Selecting columns and rows (loc / iloc)")
    log("Purpose: pull out just the soil-nutrient columns for the first 5 rows.")
    log(work.loc[0:4, config.NUTRIENT_COLS].to_string())
    log("iloc (rows 0-2, first 3 columns by position):")
    log(work.iloc[0:3, 0:3].to_string())

    subheading("2. Filtering with conditions")
    n_med = work["Nitrogen"].median()
    high_n = work[(work["Nitrogen"] > n_med) & (work["Moisture"] >= work["Moisture"].median())]
    log("Purpose: find fields with above-median nitrogen AND at-least-median moisture.")
    log(f"Rows matching: {len(high_n)} of {len(work)}")
    log("Most common fertilizers among those rows:")
    log(high_n[T].value_counts().head(3).to_string())

    subheading("3. Derived columns (added to a copy)")
    work["NPK_Total"] = work["Nitrogen"] + work["Phosphorus"] + work["Potassium"]
    lo, hi = np.percentile(work["Temperature"], [33.3, 66.6])
    work["Temp_Band"] = np.select([work["Temperature"] <= lo, work["Temperature"] <= hi],
                                  ["Low", "Medium"], default="High")
    log("Purpose: NPK_Total = overall nutrient level; Temp_Band groups temperature "
        "into thirds so we can build frequency tables.")
    log(work[["Nitrogen", "Phosphorus", "Potassium", "NPK_Total", "Temperature", "Temp_Band"]]
        .head().to_string())

    subheading("4. Sorting")
    log("Purpose: which records have the largest total nutrient level?")
    log(work.sort_values("NPK_Total", ascending=False)[[T, "Crop_Type", "NPK_Total"]].head(5).to_string())
    log("Sorted by categorical column (Crop_Type, then Nitrogen descending):")
    log(work.sort_values(["Crop_Type", "Nitrogen"], ascending=[True, False])
        [["Crop_Type", "Nitrogen", T]].head(5).to_string())

    subheading("5. Grouping and aggregation")
    log("Purpose: compare nutrient levels across fertilizer classes.")
    agg = work.groupby(T)[config.NUTRIENT_COLS].agg(["count", "mean", "median", "min", "max", "std"]).round(2)
    log(agg.to_string())
    log("\nMean NPK_Total by soil type:")
    log(work.groupby("Soil_Type")["NPK_Total"].agg(["count", "mean", "median"]).round(2).to_string())
    log("\nMean nutrients by crop type:")
    log(work.groupby("Crop_Type")[config.NUTRIENT_COLS].mean().round(1).to_string())

    subheading("6. Frequency tables")
    log("Soil type x Temperature band (counts):")
    log(pd.crosstab(work["Soil_Type"], work["Temp_Band"]).to_string())
    log("\nFertilizer frequency (relative):")
    log(work[T].value_counts(normalize=True).round(3).to_string())

    subheading("7. Adding / deleting rows and columns (on a copy)")
    n0, c0 = work.shape
    work = work.drop(columns=["Temp_Band"])
    work = pd.concat([work, work.iloc[[0]]], ignore_index=True)   # add a row
    log(f"Dropped a column and appended a row: shape {n0, c0} -> {work.shape}")
    work = work.drop(index=work.index[-1])                         # delete a row
    log(f"Deleted the row again: shape {work.shape}")

    subheading("8. Missing observations and duplicates in the cleaned data")
    log(f"Missing cells: {int(df.isna().sum().sum())}   Duplicate rows: {int(df.duplicated().sum())}")


def eda_findings(df):
    """Print data-driven EDA findings (numbers computed here, not typed in)."""
    T = config.TARGET
    subheading("Overall structure")
    log(f"{len(df)} records, {df.shape[1]} columns, {df[T].nunique()} fertilizer classes, "
        f"{df['Soil_Type'].nunique()} soil types, {df['Crop_Type'].nunique()} crop types.")

    counts = df[T].value_counts()
    log(f"Class balance: largest class '{counts.index[0]}' ({counts.iloc[0]} rows), "
        f"smallest '{counts.index[-1]}' ({counts.iloc[-1]} rows).")
    if counts.iloc[-1] < 5:
        log("  -> Some classes have fewer than 5 examples; per-class results will be unreliable.")

    subheading("Mean of each feature per fertilizer class")
    log(df.groupby(T)[config.NUMERIC_COLS].mean().round(1).to_string())

    subheading("Which class has the highest mean of each nutrient?")
    means = df.groupby(T)[config.NUTRIENT_COLS].mean()
    for nut in config.NUTRIENT_COLS:
        log(f"{nut:11}: {means[nut].idxmax()} ({means[nut].max():.1f}); "
            f"lowest: {means[nut].idxmin()} ({means[nut].min():.1f})")

    subheading("Correlation matrix (Pearson) among numeric features")
    corr = df[config.NUMERIC_COLS].corr()
    log(corr.round(2).to_string())
    pairs = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs.append((abs(corr.iloc[i, j]), corr.iloc[i, j], cols[i], cols[j]))
    pairs.sort(reverse=True)
    for _, r, a, b in pairs[:3]:
        log(f"  strongest pairs: {a} vs {b}: r = {r:.2f}")
    log("Reminder: correlation describes association only; it does not show causation.")
