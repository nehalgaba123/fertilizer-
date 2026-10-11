"""Step 5 - Visualisation (Matplotlib + Seaborn only).

Each plot function saves a PNG and returns ONE sentence describing what the
actual data shows. The sentence is computed from the data, not typed in.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import config

sns.set_theme(style="whitegrid")
_STATE = {"keep_open": False}


def set_keep_open(flag):
    """If True, figures stay open so main.py can call plt.show()."""
    _STATE["keep_open"] = bool(flag)


def _save(fig, name):
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    path = os.path.join(config.FIGURES_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    if not _STATE["keep_open"]:
        plt.close(fig)
    return path


# 1 ---------------------------------------------------------------------------
def plot_missing(raw_df):
    miss = raw_df.isna().sum()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(x=miss.values, y=miss.index, color="indianred", ax=ax)
    ax.set_title("Missing values per column (raw data)")
    ax.set_xlabel("Number of missing values")
    ax.set_ylabel("Column")
    _save(fig, "01_missing_values.png")
    total = int(miss.sum())
    return "No missing values in the raw data." if total == 0 else \
        f"{total} missing cells; worst column is {miss.idxmax()} ({int(miss.max())})."


# 2-4 -------------------------------------------------------------------------
def _count_plot(df, col, title, fname, color):
    order = df[col].value_counts().index
    fig, ax = plt.subplots(figsize=(8, max(4, 0.45 * len(order) + 1.5)))
    sns.countplot(data=df, y=col, order=order, color=color, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Number of records")
    ax.set_ylabel(col.replace("_", " "))
    for c in ax.containers:
        ax.bar_label(c, padding=2)
    _save(fig, fname)
    vc = df[col].value_counts()
    return (f"Most common: {vc.index[0]} ({vc.iloc[0]}); least common: "
            f"{vc.index[-1]} ({vc.iloc[-1]}) out of {len(vc)} categories.")


def plot_fertilizer_distribution(df):
    return _count_plot(df, config.TARGET, "Fertilizer class distribution", "02_fertilizer_distribution.png", "seagreen")


def plot_crop_distribution(df):
    return _count_plot(df, "Crop_Type", "Crop type distribution", "03_crop_distribution.png", "goldenrod")


def plot_soil_distribution(df):
    return _count_plot(df, "Soil_Type", "Soil type distribution", "04_soil_distribution.png", "sienna")


# 5 ---------------------------------------------------------------------------
def plot_histograms(df):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, col in zip(axes.ravel(), config.NUMERIC_COLS):
        ax.hist(df[col], bins=min(15, max(5, int(np.sqrt(len(df))))), color="steelblue", edgecolor="white")
        ax.axvline(df[col].mean(), color="red", linestyle="--", label="mean")
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        ax.legend()
    fig.suptitle("Histograms of numerical features", fontsize=14)
    _save(fig, "05_histograms.png")
    skew = df[config.NUMERIC_COLS].skew()
    return f"Most skewed feature: {skew.abs().idxmax()} (skewness {skew[skew.abs().idxmax()]:.2f})."


# 6 ---------------------------------------------------------------------------
def plot_boxplots(df):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, col in zip(axes.ravel(), config.NUMERIC_COLS):
        sns.boxplot(y=df[col], color="lightsteelblue", ax=ax)
        ax.set_title(f"Box plot of {col}")
        ax.set_ylabel(col)
    fig.suptitle("Box plots of numerical features", fontsize=14)
    _save(fig, "06_boxplots.png")
    iqr = df[config.NUMERIC_COLS].quantile(0.75) - df[config.NUMERIC_COLS].quantile(0.25)
    return f"Widest spread (IQR): {iqr.idxmax()} ({iqr.max():.1f}); narrowest: {iqr.idxmin()} ({iqr.min():.1f})."


# 7 ---------------------------------------------------------------------------
def plot_outliers(df, outlier_table):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, col in zip(axes.ravel(), config.NUMERIC_COLS):
        low = outlier_table.loc[col, "lower_fence"]
        high = outlier_table.loc[col, "upper_fence"]
        is_out = (df[col] < low) | (df[col] > high)
        ax.scatter(df.index[~is_out], df.loc[~is_out, col], s=14, color="steelblue", label="within fences")
        ax.scatter(df.index[is_out], df.loc[is_out, col], s=30, color="red", label="IQR outlier")
        ax.axhline(low, color="grey", linestyle=":")
        ax.axhline(high, color="grey", linestyle=":")
        ax.set_title(f"{col}: {int(is_out.sum())} outlier(s)")
        ax.set_xlabel("Row index")
        ax.set_ylabel(col)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Potential outliers by the 1.5 x IQR rule (dotted = fences)", fontsize=14)
    _save(fig, "07_outliers.png")
    total = int(outlier_table["n_outliers"].sum())
    if total == 0:
        return "No values fall outside the IQR fences."
    worst = outlier_table["n_outliers"].idxmax()
    return f"{total} flagged value(s) in total; most in {worst} ({int(outlier_table.loc[worst, 'n_outliers'])})."


# 8 ---------------------------------------------------------------------------
def plot_correlation_heatmap(df):
    corr = df[config.NUMERIC_COLS].corr()
    fig, ax = plt.subplots(figsize=(7.5, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True, ax=ax)
    ax.set_title("Correlation heatmap (Pearson) of numerical features")
    _save(fig, "08_correlation_heatmap.png")
    best, pair = 0.0, ("", "")
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if abs(corr.iloc[i, j]) > abs(best):
                best, pair = corr.iloc[i, j], (cols[i], cols[j])
    return f"Strongest association: {pair[0]} vs {pair[1]} (r = {best:.2f}); association, not causation."


# 9 ---------------------------------------------------------------------------
def plot_scatter(df):
    pairs = [("Nitrogen", "Phosphorus"), ("Nitrogen", "Potassium"), ("Temperature", "Humidity")]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, (x, y) in zip(axes, pairs):
        sns.scatterplot(data=df, x=x, y=y, hue=config.TARGET, palette="tab10", s=45, ax=ax, legend=(ax is axes[-1]))
        r = df[x].corr(df[y])
        ax.set_title(f"{x} vs {y} (r = {r:.2f})")
    axes[-1].legend(title="Fertilizer", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.suptitle("Scatter plots of numerical features coloured by fertilizer", fontsize=14)
    _save(fig, "09_scatter_plots.png")
    r_vals = {f"{x}-{y}": df[x].corr(df[y]) for x, y in pairs}
    return "Pearson r: " + "; ".join(f"{k} = {v:.2f}" for k, v in r_vals.items()) + \
           ". Check the colours for visible class groupings."


# 10 --------------------------------------------------------------------------
def plot_pairplot(df):
    grid = sns.pairplot(df, vars=config.NUMERIC_COLS, hue=config.TARGET, palette="tab10",
                        corner=True, height=1.7, plot_kws={"s": 18})
    grid.figure.suptitle("Pair plot of numerical features by fertilizer class", y=1.01, fontsize=14)
    _save(grid.figure, "10_pairplot.png")
    return "Diagonal shows each feature's distribution per class; off-diagonal panels show pairwise structure."


# 11 --------------------------------------------------------------------------
def plot_nutrients_by_class(df):
    order = df.groupby(config.TARGET)["Nitrogen"].median().sort_values().index
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    for ax, nut in zip(axes, config.NUTRIENT_COLS):
        sns.boxplot(data=df, x=config.TARGET, y=nut, order=order, color="lightgreen", ax=ax)
        sns.stripplot(data=df, x=config.TARGET, y=nut, order=order, color="black", size=3, alpha=0.6, ax=ax)
        ax.set_title(f"{nut} by fertilizer class")
        ax.set_xlabel("Fertilizer")
        ax.set_ylabel(nut)
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle("Nutrient levels grouped by fertilizer class", fontsize=14)
    _save(fig, "11_nutrients_by_fertilizer.png")
    med = df.groupby(config.TARGET)[config.NUTRIENT_COLS].median()
    return "Highest median: " + "; ".join(f"{n} -> {med[n].idxmax()} ({med[n].max():g})" for n in config.NUTRIENT_COLS) + "."


# 12 --------------------------------------------------------------------------
def plot_average_nutrients(df):
    means = df.groupby(config.TARGET)[config.NUTRIENT_COLS].mean()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    means.plot(kind="bar", ax=ax, width=0.8)
    ax.set_title("Average N, P, K per fertilizer class")
    ax.set_xlabel("Fertilizer")
    ax.set_ylabel("Mean nutrient level")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Nutrient")
    _save(fig, "12_average_nutrients.png")
    return "Highest average: " + "; ".join(f"{n} -> {means[n].idxmax()} ({means[n].max():.1f})" for n in config.NUTRIENT_COLS) + "."


# 13-14 -----------------------------------------------------------------------
def _crosstab_heatmap(df, col, title, fname):
    table = pd.crosstab(df[col], df[config.TARGET])
    fig, ax = plt.subplots(figsize=(1.0 * table.shape[1] + 4, 0.55 * table.shape[0] + 3))
    sns.heatmap(table, annot=True, fmt="d", cmap="YlGnBu", ax=ax, cbar_kws={"label": "Records"})
    ax.set_title(title)
    ax.set_xlabel("Fertilizer")
    ax.set_ylabel(col.replace("_", " "))
    ax.tick_params(axis="x", rotation=45)
    _save(fig, fname)
    stacked = table.stack()
    (row_name, fert), top = stacked.idxmax(), stacked.max()
    empty = int((table == 0).sum().sum())
    return (f"Most frequent combination: {row_name} + {fert} ({int(top)} records); "
            f"{empty} of {table.size} combinations never occur.")


def plot_crop_vs_fertilizer(df):
    return _crosstab_heatmap(df, "Crop_Type", "Crop type versus fertilizer class (record counts)", "13_crop_vs_fertilizer.png")


def plot_soil_vs_fertilizer(df):
    return _crosstab_heatmap(df, "Soil_Type", "Soil type versus fertilizer class (record counts)", "14_soil_vs_fertilizer.png")


# 15 --------------------------------------------------------------------------
def plot_confusion_matrix(cm, accuracy):
    fig, ax = plt.subplots(figsize=(1.0 * cm.shape[1] + 4, 0.8 * cm.shape[0] + 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar_kws={"label": "Test records"})
    ax.set_title(f"Confusion matrix - nearest-neighbour classifier (test accuracy {accuracy:.1%})")
    ax.set_xlabel("Predicted fertilizer")
    ax.set_ylabel("Actual fertilizer")
    ax.tick_params(axis="x", rotation=45)
    _save(fig, "15_confusion_matrix.png")
    values = cm.to_numpy().copy()
    np.fill_diagonal(values, 0)
    if values.max() == 0:
        return "No misclassifications on the test set."
    i, j = np.unravel_index(values.argmax(), values.shape)
    return f"Most frequent error: actual '{cm.index[i]}' predicted as '{cm.columns[j]}' ({values[i, j]} time(s))."
