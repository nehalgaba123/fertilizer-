"""Step 4 - Statistical analysis (statistics module, NumPy, SciPy).

Every test prints: purpose, hypotheses, assumption checks, statistic, p-value
and a plain-language interpretation. Tests are only run when their
requirements (group sizes etc.) are met.
"""
import statistics

import numpy as np
import pandas as pd
from scipy import stats

from src import config
from src.report import log, subheading

ALPHA = config.ALPHA


def descriptive_table(df, cols):
    rows = []
    for col in cols:
        x = df[col].to_numpy(dtype=float)
        q1, q3 = np.percentile(x, [25, 75])
        data = x.tolist()
        rows.append({
            "feature": col,
            "mean": statistics.mean(data),
            "median": statistics.median(data),
            "mode": statistics.mode(data),            # first-most-common value
            "variance": np.var(x, ddof=1) if len(x) > 1 else np.nan,
            "std": np.std(x, ddof=1) if len(x) > 1 else np.nan,
            "Q1": q1, "Q3": q3, "IQR": q3 - q1,
            "skew": stats.skew(x), "kurtosis": stats.kurtosis(x),
        })
    return pd.DataFrame(rows).set_index("feature")


def run_descriptive(df):
    subheading("Descriptive statistics, quartiles, skewness")
    table = descriptive_table(df, config.NUMERIC_COLS)
    log(table.round(2).to_string())
    log("Skewness: ~0 symmetric, > 0 right tail, < 0 left tail "
        "(|skew| > 1 is usually called strongly skewed).")
    strong = [c for c in table.index if abs(table.loc[c, "skew"]) > 1]
    log("Strongly skewed features: " + (", ".join(strong) if strong else "none"))


def run_normality(df):
    subheading("Shapiro-Wilk normality test (each numeric feature)")
    log("Purpose: check whether a feature looks normally distributed.")
    log("H0: the feature is drawn from a normal distribution.  H1: it is not.")
    rows = []
    for col in config.NUMERIC_COLS:
        x = df[col].to_numpy(dtype=float)
        if len(x) < 3 or np.ptp(x) == 0:
            rows.append({"feature": col, "W": np.nan, "p_value": np.nan, "result": "test not possible"})
            continue
        w, p = stats.shapiro(x)
        rows.append({"feature": col, "W": w, "p_value": p,
                     "result": "reject H0 (not normal)" if p < ALPHA else "cannot reject H0"})
    log(pd.DataFrame(rows).set_index("feature").round(4).to_string())
    log(f"(alpha = {ALPHA}. 'Cannot reject H0' does not prove normality, "
        "especially with small samples.)")


def _group_arrays(df, feature, min_n=3):
    groups, names, skipped = [], [], []
    for name, sub in df.groupby(config.TARGET):
        if len(sub) >= min_n:
            groups.append(sub[feature].to_numpy(dtype=float))
            names.append(name)
        else:
            skipped.append(name)
    return groups, names, skipped


def run_group_comparisons(df):
    subheading("Do nutrient levels differ between fertilizer classes?")
    log("Purpose: compare a feature across 2+ fertilizer groups.")
    log("H0: all groups have the same distribution/mean.  H1: at least one group differs.")
    log("Choice of test: if every group passes Shapiro-Wilk AND Levene (equal variances)")
    log("  -> one-way ANOVA; otherwise -> Kruskal-Wallis (non-parametric).")
    log("t-test / Mann-Whitney / Wilcoxon compare only TWO groups (or paired data); "
        "with several fertilizer classes ANOVA / Kruskal-Wallis are the right tools.")
    for feature in config.NUTRIENT_COLS + ["Temperature", "Humidity", "Moisture"]:
        groups, names, skipped = _group_arrays(df, feature)
        log(f"\n[{feature}]")
        if skipped:
            log(f"  Classes with < 3 rows left out of the test: {skipped}")
        if len(groups) < 2:
            log("  Not enough groups with >= 3 rows; test skipped.")
            continue
        normal = True
        for g, n in zip(groups, names):
            if np.ptp(g) == 0:
                normal = False                      # constant group: normality cannot hold
                continue
            if stats.shapiro(g)[1] < ALPHA:
                normal = False
        lev_stat, lev_p = stats.levene(*groups, center="median")
        equal_var = lev_p >= ALPHA
        log(f"  Group normality (Shapiro) satisfied: {normal}; "
            f"Levene W = {lev_stat:.3f}, p = {lev_p:.4f} -> equal variances: {equal_var}")
        n_total = sum(len(g) for g in groups)
        if normal and equal_var:
            f, p = stats.f_oneway(*groups)
            grand = np.concatenate(groups).mean()
            ssb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
            sst = sum(((g - grand) ** 2).sum() for g in groups)
            eta2 = ssb / sst if sst > 0 else np.nan
            log(f"  Test: one-way ANOVA   F = {f:.3f}, p = {p:.4g}, eta^2 = {eta2:.2f}")
        else:
            h, p = stats.kruskal(*groups)
            eta2 = (h - len(groups) + 1) / (n_total - len(groups)) if n_total > len(groups) else np.nan
            log(f"  Test: Kruskal-Wallis  H = {h:.3f}, p = {p:.4g}, eta^2_H = {eta2:.2f}")
        if p < ALPHA:
            log(f"  Interpretation: p < {ALPHA}: reject H0 - {feature} differs between at least "
                "two fertilizer classes (a post-hoc test would be needed to say which).")
        else:
            log(f"  Interpretation: p >= {ALPHA}: no evidence that {feature} differs between classes.")
    log("\nNote: several tests were run, so a few 'significant' results can occur by chance; "
        "interpret borderline p-values cautiously.")


def run_chi_square(df):
    subheading("Chi-square tests of independence (categorical vs fertilizer)")
    log("Purpose: is the fertilizer label associated with soil type / crop type?")
    log("H0: the two variables are independent.  H1: they are associated.")
    for col in config.CATEGORICAL_COLS:
        table = pd.crosstab(df[col], df[config.TARGET])
        log(f"\n[{col} x {config.TARGET}]  table size: {table.shape[0]} x {table.shape[1]}")
        if table.shape[0] < 2 or table.shape[1] < 2:
            log("  Need at least 2 categories in each variable; test skipped.")
            continue
        chi2, p, dof, expected = stats.chi2_contingency(table.to_numpy())
        low_cells = float((expected < 5).mean() * 100)
        n = table.to_numpy().sum()
        v = np.sqrt(chi2 / (n * (min(table.shape) - 1)))
        log(f"  chi2 = {chi2:.3f}, dof = {dof}, p = {p:.4g}, Cramer's V = {v:.2f}")
        log(f"  Cells with expected count < 5: {low_cells:.0f}%")
        if low_cells > 20:
            log("  WARNING: more than 20% of expected counts are < 5, so the chi-square "
                "approximation is unreliable here (dataset too small for this table).")
        if p < ALPHA:
            log(f"  Interpretation: p < {ALPHA}: evidence of association (strength given by V).")
        else:
            log(f"  Interpretation: p >= {ALPHA}: no evidence of association.")


def run_correlation(df):
    subheading("Correlation analysis")
    log("Pearson measures linear association; Spearman measures monotonic (rank) association.")
    log("H0 for each pair: true correlation = 0.")
    pearson = df[config.NUMERIC_COLS].corr(method="pearson")
    spearman = df[config.NUMERIC_COLS].corr(method="spearman")
    log("Pearson:")
    log(pearson.round(2).to_string())
    log("Spearman:")
    log(spearman.round(2).to_string())
    cols = config.NUMERIC_COLS
    rows = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r, p = stats.pearsonr(df[cols[i]], df[cols[j]])
            rows.append({"pair": f"{cols[i]} vs {cols[j]}", "pearson_r": r, "p_value": p})
    table = pd.DataFrame(rows)
    table["abs_r"] = table["pearson_r"].abs()
    table = table.sort_values("abs_r", ascending=False).drop(columns="abs_r").head(5)
    log("Five strongest Pearson pairs with p-values:")
    log(table.round(4).to_string(index=False))
    log("Correlation shows association only; it does not prove one variable causes another.")


def run_all_statistics(df):
    if len(df) < 30:
        log(f"CAUTION: only {len(df)} rows - tests have low power and assumptions are hard to verify.")
    run_descriptive(df)
    run_normality(df)
    run_correlation(df)
    run_group_comparisons(df)
    run_chi_square(df)
