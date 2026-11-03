"""Step 6 - Visualisation (Matplotlib + Seaborn + NumPy/SciPy for the statistics shown).

Exactly NINE analytical figures are produced. Each plot function saves a PNG
and returns ONE sentence describing what the actual data shows (computed from
the data, never typed in).

 01 Class profiles        - what defines each fertilizer (z-scored heatmap)
 02 NPK distributions     - nutrient ranges per fertilizer
 03 Nutrient space        - N vs P with K as bubble size
 04 Environment by class  - temperature / humidity / moisture + Kruskal-Wallis
 05 Variable importance   - effect size of every feature on the fertilizer label
 06 Crop & soil mix       - which fertilizers each crop / soil receives
 07 Correlations          - Pearson r with significance stars
 08 PCA projection        - multivariate class separability (biplot)
 09 Confusion matrix      - recommender performance vs majority baseline
"""
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats

from src import config

T = config.TARGET
_STATE = {"keep_open": False}

# Coordinated, colour-blind-friendly palette (one colour per fertilizer class)
PALETTE = ["#1F4E79", "#C0504D", "#2E8B57", "#E08E0B", "#7B4FA0",
           "#2AA3B0", "#8C6D46", "#D65F9B", "#5A5A5A", "#9AA63A"]
C_NUM, C_CAT = "#1F4E79", "#E08E0B"

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#444444", "axes.labelcolor": "#222222",
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "grid.color": "#E6E6E6", "grid.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "DejaVu Sans",
})

_UNITS = {"Temperature": "Temperature (\u00b0C)", "Humidity": "Humidity (%)",
          "Moisture": "Moisture (%)"}


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def set_keep_open(flag):
    """If True, figures stay open so main.py can call plt.show()."""
    _STATE["keep_open"] = bool(flag)


def clear_old_figures():
    """Delete numbered PNGs from earlier runs so only the 9 new figures remain."""
    for path in glob.glob(os.path.join(config.FIGURES_DIR, "*.png")):
        if re.match(r"^\d\d_.*\.png$", os.path.basename(path)):
            os.remove(path)


def _save(fig, name):
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    path = os.path.join(config.FIGURES_DIR, name)
    top = 0.94 if getattr(fig, "_suptitle", None) is not None else 1.0
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    if not _STATE["keep_open"]:
        plt.close(fig)
    return path


def _lab(col):
    return _UNITS.get(col, f"{col} (level)")


def _order(df):
    """Fertilizer classes ordered by mean N, then P, then K (same in every plot)."""
    means = df.groupby(T)[config.NUTRIENT_COLS].mean()
    return means.sort_values(["Nitrogen", "Phosphorus", "Potassium"], ascending=False).index.tolist()


def _colors(df):
    return {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(_order(df))}


def _stars(p):
    if p is None or np.isnan(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < config.ALPHA else ""


def _kruskal_p(df, col):
    groups = [g[col].to_numpy(float) for _, g in df.groupby(T) if len(g) >= 3]
    if len(groups) < 2:
        return np.nan
    try:
        return float(stats.kruskal(*groups)[1])
    except ValueError:                       # all values identical
        return np.nan


def _eta_squared(df, col):
    x = df[col].to_numpy(float)
    grand = x.mean()
    sst = ((x - grand) ** 2).sum()
    if sst == 0:
        return np.nan
    ssb = sum(len(s) * (s.mean() - grand) ** 2 for _, s in df.groupby(T)[col])
    return float(ssb / sst)


def _cramers_v(df, col):
    table = pd.crosstab(df[col], df[T])
    if table.shape[0] < 2 or table.shape[1] < 2:
        return np.nan, np.nan
    chi2, p, _, _ = stats.chi2_contingency(table.to_numpy())
    n = table.to_numpy().sum()
    return float(np.sqrt(chi2 / (n * (min(table.shape) - 1)))), float(p)


def _class_boxes(ax, df, col, order, colors):
    """Box plot + individual points for one feature, one box per fertilizer."""
    data = [df.loc[df[T] == c, col].to_numpy(float) for c in order]
    bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=False,
                    medianprops=dict(color="black", linewidth=1.6),
                    whiskerprops=dict(color="#555555"), capprops=dict(color="#555555"))
    rng = np.random.default_rng(config.RANDOM_SEED)
    for i, (patch, c, d) in enumerate(zip(bp["boxes"], order, data), 1):
        patch.set_facecolor(colors[c])
        patch.set_alpha(0.45)
        patch.set_edgecolor("#333333")
        ax.scatter(i + rng.uniform(-0.17, 0.17, len(d)), d, s=14, color=colors[c],
                   edgecolor="white", linewidth=0.4, alpha=0.95, zorder=3)
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(order, rotation=35, ha="right")
    ax.set_xlabel("Fertilizer")
    ax.set_ylabel(_lab(col))
    ax.grid(False, axis="x")


def _suptitle(fig, text):
    fig.suptitle(text, fontsize=14, fontweight="bold")


# ----------------------------------------------------------------------------
# 01  Class profiles
# ----------------------------------------------------------------------------
def plot_class_profile(df):
    feats = config.NUMERIC_COLS
    order = _order(df)
    sd = df[feats].std(ddof=0).replace(0, 1)
    z = ((df[feats] - df[feats].mean()) / sd).groupby(df[T]).mean().loc[order]
    means = df.groupby(T)[feats].mean().loc[order]
    n = df[T].value_counts()
    vmax = max(1.0, float(np.nanmax(np.abs(z.to_numpy()))))

    fig, ax = plt.subplots(figsize=(10, 0.7 * len(order) + 3))
    sns.heatmap(z, annot=means, fmt=".1f", cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
                linewidths=0.6, linecolor="white", ax=ax,
                cbar_kws={"label": "Class mean relative to overall mean (z-score)"})
    ax.set_xticklabels([_lab(c).replace(" (level)", "") for c in feats], rotation=30, ha="right")
    ax.set_yticklabels([f"{c}  (n={n[c]})" for c in order], rotation=0)
    ax.set_xlabel("Feature")
    ax.set_ylabel("Fertilizer class")
    ax.set_title("Cell colour = standardised class mean; printed value = actual class mean")
    _suptitle(fig, "What defines each fertilizer? Standardised feature profile per class")
    _save(fig, "01_class_profiles.png")
    r, c = np.unravel_index(np.nanargmax(np.abs(z.to_numpy())), z.shape)
    return (f"Most distinctive profile: {order[r]} has {feats[c]} {z.iloc[r, c]:+.1f} SD from the "
            f"overall mean (class mean {means.iloc[r, c]:.1f}).")


# ----------------------------------------------------------------------------
# 02  NPK distributions
# ----------------------------------------------------------------------------
def plot_nutrient_distributions(df):
    order, colors = _order(df), _colors(df)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6))
    for ax, nut in zip(axes, config.NUTRIENT_COLS):
        _class_boxes(ax, df, nut, order, colors)
        ax.set_title(f"{nut} requirement by fertilizer")
    _suptitle(fig, "Nutrient ranges recorded for each fertilizer class (box = IQR, dots = records)")
    _save(fig, "02_npk_distributions.png")
    med = df.groupby(T)[config.NUTRIENT_COLS].median()
    return "Highest median: " + "; ".join(
        f"{n} -> {med[n].idxmax()} ({med[n].max():g})" for n in config.NUTRIENT_COLS) + "."


# ----------------------------------------------------------------------------
# 03  Nutrient space (N vs P, bubble = K)
# ----------------------------------------------------------------------------
def plot_nutrient_space(df):
    order, colors = _order(df), _colors(df)
    k = df["Potassium"].to_numpy(float)
    k_rng = (k.max() - k.min()) or 1.0

    def size(v):
        return 40 + 260 * (np.asarray(v, float) - k.min()) / k_rng

    fig, ax = plt.subplots(figsize=(10.5, 6.6))
    for c in order:
        sub = df[df[T] == c]
        ax.scatter(sub["Nitrogen"], sub["Phosphorus"], s=size(sub["Potassium"]), color=colors[c],
                   alpha=0.72, edgecolor="white", linewidth=0.8, zorder=3)
    ax.set_xlabel(_lab("Nitrogen"))
    ax.set_ylabel(_lab("Phosphorus"))
    ax.set_title("Each fertilizer occupies its own region of nutrient space")
    cls_handles = [Line2D([], [], marker="o", linestyle="", color=colors[c], markersize=8, label=c)
                   for c in order]
    leg1 = ax.legend(handles=cls_handles, title="Fertilizer", bbox_to_anchor=(1.02, 1.0),
                     loc="upper left", frameon=False)
    ax.add_artist(leg1)
    vals = sorted({float(k.min()), float(np.median(k)), float(k.max())})
    size_handles = [ax.scatter([], [], s=size(v), color="#999999", alpha=0.7, label=f"K = {v:g}")
                    for v in vals]
    ax.legend(handles=size_handles, title="Potassium (bubble size)", bbox_to_anchor=(1.02, 0.35),
              loc="upper left", frameon=False, labelspacing=1.2)
    _suptitle(fig, "Nitrogen vs phosphorus requirement, with potassium as bubble size")
    _save(fig, "03_nutrient_space.png")
    pure = df.groupby(["Nitrogen", "Phosphorus"])[T].transform("nunique") == 1
    return (f"{pure.mean():.0%} of records sit at an N-P combination that belongs to only one "
            f"fertilizer class; {int((df['Phosphorus'] == 0).sum())} records need no phosphorus.")


# ----------------------------------------------------------------------------
# 04  Environmental conditions by class
# ----------------------------------------------------------------------------
def plot_environment_by_class(df):
    order, colors = _order(df), _colors(df)
    env = ["Temperature", "Humidity", "Moisture"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6))
    pvals = {}
    for ax, col in zip(axes, env):
        _class_boxes(ax, df, col, order, colors)
        pvals[col] = _kruskal_p(df, col)
        p = pvals[col]
        ptxt = "n/a" if np.isnan(p) else (f"p = {p:.3f}" if p >= 0.001 else "p < 0.001")
        ax.axhline(df[col].mean(), color="#777777", linestyle="--", linewidth=1)
        ax.set_title(f"{col}: Kruskal-Wallis {ptxt}{_stars(p)}")
    _suptitle(fig, "Do field conditions differ between fertilizer classes? (dashed = overall mean)")
    _save(fig, "04_environment_by_fertilizer.png")
    sig = [c for c, p in pvals.items() if not np.isnan(p) and p < config.ALPHA]
    return ("Significant difference between classes for: " + ", ".join(sig) + "."
            if sig else "No environmental variable differs significantly between fertilizer classes "
            f"(all Kruskal-Wallis p >= {config.ALPHA}).")


# ----------------------------------------------------------------------------
# 05  Variable importance (effect sizes)
# ----------------------------------------------------------------------------
def plot_variable_importance(df):
    rows = []
    for col in config.NUMERIC_COLS:
        rows.append((col, _eta_squared(df, col), _kruskal_p(df, col), "Numeric (eta\u00b2)"))
    for col in config.CATEGORICAL_COLS:
        v, p = _cramers_v(df, col)
        rows.append((col.replace("_", " "), v, p, "Categorical (Cram\u00e9r's V)"))
    res = pd.DataFrame(rows, columns=["feature", "effect", "p", "kind"]).dropna(subset=["effect"])
    res = res.sort_values("effect")

    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    colors = [C_NUM if k.startswith("Numeric") else C_CAT for k in res["kind"]]
    ax.barh(res["feature"], res["effect"], color=colors, edgecolor="white", height=0.65, zorder=3)
    for y, (v, p) in enumerate(zip(res["effect"], res["p"])):
        ax.text(v + 0.012, y, f"{v:.2f}{_stars(p)}", va="center", fontsize=9)
    ax.set_xlim(0, min(1.15, max(0.3, res["effect"].max() + 0.15)))
    ax.set_xlabel("Effect size on fertilizer label (0 = none, 1 = perfect)\n"
                  f"* significant at p < {config.ALPHA} (Kruskal-Wallis / chi-square); "
                  "V is upward-biased in small samples")
    ax.set_ylabel("Feature")
    ax.grid(False, axis="y")
    ax.legend(handles=[Patch(color=C_NUM, label="Numeric: eta\u00b2 (variance explained)"),
                       Patch(color=C_CAT, label="Categorical: Cram\u00e9r's V")],
              loc="lower right", frameon=False)
    ax.set_title("Which variables are most associated with the fertilizer choice?")
    _suptitle(fig, "Feature association strength with fertilizer class")
    _save(fig, "05_variable_importance.png")
    top, low = res.iloc[-1], res.iloc[0]
    return (f"Strongest association: {top['feature']} ({top['effect']:.2f}); "
            f"weakest: {low['feature']} ({low['effect']:.2f}).")


# ----------------------------------------------------------------------------
# 06  Crop and soil composition
# ----------------------------------------------------------------------------
def plot_categorical_associations(df):
    order = _order(df)
    specs = [("Crop_Type", "Crop type"), ("Soil_Type", "Soil type")]
    tables = [pd.crosstab(df[c], df[T]).reindex(columns=order, fill_value=0) for c, _ in specs]
    heights = [max(3, len(t)) for t in tables]
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 0.5 * sum(heights) + 4.2),
                             gridspec_kw={"height_ratios": heights})
    for ax, (col, name), tab in zip(axes, specs, tables):
        share = tab.div(tab.sum(axis=1).replace(0, 1), axis=0) * 100
        sns.heatmap(share, annot=True, fmt=".0f", cmap="YlGnBu", vmin=0, vmax=100, linewidths=0.6,
                    linecolor="white", ax=ax, cbar_kws={"label": "% of that category's records"})
        ax.set_yticklabels([f"{i}  (n={tab.loc[i].sum()})" for i in tab.index], rotation=0)
        ax.set_xticklabels(order, rotation=30, ha="right")
        ax.set_xlabel("Fertilizer")
        ax.set_ylabel(name)
        ax.set_title(f"{name}: share of records receiving each fertilizer (rows sum to 100%)")
    _suptitle(fig, "Which fertilizers are used on which crops and soils?")
    _save(fig, "06_crop_soil_vs_fertilizer.png")

    best = None
    for (col, name), tab in zip(specs, tables):
        big = tab[tab.sum(axis=1) >= 3]
        if big.empty:
            continue
        shares = big.div(big.sum(axis=1), axis=0)
        cat, fert = shares.stack().idxmax()
        val = shares.loc[cat, fert]
        if best is None or val > best[0]:
            best = (val, name.lower(), cat, fert, int(big.loc[cat, fert]), int(big.loc[cat].sum()))
    if best is None:
        return "Not enough records per crop/soil category to compare."
    return (f"Strongest pairing (categories with >= 3 records): {best[1]} '{best[2]}' -> {best[3]} "
            f"({best[0]:.0%}, {best[4]} of {best[5]} records).")


# ----------------------------------------------------------------------------
# 07  Correlations with significance
# ----------------------------------------------------------------------------
def plot_correlation_significance(df):
    cols = config.NUMERIC_COLS
    n = len(cols)
    r = np.eye(n)
    p = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            try:
                rij, pij = stats.pearsonr(df[cols[i]], df[cols[j]])
            except ValueError:
                rij, pij = np.nan, np.nan
            r[i, j] = r[j, i] = rij
            p[i, j] = p[j, i] = pij
    annot = np.array([[f"{r[i, j]:.2f}{_stars(p[i, j]) if i != j else ''}" for j in range(n)]
                      for i in range(n)])
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=(8.2, 6.6))
    sns.heatmap(pd.DataFrame(r, index=cols, columns=cols), mask=mask, annot=annot, fmt="",
                cmap="RdBu_r", vmin=-1, vmax=1, center=0, square=True, linewidths=0.6,
                linecolor="white", ax=ax, cbar_kws={"label": "Pearson r", "shrink": 0.8})
    ax.set_title("* p<0.05   ** p<0.01   *** p<0.001   (association, not causation)")
    _suptitle(fig, "How are soil, climate and nutrient variables related?")
    _save(fig, "07_correlation_significance.png")
    best = max(((abs(r[i, j]), r[i, j], p[i, j], cols[i], cols[j])
                for i in range(n) for j in range(i + 1, n) if not np.isnan(r[i, j])),
               default=None)
    if best is None:
        return "Correlations could not be computed."
    return (f"Strongest relationship: {best[3]} vs {best[4]} (r = {best[1]:.2f}, "
            f"p = {best[2]:.3g}); {int(((p < config.ALPHA) & ~mask & ~np.eye(n, dtype=bool)).sum())} "
            "of 15 pairs are significant.")


# ----------------------------------------------------------------------------
# 08  PCA projection (biplot)
# ----------------------------------------------------------------------------
def plot_pca_projection(df):
    feats = config.NUMERIC_COLS
    order, colors = _order(df), _colors(df)
    X = df[feats].to_numpy(float)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    Z = (X - X.mean(axis=0)) / sd
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    scores = U[:, :2] * S[:2]
    var = S ** 2 / (S ** 2).sum()
    load = Vt[:2].T                                    # features x 2

    fig, ax = plt.subplots(figsize=(10.5, 7))
    labels = df[T].to_numpy()
    for c in order:
        m = labels == c
        ax.scatter(scores[m, 0], scores[m, 1], s=46, color=colors[c], alpha=0.8,
                   edgecolor="white", linewidth=0.6, zorder=3)
        ax.scatter(scores[m, 0].mean(), scores[m, 1].mean(), s=150, marker="D", color=colors[c],
                   edgecolor="black", linewidth=1.3, zorder=4)
    scale = 0.85 * np.abs(scores).max() / max(np.abs(load).max(), 1e-9)
    for name, (a, b) in zip(feats, load):
        ax.annotate("", xy=(a * scale, b * scale), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="#333333", lw=1.2), zorder=5)
        ax.text(a * scale * 1.07, b * scale * 1.07, name, fontsize=9, color="#333333",
                ha="center", va="center", zorder=6)
    ax.axhline(0, color="#BBBBBB", linewidth=0.8)
    ax.axvline(0, color="#BBBBBB", linewidth=0.8)
    ax.set_xlabel(f"PC1 ({var[0]:.0%} of variance)")
    ax.set_ylabel(f"PC2 ({var[1]:.0%} of variance)")
    handles = [Line2D([], [], marker="o", linestyle="", color=colors[c], markersize=8,
                      label=f"{c} (n={int((labels == c).sum())})") for c in order]
    handles.append(Line2D([], [], marker="D", linestyle="", markerfacecolor="white",
                          markeredgecolor="black", markersize=8, label="Class centroid"))
    ax.legend(handles=handles, title="Fertilizer", bbox_to_anchor=(1.02, 1), loc="upper left",
              frameon=False)
    ax.set_title("Records projected onto the first two principal components (arrows = feature loadings)")
    _suptitle(fig, "Multivariate separability of fertilizer classes (PCA of standardised features)")
    _save(fig, "08_pca_projection.png")
    f1, f2 = feats[int(np.abs(load[:, 0]).argmax())], feats[int(np.abs(load[:, 1]).argmax())]
    return (f"PC1 + PC2 explain {var[:2].sum():.0%} of variance; PC1 is driven mainly by {f1}, "
            f"PC2 by {f2}.")


# ----------------------------------------------------------------------------
# 09  Confusion matrix (recommender performance)
# ----------------------------------------------------------------------------
def plot_confusion_matrix(cm, accuracy):
    m = cm.to_numpy()
    support = m.sum(axis=1)
    recall = np.divide(m, support[:, None], out=np.zeros(m.shape, dtype=float),
                       where=support[:, None] > 0)
    annot = np.array([[f"{m[i, j]}\n{recall[i, j]:.0%}" if m[i, j] else ""
                       for j in range(m.shape[1])] for i in range(m.shape[0])])
    baseline = support.max() / m.sum() if m.sum() else 0.0

    fig, ax = plt.subplots(figsize=(1.0 * m.shape[1] + 4.5, 0.85 * m.shape[0] + 3.2))
    sns.heatmap(pd.DataFrame(recall, index=cm.index, columns=cm.columns), annot=annot, fmt="",
                cmap="Blues", vmin=0, vmax=1, linewidths=0.6, linecolor="white", ax=ax,
                cbar_kws={"label": "Share of actual class (recall)"})
    ax.set_xlabel("Predicted fertilizer")
    ax.set_ylabel("Actual fertilizer")
    ax.set_xticklabels(cm.columns, rotation=35, ha="right")
    ax.set_yticklabels(cm.index, rotation=0)
    ax.set_title(f"Held-out test set: accuracy {accuracy:.1%} vs majority-class baseline "
                 f"{baseline:.1%}  (cell = count, % of actual class)")
    _suptitle(fig, "How well does the nearest-neighbour recommender perform?")
    _save(fig, "09_confusion_matrix.png")
    off = m.copy()
    np.fill_diagonal(off, 0)
    if off.max() == 0:
        return f"No misclassifications on the test set (accuracy {accuracy:.1%}, baseline {baseline:.1%})."
    i, j = np.unravel_index(off.argmax(), off.shape)
    return (f"Accuracy {accuracy:.1%} vs baseline {baseline:.1%}; most frequent error: actual "
            f"'{cm.index[i]}' predicted as '{cm.columns[j]}' ({off[i, j]} time(s)).")