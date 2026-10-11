"""Step 6 - Fertilizer recommendation: k-Nearest-Neighbours written FROM SCRATCH.

Only Python, NumPy and Pandas are used. This is NOT scikit-learn, NOT a
Random Forest and NOT a Decision Tree.

How a prediction is made
------------------------
1. Numerical features (6) are min-max scaled to [0, 1] using the TRAINING
   minimum and maximum:  (x - min) / (max - min). If max == min the divisor
   is replaced by 1, so there is never a division by zero.
2. Categorical features (Soil_Type, Crop_Type) cannot be subtracted, so each
   contributes a mismatch count: 0 if the category is the same, 1 if not.
3. Distance (Euclidean-style):
        d = sqrt( sum_over_numeric (a - b)^2  +  cat_weight * mismatches )
4. The k closest training records vote. The fertilizer with most votes wins;
   a tie is broken by the smaller total distance of that fertilizer's voters.
5. k is chosen from config.K_CANDIDATES using leave-one-out accuracy on the
   TRAINING set only, so the test set is never used to tune anything.
"""
import numpy as np
import pandas as pd

from src import config
from src.report import log, subheading


# ----------------------------------------------------------------------------
# Train / test split (manual, stratified by fertilizer class)
# ----------------------------------------------------------------------------
def train_test_split(df, test_fraction, seed):
    rng = np.random.default_rng(seed)
    test_idx = []
    for _, sub in df.groupby(config.TARGET):
        idx = sub.index.to_numpy().copy()
        rng.shuffle(idx)
        n_test = int(round(test_fraction * len(idx)))
        if len(idx) < 2:
            n_test = 0                       # a single example must stay in training
        n_test = min(n_test, len(idx) - 1)   # always keep >= 1 training example
        test_idx.extend(idx[:n_test].tolist())
    test_mask = df.index.isin(test_idx)
    return df[~test_mask].reset_index(drop=True), df[test_mask].reset_index(drop=True)


# ----------------------------------------------------------------------------
# Classifier
# ----------------------------------------------------------------------------
class NearestNeighbourClassifier:
    def __init__(self, numeric_cols, categorical_cols, target, k=3, cat_weight=1.0):
        self.numeric_cols = list(numeric_cols)
        self.categorical_cols = list(categorical_cols)
        self.target = target
        self.k = k
        self.cat_weight = cat_weight

    def fit(self, train_df):
        X = train_df[self.numeric_cols].to_numpy(dtype=float)
        self.min_ = X.min(axis=0)
        span = X.max(axis=0) - self.min_
        self.span_ = np.where(span == 0, 1.0, span)          # avoid division by zero
        self.Xn_ = (X - self.min_) / self.span_
        self.Xc_ = train_df[self.categorical_cols].to_numpy(dtype=object)
        self.y_ = train_df[self.target].to_numpy(dtype=object)
        self.classes_ = sorted(set(self.y_.tolist()))
        return self

    def _scale(self, df):
        return (df[self.numeric_cols].to_numpy(dtype=float) - self.min_) / self.span_

    def _distances_arrays(self, qn, qc):
        """Distances between scaled numeric rows qn and categorical rows qc
        and every training record (n_queries x n_train)."""
        numeric_sq = ((qn[:, None, :] - self.Xn_[None, :, :]) ** 2).sum(axis=2)
        mismatches = (qc[:, None, :] != self.Xc_[None, :, :]).sum(axis=2)
        return np.sqrt(numeric_sq + self.cat_weight * mismatches)

    def _distances(self, df):
        qc = df[self.categorical_cols].to_numpy(dtype=object)
        return self._distances_arrays(self._scale(df), qc)

    def _vote(self, d_row, k):
        order = np.argsort(d_row, kind="stable")[:k]
        tally = {}
        for i in order:
            label = self.y_[i]
            count, dist = tally.get(label, (0, 0.0))
            tally[label] = (count + 1, dist + float(d_row[i]))
        winner = sorted(tally.items(), key=lambda kv: (-kv[1][0], kv[1][1], kv[0]))[0][0]
        return winner, tally, order

    def predict(self, df, k=None):
        k = self.k if k is None else k
        D = self._distances(df)
        return [self._vote(row, k)[0] for row in D]

    def explain_one(self, query_df, k=None):
        """Prediction for ONE query row plus the neighbours that decided it."""
        k = self.k if k is None else k
        d_row = self._distances(query_df)[0]
        winner, tally, order = self._vote(d_row, k)
        neighbours = pd.DataFrame({
            "distance": d_row[order].round(3),
            "fertilizer": self.y_[order],
        })
        for j, col in enumerate(self.categorical_cols):
            neighbours[col] = self.Xc_[order, j]
        for j, col in enumerate(self.numeric_cols):
            neighbours[col] = (self.Xn_[order, j] * self.span_[j] + self.min_[j]).round(1)
        votes = {lab: c for lab, (c, _) in tally.items()}
        return winner, votes, neighbours

    def select_k(self, candidates):
        """Leave-one-out accuracy on the training data for each candidate k."""
        D = self._distances_arrays(self.Xn_, self.Xc_)
        np.fill_diagonal(D, np.inf)                           # a record may not vote for itself
        n = len(self.y_)
        results = {}
        for k in candidates:
            if k > n - 1:
                continue
            hits = sum(self._vote(D[i], k)[0] == self.y_[i] for i in range(n))
            results[k] = hits / n
        if not results:
            return self.k, {}
        best_acc = max(results.values())
        best_k = min(k for k, a in results.items() if a == best_acc)
        self.k = best_k
        return best_k, results


# ----------------------------------------------------------------------------
# Evaluation metrics (manual)
# ----------------------------------------------------------------------------
def confusion_matrix(y_true, y_pred, classes):
    pos = {c: i for i, c in enumerate(classes)}
    m = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(y_true, y_pred):
        m[pos[t], pos[p]] += 1
    return pd.DataFrame(m, index=classes, columns=classes)


def classification_report(cm):
    m = cm.to_numpy()
    tp = np.diag(m).astype(float)
    support = m.sum(axis=1)
    predicted = m.sum(axis=0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    table = pd.DataFrame({"precision": precision, "recall": recall, "f1": f1,
                          "support": support.astype(int)}, index=cm.index)
    accuracy = tp.sum() / m.sum() if m.sum() else 0.0
    present = support > 0
    macro = {"precision": precision[present].mean(), "recall": recall[present].mean(),
             "f1": f1[present].mean()}
    return table, float(accuracy), macro


def evaluate(df, seed):
    """Split, choose k on training data, score on the held-out test set."""
    train, test = train_test_split(df, config.TEST_FRACTION, seed)
    clf = NearestNeighbourClassifier(config.NUMERIC_COLS, config.CATEGORICAL_COLS, config.TARGET)
    clf.fit(train)
    best_k, loo = clf.select_k(config.K_CANDIDATES)
    pred = clf.predict(test)
    cm = confusion_matrix(test[config.TARGET].tolist(), pred, clf.classes_)
    return clf, train, test, cm, best_k, loo


def run_evaluation(df):
    """Full evaluation, printed. Returns (classifier, confusion_matrix, accuracy)."""
    subheading("Train/test split and k selection")
    clf, train, test, cm, best_k, loo = evaluate(df, config.RANDOM_SEED)
    log(f"Training rows: {len(train)}   Test rows: {len(test)}   (seed = {config.RANDOM_SEED}, stratified)")
    if loo:
        log("Leave-one-out accuracy on training data for each k: " +
            ", ".join(f"k={k}: {a:.3f}" for k, a in loo.items()))
    log(f"Selected k = {best_k}")

    report, acc, macro = classification_report(cm)
    majority = df[config.TARGET].value_counts(normalize=True).iloc[0]
    subheading("Test-set results (manual metrics)")
    log(f"Accuracy: {acc:.3f}  ({int(np.trace(cm.to_numpy()))}/{int(cm.to_numpy().sum())} correct)")
    log(f"Baseline (always predict the most common class): {majority:.3f}")
    log(f"Macro precision {macro['precision']:.3f}   recall {macro['recall']:.3f}   F1 {macro['f1']:.3f}")
    log(report.round(3).to_string())
    log("(precision is reported as 0 for a class that was never predicted)")
    log("\nConfusion matrix (rows = actual, columns = predicted):")
    log(cm.to_string())

    subheading(f"Robustness: {config.REPEATED_SPLITS} repeated random splits")
    accs = []
    for s in range(config.REPEATED_SPLITS):
        _, _, _, cm_s, _, _ = evaluate(df, config.RANDOM_SEED + 1 + s)
        accs.append(classification_report(cm_s)[1])
    log(f"Accuracy per split: {[round(a, 3) for a in accs]}")
    log(f"Mean {np.mean(accs):.3f}  std {np.std(accs, ddof=1):.3f}  min {min(accs):.3f}  max {max(accs):.3f}")
    if len(test) < 30:
        log(f"CAUTION: the test set has only {len(test)} rows, so one split's accuracy is very noisy; "
            "trust the repeated-split average more than a single number.")

    # Final model for the interactive tool: fit on ALL cleaned rows, reuse chosen k
    final = NearestNeighbourClassifier(config.NUMERIC_COLS, config.CATEGORICAL_COLS, config.TARGET, k=best_k)
    final.fit(df)
    return final, cm, acc


# ----------------------------------------------------------------------------
# Terminal interface
# ----------------------------------------------------------------------------
class _Quit(Exception):
    pass


def _ask_number(label, lo, hi, observed):
    while True:
        try:
            text = input(f"{label} [{observed[0]:g}-{observed[1]:g} seen in data] (q to quit): ").strip()
        except (EOFError, KeyboardInterrupt):
            raise _Quit()
        if text.lower() in ("q", "quit", "exit"):
            raise _Quit()
        try:
            value = float(text)
        except ValueError:
            print("  Please enter a number (e.g. 26 or 26.5).")
            continue
        if np.isnan(value) or np.isinf(value):
            print("  That is not a usable number.")
            continue
        if (lo is not None and value < lo) or (hi is not None and value > hi):
            print(f"  Value must be between {lo if lo is not None else '-inf'} and {hi if hi is not None else 'inf'}.")
            continue
        if value < observed[0] or value > observed[1]:
            print("  Note: outside the range seen in the training data; the prediction is an extrapolation.")
        return value


def _ask_category(label, options):
    print(f"{label} options:")
    for i, name in enumerate(options, 1):
        print(f"  {i}. {name}")
    lookup = {name.lower(): name for name in options}
    while True:
        try:
            text = input(f"{label} (number or name, q to quit): ").strip()
        except (EOFError, KeyboardInterrupt):
            raise _Quit()
        if text.lower() in ("q", "quit", "exit"):
            raise _Quit()
        if text.isdigit() and 1 <= int(text) <= len(options):
            return options[int(text) - 1]
        if text.lower() in lookup:
            return lookup[text.lower()]
        print("  Not a valid option; choose from the list.")


def interactive_recommender(clf, df):
    log("\nHow it works: scaled distance on 6 numeric features + 1 mismatch per differing "
        f"soil/crop; the {clf.k} closest records vote.")
    log("Type q at any prompt to leave. The result is a data-based suggestion, not agronomic advice.")
    seen = {c: (float(df[c].min()), float(df[c].max())) for c in config.NUMERIC_COLS}
    soils = sorted(x for x in df["Soil_Type"].unique() if x != "Unknown")
    crops = sorted(x for x in df["Crop_Type"].unique() if x != "Unknown")
    order = ["Nitrogen", "Phosphorus", "Potassium", "Temperature", "Humidity", "Moisture"]
    while True:
        print("\n--- Enter field details ---")
        try:
            row = {}
            for col in order:
                lo, hi = config.VALID_RANGES[col]
                row[col] = _ask_number(col, lo, hi, seen[col])
            row["Soil_Type"] = _ask_category("Soil type", soils)
            row["Crop_Type"] = _ask_category("Crop type", crops)
        except _Quit:
            print("\nLeaving the recommender.")
            return
        query = pd.DataFrame([row])
        winner, votes, neighbours = clf.explain_one(query)
        print(f"\nRecommended fertilizer: {winner}")
        print(f"Votes among the {clf.k} nearest records: {votes}")
        print("Nearest records that decided it:")
        print(neighbours.to_string(index=False))
        print("Caution: based on similarity to past records only; check with local agronomy advice.")
        try:
            again = input("\nAnother recommendation? (y/n): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return
        if again not in ("y", "yes"):
            return
