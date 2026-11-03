"""Fertilizer Recommendation System - run with:  python3 main.py

Pipeline: Data Input -> Cleaning -> Manipulation -> Statistics ->
          Visualisation -> Recommendation (terminal)

Optional environment variables:
    SHOW_FIGURES=1   also open the figures on screen (needs a display)
"""
import os

# Choose the Matplotlib backend BEFORE pyplot is imported anywhere.
SHOW_FIGURES = os.environ.get("SHOW_FIGURES", "0") == "1"
if not SHOW_FIGURES:
    os.environ["MPLBACKEND"] = "Agg"          # save PNGs only, no pop-up windows

# Always work relative to this file so 'python3 main.py' works from any folder.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt  # noqa: E402

from src import config  # noqa: E402
from src import data_analysis, data_cleaning, data_loading  # noqa: E402
from src import recommendation, statistical_analysis, visualization  # noqa: E402
from src.report import heading, log, save_report, subheading  # noqa: E402


def make_figures(raw_std, clean, summary, cm, accuracy):
    subheading("Generating figures (saved in results/figures/)")
    visualization.clear_old_figures()          # remove PNGs from earlier runs
    jobs = [
        ("Fertilizer class profiles", lambda: visualization.plot_class_profile(clean)),
        ("NPK distributions by fertilizer", lambda: visualization.plot_nutrient_distributions(clean)),
        ("Nutrient space (N vs P, K as size)", lambda: visualization.plot_nutrient_space(clean)),
        ("Environment by fertilizer", lambda: visualization.plot_environment_by_class(clean)),
        ("Variable importance (effect sizes)", lambda: visualization.plot_variable_importance(clean)),
        ("Crop & soil vs fertilizer", lambda: visualization.plot_categorical_associations(clean)),
        ("Correlation with significance", lambda: visualization.plot_correlation_significance(clean)),
        ("PCA projection", lambda: visualization.plot_pca_projection(clean)),
        ("Confusion matrix", lambda: visualization.plot_confusion_matrix(cm, accuracy)),
    ]
    for i, (name, job) in enumerate(jobs, 1):
        try:
            observation = job()
            log(f"[{i:02d}] {name}: {observation}")
        except Exception as exc:  # keep going if one plot fails, but say so
            log(f"[{i:02d}] {name}: FAILED - {type(exc).__name__}: {exc}")


def main():
    heading("STEP 1: DATA INPUT AND INSPECTION")
    try:
        raw = data_loading.load_csv(config.DATA_PATH)
        log(f"Loaded {config.DATA_PATH}")
        log(f"Original column names: {list(raw.columns)}")
        raw_std = data_loading.standardize_columns(raw)     # copy; 'raw' is untouched
        data_loading.check_required_columns(raw_std)
    except (FileNotFoundError, ValueError) as exc:
        print(f"\nERROR: {exc}")
        return 1
    log(f"Standardised column names: {list(raw_std.columns)}")
    data_loading.inspect_dataset(raw_std)

    heading("STEP 2: DATA CLEANING")
    clean, summary = data_cleaning.clean_data(raw_std)
    data_cleaning.print_cleaning_summary(summary)
    data_cleaning.save_cleaned(clean, config.PROCESSED_PATH)
    if len(clean) < 30:
        log(f"\nCAUTION: only {len(clean)} rows remain; conclusions will be weak.")

    heading("STEP 3: DATA MANIPULATION AND EXPLORATORY ANALYSIS")
    data_analysis.manipulation_demo(clean)
    data_analysis.eda_findings(clean)

    heading("STEP 4: STATISTICAL ANALYSIS")
    statistical_analysis.run_all_statistics(clean)

    heading("STEP 5: RECOMMENDATION MODEL (nearest neighbour, from scratch)")
    classifier, cm, accuracy = recommendation.run_evaluation(clean)

    heading("STEP 6: DATA VISUALISATION")
    visualization.set_keep_open(SHOW_FIGURES)
    make_figures(raw_std, clean, summary, cm, accuracy)

    save_report(config.REPORT_PATH)
    log(f"\nFull text report saved to {config.REPORT_PATH}")
    if SHOW_FIGURES:
        plt.show()

    heading("STEP 7: INTERACTIVE FERTILIZER RECOMMENDATION")
    try:
        answer = input("Start the interactive recommender now? (y/n): ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = "n"
    if answer in ("y", "yes"):
        recommendation.interactive_recommender(classifier, clean)
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
