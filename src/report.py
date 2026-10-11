"""Tiny logging helper: prints to the terminal AND keeps the text so it can be
saved to results/analysis_report.txt using plain file handling."""
import os

_LINES = []


def log(text=""):
    text = str(text)
    print(text)
    _LINES.append(text)


def heading(title):
    log("")
    log("=" * 78)
    log(title)
    log("=" * 78)


def subheading(title):
    log("")
    log("--- " + title + " ---")


def save_report(path):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(_LINES) + "\n")
