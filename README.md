# Fertilizer Recommendation System (Python, syllabus-restricted)

## 1. Objective
Show the pipeline **Data Input -> Cleaning -> Manipulation -> Statistics -> Visualisation -> Recommendation**
on the Kaggle *Fertilizer Prediction* dataset, using only the libraries in the course syllabus.

## 2. Dataset (you must download it)
Source: https://www.kaggle.com/datasets/gdabhishek/fertilizer-prediction
Save the CSV as **`data/fertilizer.csv`**. No dataset is bundled with this project.

Expected columns (the code reads the actual file; it standardises `Temparature -> Temperature`,
`Phosphorous -> Phosphorus`, and spaces to underscores):
`Temperature, Humidity, Moisture, Soil_Type, Crop_Type, Nitrogen, Potassium, Phosphorus, Fertilizer_Name` (target).
If any of these columns is missing the program stops with a clear message.

## 3. Libraries
Python standard library (`os`, `statistics`), NumPy, Pandas, Matplotlib, Seaborn, SciPy.
Not used: scikit-learn, TensorFlow/PyTorch, Streamlit/Flask/FastAPI, SQL, JavaScript.
Statsmodels is **not** used: the dataset has no time dimension, so time-series tools would be inappropriate.

## 4. Setup and run
```bash
pip install numpy pandas matplotlib seaborn scipy
# put the CSV at data/fertilizer.csv
python3 main.py
```
Optional: `SHOW_FIGURES=1 python3 main.py` also opens the plots on screen.
At the end the program asks `Start the interactive recommender now? (y/n)`.

## 5. Outputs
- Terminal output and `results/analysis_report.txt` (full text of every step)
- `data/processed/fertilizer_cleaned.csv` (cleaned data)
- `results/figures/01 ... 15 *.png`

## 6. What each step does
| File | Purpose |
|---|---|
| `src/data_loading.py` | load CSV, standardise names, inspect (head/tail, dtypes, describe, missing, duplicates, invalid values, class counts) |
| `src/data_cleaning.py` | work on a copy: trim text, invalid numbers -> missing, drop missing-target rows, drop exact duplicates, median imputation, `Unknown` for missing categories, IQR outlier report (outliers are kept), before/after summary |
| `src/data_analysis.py` | `loc`/`iloc`, filtering, sorting, derived columns (`NPK_Total`, `Temp_Band`), groupby/aggregation, frequency tables, add/delete rows and columns; EDA findings |
| `src/statistical_analysis.py` | mean/median/mode/variance/std/quartiles/IQR/skew, Shapiro-Wilk, Pearson & Spearman, Levene, ANOVA or Kruskal-Wallis (chosen from the assumption checks), chi-square + Cramer's V (with expected-count warning) |
| `src/visualization.py` | the 15 required plots; each prints one data-derived observation |
| `src/recommendation.py` | nearest-neighbour classifier from scratch, manual metrics, confusion matrix, terminal interface |

## 7. Recommendation algorithm (k-nearest neighbours, from scratch)
1. Min-max scale the 6 numeric features with training min/max (zero range -> divisor 1).
2. Soil and crop type: distance contribution 1 if different, 0 if same.
3. `distance = sqrt(sum of squared scaled differences + number of categorical mismatches)`.
4. The k nearest training rows vote; ties go to the smaller total distance.
5. k is picked from {1,3,5,7,9} by leave-one-out accuracy on the training split only.
6. Evaluation: stratified 80/20 split (NumPy), accuracy, precision, recall, F1, confusion matrix, plus
   10 repeated random splits because a single small test set is noisy.
The interactive tool uses a model fitted on all cleaned rows.

## 8. Limitations
- It is a simple similarity method, **not** the Random Forest / Decision Tree named in the proposal. A
  proposal that requires those needs scikit-learn (or hand-written trees), which the syllabus does not allow;
  raise this with your professor.
- The dataset is small (about a hundred rows expected), so test accuracy has high variance and several
  chi-square tables will have expected counts below 5 (the program warns).
- Imputation uses medians of the whole cleaned set; scaling and k selection use training data only.
- Equal weight for numeric and categorical features is a simple assumption, not a tuned one.
- Output is a data-based suggestion, not verified agronomic advice.

## 9. Demonstration order for the professor
1. `python3 main.py`, show Step 1 inspection and Step 2 cleaning summary (before/after, duplicates, outliers kept).
2. Show Step 3 (loc/iloc, filtering, groupby) and Step 4 (test, hypotheses, p-value, interpretation).
3. Open `results/figures/` (class distribution, heatmap, nutrients by fertilizer, confusion matrix).
4. Show the accuracy, the baseline and the repeated-split result in Step 5.
5. Answer `y`, enter a field's values, and show the recommendation and the nearest records behind it.
