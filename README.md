# Social Network Ads Purchase Prediction
### Decision Tree Classifier | Scikit-learn | GridSearchCV

## Project overview
This project builds a Decision Tree classification model to predict whether a user purchases a product after seeing a social network advertisement. The model uses two input features: **Age** and **EstimatedSalary**. The target is **Purchased** (`0` = No, `1` = Yes).

The main objective is to compare an unrestricted baseline Decision Tree with a hyperparameter-tuned Decision Tree and evaluate the selected model on an untouched test set.

## Problem statement
Given a user's age and estimated salary, predict whether that user is likely to purchase the advertised product.

## Dataset
- File expected: `data/Social_Network_Ads.csv`
- Dataset size in the original notebook: **400 rows × 5 columns**
- Features used: `Age`, `EstimatedSalary`
- Target: `Purchased`
- Original dataset source: Kaggle — Social Network Ads dataset. Find the dataset at [Kaggle](https://www.kaggle.com/datasets/rakeshrau/social-network-ads).

The dataset is not bundled in this repository. Download it from the source, check its usage terms, and place `Social_Network_Ads.csv` inside the `data/` directory. See [`data/README.md`](data/README.md).

## Repository structure
```text
Social-Network-Ads-Purchase-Prediction/
│
├── data/
│   └── README.md
│
├── notebooks/
│   └── decision_tree_gridsearchcv.ipynb
│
├── requirements.txt
│
└── README.md
```

## Methodology
1. **Load and validate data** — load the CSV and check required columns.
2. **Explore the data** — inspect shape, missing values, target distribution, and feature relationships.
3. **Select variables** — use `Age` and `EstimatedSalary` as predictors and `Purchased` as the target.
4. **Train/test split** — reserve 20% of the data for final evaluation and use stratification to preserve class proportions.
5. **Baseline model** — train a default `DecisionTreeClassifier` and inspect training performance, depth, and leaf count.
6. **Cross-validation** — estimate baseline performance with 5-fold `StratifiedKFold`.
7. **Hyperparameter tuning** — use `GridSearchCV` to search over:
   - `criterion`: `gini`, `entropy`
   - `max_depth`: `2`, `3`, `4`, `5`, `None`
   - `min_samples_leaf`: `1`, `2`, `4`, `6`
8. **Model selection** — select the estimator with the highest mean cross-validation accuracy.
9. **Final evaluation** — evaluate once on the held-out test set using accuracy, precision, recall, F1-score, and a confusion matrix.
10. **Interpretation** — visualize the selected tree and inspect feature importances.

## Results
The following values are copied from the **original notebook's saved outputs**. The revised notebook uses shuffled, stratified folds, so its results may differ when rerun.

| Metric | Original notebook result |
|---|---:|
| Baseline training accuracy | 99.4% |
| Baseline mean CV accuracy | 87.2% |
| Best mean CV accuracy | 91.9% |
| Best parameters | `criterion=gini`, `max_depth=2`, `min_samples_leaf=1` |
| Best tree depth | 2 |
| Best tree leaves | 4 |
| Test accuracy | 88.7% |

Original test-set class metrics:

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| No Purchase | 0.94 | 0.88 | 0.91 |
| Purchased | 0.81 | 0.90 | 0.85 |

**Interpretation:** The unrestricted baseline has very high training accuracy (99.4%) but lower cross-validation accuracy (87.2%), which is a warning sign of overfitting. Grid search selected a shallow tree and achieved 91.9% mean CV accuracy in the original run. The original held-out test accuracy was 88.7%. Cross-validation and test metrics measure different things; the test score is the final estimate of generalization on this split.

## Run locally
### 1. Clone the repository
```bash
git clone https://github.com/Udayrajchothave/Social-Network-Ads-Purchase-Prediction
cd decision-tree-gridsearchcv
```

### 2. Create and activate a virtual environment
```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add the dataset
Download `Social_Network_Ads.csv` from the Kaggle source and put it in `data/`.

### 5. Launch Jupyter
```bash
jupyter notebook
```
Open `notebooks/decision_tree_gridsearchcv.ipynb` and run the cells from top to bottom.

## Engineering notes
- A fixed `random_state` is used for reproducibility.
- Stratification is used for the train/test split and cross-validation because the target classes are not evenly represented.
- Decision Trees do not require feature standardization, so scaling is intentionally omitted.
- Hyperparameter tuning is performed only on the training data. The test set is reserved for final evaluation.
- `n_jobs=-1` allows scikit-learn to use available CPU cores during cross-validation and grid search.
- The saved results in this README describe the original notebook run; rerunning the improved cross-validation setup can produce different values.

## Limitations and next steps
- The dataset is small (400 records), so performance estimates can vary with the split.
- Only two numerical features are used; `Gender` is not included in the model.
- Accuracy alone may hide class-specific errors. Review precision, recall, F1-score, and the confusion matrix.
- A next iteration could compare Decision Tree with Logistic Regression, Random Forest, and Gradient Boosting using the same split and cross-validation strategy.
- If this model were used in a real advertising system, validate data quality, fairness, privacy, and business costs before deployment.

## Tech stack
- Python
- pandas
- matplotlib
- scikit-learn
- Jupyter Notebook


