# train.py
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    RocCurveDisplay,
    PrecisionRecallDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from joblib import dump

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DATA_FILE = "heart.csv"   # if missing, we auto-generate synthetic data
MODEL_FILE = "model.joblib"
PREPROCESSOR_FILE = "preprocessor.joblib"
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

# ---------------------------
# 1) load or synthesize data
# ---------------------------
def synthesize_heart_dataset(n=1025, random_state=RANDOM_STATE):
    """
    Create a synthetic 'heart.csv'-style dataset with similar schema
    to the UCI Cleveland dataset. This lets you run the project
    even without downloading a real CSV.
    """
    rng = np.random.default_rng(random_state)

    age = rng.integers(29, 78, n)
    sex = rng.choice([0, 1], size=n, p=[0.35, 0.65])
    cp = rng.choice([0, 1, 2, 3], size=n, p=[0.1, 0.35, 0.4, 0.15])                 # chest pain type
    trestbps = rng.normal(130, 17, n).clip(90, 200).round().astype(int)             # resting bp
    chol = rng.normal(245, 50, n).clip(120, 600).round().astype(int)                # cholesterol
    fbs = (rng.uniform(0, 1, n) < 0.15).astype(int)                                 # fasting blood sugar > 120 mg/dl
    restecg = rng.choice([0, 1, 2], size=n, p=[0.65, 0.3, 0.05])                    # resting ECG
    thalach = rng.normal(150, 22, n).clip(70, 210).round().astype(int)              # max heart rate
    exang = (rng.uniform(0, 1, n) < 0.32).astype(int)                                # exercise induced angina
    oldpeak = np.abs(rng.normal(1.0, 1.1, n)).round(2)                               # ST depression
    slope = rng.choice([0, 1, 2], size=n, p=[0.35, 0.5, 0.15])                       # slope of peak exercise ST
    ca = rng.choice([0, 1, 2, 3, 4], size=n, p=[0.6, 0.2, 0.12, 0.06, 0.02])         # num major vessels (0–3); allow 4 rarely
    thal = rng.choice([0, 1, 2, 3], size=n, p=[0.02, 0.1, 0.6, 0.28])                # 0=unknown,1=fixed defect?,2=normal,3=reversible defect (varies by corpus)

    # outcome probability shaped by risk factors (not medical advice!)
    logit = (
        -5.0
        + 0.04 * (age - 50)
        + 0.015 * (trestbps - 120)
        + 0.008 * (chol - 200)
        + 0.03 * (150 - thalach)
        + 0.5 * exang
        + 0.35 * (oldpeak - 1.0)
        + 0.3 * (slope == 0)
        + 0.35 * (ca >= 1)
        + 0.25 * (thal == 3)
        + 0.15 * (cp == 3) * -1  # typical angina slightly protective relative to asymptomatic
        + 0.1 * sex
        + 0.1 * fbs
    )
    p = 1 / (1 + np.exp(-logit))
    target = (rng.uniform(0, 1, n) < p).astype(int)

    df = pd.DataFrame({
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
        "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal, "target": target
    })
    return df

def load_data():
    if os.path.exists(DATA_FILE):
        print(f"[INFO] Loading dataset from {DATA_FILE}")
        df = pd.read_csv(DATA_FILE)
    else:
        print(f"[WARN] {DATA_FILE} not found. Generating a synthetic dataset so you can run end-to-end.")
        df = synthesize_heart_dataset()
        df.to_csv(DATA_FILE, index=False)
        print(f"[INFO] Synthetic dataset saved to {DATA_FILE} ({len(df)} rows).")
    return df

df = load_data()

# basic schema sanity
expected_cols = {
    "age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal","target"
}
missing = expected_cols - set(df.columns)
if missing:
    raise ValueError(f"Dataset missing expected columns: {missing}")

# ---------------------------
# 2) quick EDA (saved as png)
# ---------------------------
print("[INFO] Running quick EDA (plots in ./reports)")
plt.figure()
sns.countplot(x="target", data=df)
plt.title("Target distribution (0 = no disease, 1 = disease)")
plt.savefig(REPORTS_DIR/"target_distribution.png", bbox_inches="tight"); plt.close()

plt.figure()
sns.histplot(df["age"], bins=25, kde=True)
plt.title("Age distribution")
plt.savefig(REPORTS_DIR/"age_distribution.png", bbox_inches="tight"); plt.close()

plt.figure()
corr = df.corr(numeric_only=True)
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Correlation heatmap (numeric)")
plt.savefig(REPORTS_DIR/"corr_heatmap.png", bbox_inches="tight"); plt.close()

# ---------------------------
# 3) train/validation split
# ---------------------------
X = df.drop(columns=["target"])
y = df["target"].astype(int)

numeric_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]
categorical_features = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

# ---------------------------
# 4) preprocessing pipeline
# ---------------------------
numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# ---------------------------
# 5) baseline models
# ---------------------------
models = {
    "logreg": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
    "rf": RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_split=4, random_state=RANDOM_STATE, n_jobs=-1
    ),
    "gb": GradientBoostingClassifier(random_state=RANDOM_STATE)
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

cv_results = {}
for name, clf in models.items():
    pipe = Pipeline(steps=[("pre", preprocessor), ("clf", clf)])
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    cv_results[name] = (scores.mean(), scores.std())
    print(f"[CV] {name}: ROC AUC = {scores.mean():.3f} ± {scores.std():.3f}")

# ---------------------------
# 6) choose/tune the best
# ---------------------------
# pick the best by mean CV AUC
best_name = max(cv_results, key=lambda k: cv_results[k][0])

print(f"[INFO] Best baseline appears to be: {best_name}")

if best_name == "logreg":
    param_grid = {
        "clf__C": [0.1, 0.5, 1.0, 2.0, 5.0],
        "clf__penalty": ["l2"],
        "clf__solver": ["lbfgs"]
    }
    base = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
elif best_name == "rf":
    param_grid = {
        "clf__n_estimators": [300, 500, 800],
        "clf__max_depth": [None, 6, 10, 16],
        "clf__min_samples_split": [2, 4, 6]
    }
    base = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
else:  # gb
    param_grid = {
        "clf__n_estimators": [150, 250, 350],
        "clf__learning_rate": [0.03, 0.05, 0.1],
        "clf__max_depth": [2, 3, 4]
    }
    base = GradientBoostingClassifier(random_state=RANDOM_STATE)

pipe = Pipeline(steps=[("pre", preprocessor), ("clf", base)])

grid = GridSearchCV(
    pipe,
    param_grid=param_grid,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1,
    verbose=0
)
grid.fit(X_train, y_train)
print(f"[TUNE] Best CV ROC AUC: {grid.best_score_:.3f}")
print("[TUNE] Best params:", grid.best_params_)

best_model = grid.best_estimator_

# ---------------------------
# 7) test-set evaluation
# ---------------------------
proba = best_model.predict_proba(X_test)[:, 1]
preds = (proba >= 0.3).astype(int)

test_auc = roc_auc_score(y_test, proba)
print(f"[TEST] ROC AUC: {test_auc:.3f}")

print("\n[TEST] Classification report @ 0.50 threshold:\n")
print(classification_report(y_test, preds, digits=3))

cm = confusion_matrix(y_test, preds)
print("[TEST] Confusion matrix:\n", cm)

# plots: ROC + PR
RocCurveDisplay.from_predictions(y_test, proba)
plt.title("ROC Curve (Test)")
plt.savefig(REPORTS_DIR/"roc_curve.png", bbox_inches="tight"); plt.close()

PrecisionRecallDisplay.from_predictions(y_test, proba)
plt.title("Precision-Recall Curve (Test)")
plt.savefig(REPORTS_DIR/"pr_curve.png", bbox_inches="tight"); plt.close()


# ---------------------------
# 8) permutation importance
# ---------------------------
best_model.fit(X_train, y_train)

r = permutation_importance(
    best_model, X_test, y_test, n_repeats=15,
    random_state=RANDOM_STATE, n_jobs=-1
)

# extract transformed feature names
ohe = best_model.named_steps["pre"].named_transformers_["cat"].named_steps["onehot"]
cat_names = ohe.get_feature_names_out(categorical_features)
num_names = np.array(numeric_features)
feature_names = np.concatenate([num_names, cat_names])

# align lengths just in case
n_importances = len(r.importances_mean)
if len(feature_names) != n_importances:
    print(f"[WARN] Feature names length {len(feature_names)} "
          f"!= importances length {n_importances}. Truncating to match.")
    min_len = min(len(feature_names), n_importances)
    feature_names = feature_names[:min_len]
    importances_mean = r.importances_mean[:min_len]
    importances_std = r.importances_std[:min_len]
else:
    importances_mean = r.importances_mean
    importances_std = r.importances_std

imp = pd.DataFrame({
    "feature": feature_names,
    "importance_mean": importances_mean,
    "importance_std": importances_std
}).sort_values("importance_mean", ascending=False)

imp.to_csv(REPORTS_DIR/"permutation_importance.csv", index=False)


# ---------------------------
# 9) persist artifacts
# ---------------------------
dump(best_model, MODEL_FILE)
dump(preprocessor, PREPROCESSOR_FILE)
print(f"[SAVE] Saved best model to {MODEL_FILE} and preprocessor to {PREPROCESSOR_FILE}")

print("\n[DONE] Reports saved in ./reports:")
for p in sorted(REPORTS_DIR.glob("*.png")):
    print(" -", p)
print(" -", REPORTS_DIR/"permutation_importance.csv")


