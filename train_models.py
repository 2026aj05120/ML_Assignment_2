"""
Step 1 of the project: train the 5 classifiers and save everything the app needs.

Usage (from the project folder):
    python train_models.py --data your_dataset.csv --target YOUR_TARGET_COLUMN
Optional:
    --drop id_column other_useless_column     (columns to ignore, e.g. IDs)
    --test_size 0.2   --seed 42
"""
import argparse
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from ml_utils import compute_metrics, METRIC_COLUMNS

MODEL_DIR = "model"


def build_preprocessor(X):
    """Numeric cols: fill missing + scale. Text cols: fill missing + one-hot encode."""
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("scale", StandardScaler())])
    cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                         ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    transformers = [("num", num_pipe, num_cols)]
    if cat_cols:
        transformers.append(("cat", cat_pipe, cat_cols))
    return ColumnTransformer(transformers)


def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": SVC(kernel="rbf", probability=True, random_state=42),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--drop", nargs="*", default=[])
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    df = df.drop(columns=args.drop, errors="ignore")
    df = df.dropna(subset=[args.target]).drop_duplicates().reset_index(drop=True)
    print(f"Dataset: {df.shape[0]} rows, {df.shape[1] - 1} features")

    feature_cols = [c for c in df.columns if c != args.target]
    le = LabelEncoder().fit(df[args.target])
    class_names = [str(c) for c in le.classes_]
    n_classes = len(class_names)
    print("Classes:", class_names)

    train_df, test_df = train_test_split(df, test_size=args.test_size,
                                         random_state=args.seed, stratify=df[args.target])
    X_train, y_train = train_df[feature_cols], le.transform(train_df[args.target])
    X_test, y_test = test_df[feature_cols], le.transform(test_df[args.target])

    os.makedirs(MODEL_DIR, exist_ok=True)
    rows, files = [], {}
    for name, clf in get_models().items():
        pipe = Pipeline([("prep", build_preprocessor(X_train)), ("clf", clf)])
        pipe.fit(X_train, y_train)
        m = compute_metrics(y_test, pipe.predict(X_test), pipe.predict_proba(X_test), n_classes)
        rows.append({"ML Model Name": name, **m})
        fname = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(pipe, os.path.join(MODEL_DIR, fname))
        files[name] = fname
        print(f"trained {name}")

    joblib.dump({"feature_cols": feature_cols, "target": args.target,
                 "label_encoder": le, "class_names": class_names, "model_files": files},
                os.path.join(MODEL_DIR, "meta.joblib"))

    test_df.to_csv("test_data.csv", index=False)          # upload this file in the app
    res = pd.DataFrame(rows)
    res.to_csv("model_comparison.csv", index=False)
    print("\n=== Comparison table (copy into README.md) ===")
    print("| ML Model Name | " + " | ".join(METRIC_COLUMNS) + " |")
    print("|---|" + "---|" * len(METRIC_COLUMNS))
    for r in rows:
        print(f"| {r['ML Model Name']} | " + " | ".join(f"{r[c]:.4f}" for c in METRIC_COLUMNS) + " |")
    print("\nSaved: model/*.joblib, test_data.csv, model_comparison.csv")


if __name__ == "__main__":
    main()
