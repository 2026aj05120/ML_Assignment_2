"""
Step 0: turn the raw "Global Salary Data by Skill & Role" CSV into a clean CLASSIFICATION dataset.

The raw file predicts a number (salary). The assignment needs classes, so we cut the
USD mid-point salary into three equally sized bands:  1-Low / 2-Mid / 3-High.

Usage:
    python prepare_dataset.py --raw skill_salary_percentiles_global_sample.csv --out salary_band_dataset.csv
"""
import argparse
import numpy as np
import pandas as pd

TOP_SKILLS = ["python", "sql", "java", "excel", "javascript", "aws", "project management"]
TITLE_WORDS = ["senior", "manager", "engineer", "developer", "analyst", "intern", "director"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", default="salary_band_dataset.csv")
    ap.add_argument("--n", type=int, default=12000, help="rows to keep (stratified sample)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.raw)
    print("raw shape:", df.shape)
    ## Drop duplicate rows and missing values present in existing columns
    df = df.drop_duplicates().dropna()
    print(f"Cleaned shape: {df.shape}")

    # Save to output file
    df.to_csv(args.out, index=False)
    print(f"Saved prepared dataset to {args.out}")
    return 

    # ---------- feature engineering ----------
    out = pd.DataFrame(index=df.index)
    # month = pd.to_datetime(df["posted_month"], format="%Y-%m", errors="coerce")
    # out["posted_year"] = month.dt.year
    # out["posted_month_num"] = month.dt.month

    ctry = df["country"].astype(str)
    rare = ctry.map(ctry.value_counts()) < 100            # merge tiny countries
    out["country"] = np.where(rare, "OTHER", ctry)
    out["region_group"] = df["region_group"].fillna("unknown")
    out["role_family"] = df["role_family"].fillna("unknown")
    out["seniority"] = df["seniority"].fillna("unknown")
    out["industry_category"] = df["industry_category"].fillna("unknown")

    if "skills" in df.columns:
     skills = df["skills"].fillna("").str.lower()
    # (Any existing skill processing code here)
    out["num_skills"] = skills.apply(lambda s: len(s.split("|")) if s else 0)
    out["has_skills_listed"] = (out["num_skills"] > 0).astype(int)
    for sk in TOP_SKILLS:
        out["skill_" + sk.replace(" ", "_")] = skills.apply(lambda s, k=sk: int(k in s.split("|")))

    title = df["job_title"].astype(str).str.lower()
    out["title_word_count"] = title.str.split().str.len()
    out["title_char_count"] = title.str.len()
    for w in TITLE_WORDS:
        out["title_has_" + w] = title.str.contains(w, regex=False).astype(int)
    out["has_company_domain"] = df["company_domain"].notna().astype(int)

    out["salary_band"] = df["salary_band"]

    # ---------- stratified sample (keeps SVM/KNN fast and the Streamlit free tier happy) ----------
    if len(out) > args.n:
        out = out.groupby("salary_band", group_keys=False).sample(frac=args.n / len(out), random_state=args.seed)
    out = out.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    out.to_csv(args.out, index=False)
    print(f"saved {args.out}: {out.shape[0]} rows, {out.shape[1] - 1} features")
    print(out["salary_band"].value_counts())


if __name__ == "__main__":
    main()
