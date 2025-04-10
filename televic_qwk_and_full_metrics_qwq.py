
import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, cohen_kappa_score
from scipy.stats import pearsonr

# === QWK and Fisher transform ===
def quadratic_weighted_kappa(y_true, y_pred, min_rating=0, max_rating=10):
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    num_ratings = max_rating - min_rating + 1

    O = np.zeros((num_ratings, num_ratings))
    for a, b in zip(y_true, y_pred):
        O[a, b] += 1

    hist_true = np.bincount(y_true, minlength=num_ratings)
    hist_pred = np.bincount(y_pred, minlength=num_ratings)
    E = np.outer(hist_true, hist_pred)
    E = E / E.sum()

    W = np.zeros((num_ratings, num_ratings))
    for i in range(num_ratings):
        for j in range(num_ratings):
            W[i, j] = ((i - j) ** 2) / ((num_ratings - 1) ** 2)

    O = O / O.sum()
    num = (W * O).sum()
    den = (W * E).sum()
    return 1.0 - num / den if den != 0 else 1.0

def fisher_z(kappa):
    kappa = max(-0.999, min(0.999, kappa))
    return 0.5 * np.log((1 + kappa) / (1 - kappa))

def inverse_fisher_z(z):
    return (np.exp(2*z) - 1) / (np.exp(2*z) + 1)

# === Load Data ===
input_path = "/scratch/leuven/365/vsc36597/televic_data/final-essay/qwq_result_images/qwq_eval/overall/merged_all_subjects.csv"
output_dir = "/scratch/leuven/365/vsc36597/televic_data/final-essay/qwq_result_images/qwq_eval/overall"

df = pd.read_csv(input_path)
df = df.dropna(subset=["Score", "QWQ_Score"])
df["Teacher"] = df["Score"]
df["QWQ"] = df["QWQ_Score"]
df = df[(df["Teacher"] <= 1) & (df["QWQ"] <= 1)]

def compute_metrics(sub_df):
    binary_t = (sub_df["Teacher"] >= 0.5).astype(int)
    binary_q = (sub_df["QWQ"] >= 0.5).astype(int)
    try:
        pearson_corr, p_val = pearsonr(sub_df["Teacher"], sub_df["QWQ"])
    except:
        pearson_corr, p_val = np.nan, np.nan
    try:
        auc = roc_auc_score(binary_t, sub_df["QWQ"]) if len(np.unique(binary_t)) > 1 else np.nan
    except:
        auc = np.nan
    try:
        acc = accuracy_score(binary_t, binary_q)
    except:
        acc = np.nan
    try:
        kappa = cohen_kappa_score(binary_t, binary_q)
    except:
        kappa = np.nan
    sub_df["T_bin"] = (sub_df["Teacher"] * 10).round().astype(int).clip(0, 10)
    sub_df["Q_bin"] = (sub_df["QWQ"] * 10).round().astype(int).clip(0, 10)
    try:
        qwk = quadratic_weighted_kappa(sub_df["T_bin"], sub_df["Q_bin"])
        z = fisher_z(qwk)
        qwk_recovered = inverse_fisher_z(z)
    except:
        qwk = z = qwk_recovered = np.nan

    return {
        "subject": None,
        "samples": len(sub_df),
        "qwk": qwk,
        "fisher_z": z,
        "recovered_kappa": qwk_recovered,
        "pearson": pearson_corr,
        "p_value": p_val,
        "auc": auc,
        "accuracy": acc,
        "kappa": kappa,
    }

# === Process per subject and overall ===
metrics_list = []
for subject, sub_df in df.groupby("Subject"):
    if len(sub_df) >= 5:
        out_dir = os.path.join(output_dir, subject)
        os.makedirs(out_dir, exist_ok=True)
        result = compute_metrics(sub_df)
        result["subject"] = subject
        pd.DataFrame([result]).to_csv(os.path.join(out_dir, "metrics.csv"), index=False)
        metrics_list.append(result)

# overall
overall_result = compute_metrics(df)
overall_result["subject"] = "_ALL_"
all_dir = os.path.join(output_dir, "_ALL_")
os.makedirs(all_dir, exist_ok=True)
pd.DataFrame([overall_result]).to_csv(os.path.join(all_dir, "metrics.csv"), index=False)

# Save merged
summary_df = pd.DataFrame(metrics_list + [overall_result])
summary_df.to_csv(os.path.join(output_dir, "all_metrics_summary.csv"), index=False)

print("✅ All evaluation metrics computed and saved.")
