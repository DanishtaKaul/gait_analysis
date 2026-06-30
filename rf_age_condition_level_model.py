"""Classify age group (young vs old) from pooled gait features using a random forest, with SHAP feature explanations."""
import numpy as np
import pandas as pd
import shap
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, accuracy_score
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn import set_config
set_config(enable_metadata_routing=True)

BASE_DIR = r"D:\Gait_Analysis"

df_cond = pd.read_csv(
    os.path.join(BASE_DIR, "rf_pooled_gait_features.csv"))
df_cond = df_cond[df_cond["pid"] != "PID 49"]

all_features = [
    "stride_length_mean",
    "stride_time_mean",
    "stride_time_cv_pct",
    "cadence",
    "step_width_mean",
    "stride_length_cv_pct",
    "step_width_sd",
    "lateral_path_dev",
]


print(
    df_cond.groupby("age")[[
        "stride_time_mean",
        "stride_length_cv_pct",
        "cadence"
    ]].mean()
)


print("Rows:", len(df_cond))
print("Unique participants:", df_cond["pid"].nunique())

X = df_cond[all_features]
y = (df_cond["age"] == "young").astype(int)
groups = df_cond["pid"]

rf = RandomForestClassifier(
    n_estimators=500,
    random_state=42,
    class_weight="balanced"
)

gkf = GroupKFold(n_splits=5)

# --------------------------------------------------
# Sequential Feature Selection
# --------------------------------------------------

sfs = SequentialFeatureSelector(
    rf,
    n_features_to_select="auto",
    direction="forward",
    scoring="balanced_accuracy",
    cv=gkf,
    n_jobs=-1
)

sfs.fit(X, y, groups=groups)

selected_features = X.columns[sfs.get_support()]

print("\nSelected features:")
print(selected_features)

X = df_cond[selected_features]


ba, auc, acc = [], [], []

all_preds = []
all_true = []
all_pids = []
all_probs = []

for fold, (tr, te) in enumerate(gkf.split(X, y, groups), start=1):

    rf.fit(X.iloc[tr], y.iloc[tr])

    prob = rf.predict_proba(X.iloc[te])[:, 1]
    pred = (prob >= 0.5).astype(int)

    ba.append(balanced_accuracy_score(y.iloc[te], pred))
    auc.append(roc_auc_score(y.iloc[te], prob))
    acc.append(accuracy_score(y.iloc[te], pred))

    # Store predictions
    all_preds.extend(pred)
    all_true.extend(y.iloc[te])
    all_pids.extend(groups.iloc[te])
    all_probs.extend(prob)

print("Balanced Accuracy:", np.mean(ba))
print("ROC-AUC:", np.mean(auc))
print("Accuracy:", np.mean(acc))

results_df = pd.DataFrame({
    "pid": all_pids,
    "true_label": all_true,
    "predicted_label": all_preds,
    "prob_young": all_probs
})

results_df["correct"] = results_df["true_label"] == results_df["predicted_label"]

print(results_df.head())

pid_summary = results_df.groupby("pid").agg(
    true_label=("true_label", "first"),
    mean_prob_young=("prob_young", "mean"),
    correct_fraction=("correct", "mean")
).reset_index()

print(pid_summary.sort_values("correct_fraction"))

rf.fit(X, y)

explainer = shap.TreeExplainer(rf)
sv = explainer(X)

shap.summary_plot(sv[:, :, 1], X)

# Save SHAP values for class 1 (young)
shap_df = pd.DataFrame(
    sv.values[:, :, 1],
    columns=selected_features
)

shap_df["pid"] = df_cond["pid"].values
shap_df["light"] = df_cond["light"].values
shap_df["obstacle"] = df_cond["obstacle"].values
shap_df["age"] = df_cond["age"].values

shap_df.to_csv(
    os.path.join(BASE_DIR, "rf_outputs", "shap_condition_level_values.csv"),
    index=False
)

print("Saved SHAP values.")

shap_df = pd.read_csv(
    os.path.join(BASE_DIR, "rf_outputs", "shap_condition_level_values.csv")
)

for light in shap_df["light"].unique():
    sub = shap_df[shap_df["light"] == light]

    mean_abs = sub[selected_features].abs().mean().sort_values(ascending=False)

    print("\nLight:", light)
    print(mean_abs)

for obs in shap_df["obstacle"].unique():
    sub = shap_df[shap_df["obstacle"] == obs]

    mean_abs = sub[selected_features].abs().mean().sort_values(ascending=False)

    print("\nObstacle:", obs)
    print(mean_abs)
# ======================================================
# DIRECTION CHECK: Does high stride_length_cv_pct push toward old or young
# ======================================================


print("\nGenerating SHAP dependence plot for stride_length_cv_pct...")

shap.dependence_plot(
    "stride_length_cv_pct",
    sv.values[:, :, 1],
    X
)
