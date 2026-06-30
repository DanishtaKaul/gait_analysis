"""Pool gait strides per participant and condition, removing outlier trials and strides, and save one mean and CV/SD per cell."""

import os
import numpy as np
import pandas as pd
import kineticstoolkit.lab as ktk

from random_forest_dataset import (
    get_all_non_suspicious_trials,
    get_existence_from_timeseries
)

from helper_functions import extract_light, extract_forewarn
BASE_DIR = r"D:\Gait_Analysis"

# ======================================================
# AGE GROUP DEFINITIONS
# ======================================================

Young = {
    "PID 3", "PID 4", "PID 5", "PID 6", "PID 7", "PID 8", "PID 9",
    "PID 10", "PID 11", "PID 14", "PID 15", "PID 18", "PID 19",
    "PID 20", "PID 23", "PID 26", "PID 30", "PID 31", "PID 35",
    "PID 50", "PID 57", "PID 58"
}

Old = {
    "PID 13", "PID 16", "PID 17", "PID 22", "PID 24", "PID 25", "PID 27",
    "PID 28", "PID 29", "PID 32", "PID 33", "PID 36", "PID 38", "PID 40",
    "PID 41", "PID 43", "PID 45", "PID 46", "PID 49",
    "PID 52", "PID 55"
}


# ======================================================
# HELPER FUNCTIONS
# ======================================================

def get_hs_times(ts):
    """Return sorted HS_L and HS_R times."""
    hs_L = sorted([ev.time for ev in ts.events if ev.name == "HS_L"])
    hs_R = sorted([ev.time for ev in ts.events if ev.name == "HS_R"])
    return hs_L, hs_R


def get_indices_from_times(time_array, event_times):
    """Convert event times to nearest sample indices."""
    return [int(np.argmin(np.abs(time_array - t))) for t in event_times]


def extract_raw_strides(ts):
    """
    Extract individual stride-level values from a trial.

    Returns:
        stride_lengths: array of individual stride lengths
        stride_times:   array of individual stride times
        speeds:         array of individual stride speeds
        step_widths:    array of individual step widths at each HS
    """
    hs_L, hs_R = get_hs_times(ts)

    if len(hs_L) < 2:
        raise ValueError("Not enough HS_L events to compute stride metrics.")

    time = ts.time
    lfoot_ap = ts.data["lfoot_ap"]

    # --- Stride lengths and times (left foot) ---
    hs_L_indices = get_indices_from_times(time, hs_L)

    stride_lengths = []
    for i in range(len(hs_L_indices) - 1):
        idx1 = hs_L_indices[i]
        idx2 = hs_L_indices[i + 1]
        stride_lengths.append(abs(lfoot_ap[idx2] - lfoot_ap[idx1]))

    stride_lengths = np.array(stride_lengths)
    stride_times = np.diff(hs_L)

    min_len = min(len(stride_lengths), len(stride_times))
    stride_lengths = stride_lengths[:min_len]
    stride_times = stride_times[:min_len]

    # --- Speed per stride ---
    speeds = stride_lengths / stride_times

    # --- Step widths at every heel strike (L + R) ---
    all_hs = sorted(hs_L + hs_R)
    left_ankle_ml = ts.data["lfoot_ml"]
    right_ankle_ml = ts.data["rfoot_ml"]
    hs_all_indices = get_indices_from_times(time, all_hs)

    step_widths = []
    for idx in hs_all_indices:
        width = abs(left_ankle_ml[idx] - right_ankle_ml[idx])
        step_widths.append(width)

    step_widths = np.array(step_widths)

    return stride_lengths, stride_times, speeds, step_widths


def compute_cadence(ts):
    """Cadence = total steps / duration * 60."""
    hs_L, hs_R = get_hs_times(ts)

    first_step = min(hs_L[0], hs_R[0])
    last_step = max(hs_L[-1], hs_R[-1])
    duration = last_step - first_step

    if duration <= 0:
        raise ValueError(
            "Non-positive walking duration in cadence calculation.")

    total_steps = len(hs_L) + len(hs_R)
    return (total_steps / duration) * 60


def compute_lateral_path_deviation(ts):
    """
    Lateral path deviation using pelvis ML relative to
    start-to-end straight line during walking phase only.

    Returns:
        lateral_path_dev (mean absolute deviation in metres)
    """
    hs_L, hs_R = get_hs_times(ts)

    if len(hs_L) + len(hs_R) < 2:
        raise ValueError(
            "Not enough heel strikes to compute lateral deviation.")

    first_step = min(hs_L[0], hs_R[0])
    last_step = max(hs_L[-1], hs_R[-1])

    time = ts.time
    start_idx = np.argmin(np.abs(time - first_step))
    end_idx = np.argmin(np.abs(time - last_step))

    pel_ml = ts.data["pel_ml"][start_idx:end_idx+1]
    pel_ap = ts.data["pelvis_ap"][start_idx:end_idx+1]

    ap_start, ap_end = pel_ap[0], pel_ap[-1]
    ml_start, ml_end = pel_ml[0], pel_ml[-1]

    ml_line = ml_start + (ml_end - ml_start) * \
        (pel_ap - ap_start) / (ap_end - ap_start)

    residuals = pel_ml - ml_line
    lateral_path_dev = np.mean(np.abs(residuals))

    return lateral_path_dev


# ======================================================
# THRESHOLDS
# ======================================================

stride_thresholds = {
    "stride_length": (0.6, 1.7),
    "stride_time": (0.85, 2.0),
    "speed": (0.3, 2.0),
}

step_width_threshold = (0.02, 0.40)

trial_thresholds = {
    "cadence": (60, 141),
    "lateral_path_dev": (0, 0.2),
}


# ======================================================
# MAIN LOOP — extract raw strides + trial-level variables
# ======================================================

all_clean_trials = get_all_non_suspicious_trials()
print("Total trials:", len(all_clean_trials))

stride_rows = []
sw_rows = []
trial_rows = []

for i, ts_path in enumerate(all_clean_trials, 1):

    if i % 100 == 0 or i == 1:
        print(f"[{i}/{len(all_clean_trials)}] Processing {os.path.basename(ts_path)}")

    ts = ktk.load(ts_path)

    pid = os.path.basename(os.path.dirname(ts_path))

    # Age
    if pid in Young:
        age = "young"
    elif pid in Old:
        age = "old"
    else:
        raise ValueError(f"PID {pid} not found in age groups.")

    # Light
    light = str(extract_light(ts_path)).lower()

    # Obstacle
    forewarn = str(extract_forewarn(ts_path)).lower()
    existence = get_existence_from_timeseries(ts_path)
    obstacle = f"{forewarn}_{existence}"

    trial_file = os.path.basename(ts_path)

    # --- Extract raw strides ---
    stride_lengths, stride_times, speeds, step_widths = extract_raw_strides(ts)

    for j in range(len(stride_lengths)):
        stride_rows.append({
            "pid": pid,
            "age": age,
            "light": light,
            "obstacle": obstacle,
            "trial_file": trial_file,
            "stride_idx": j,
            "stride_length": stride_lengths[j],
            "stride_time": stride_times[j],
            "speed": speeds[j],
        })

    for j in range(len(step_widths)):
        sw_rows.append({
            "pid": pid,
            "age": age,
            "light": light,
            "obstacle": obstacle,
            "trial_file": trial_file,
            "sw_idx": j,
            "step_width": step_widths[j],
        })

    # --- Trial-level variables ---
    cadence = compute_cadence(ts)
    lateral_dev = compute_lateral_path_deviation(ts)

    trial_rows.append({
        "pid": pid,
        "age": age,
        "light": light,
        "obstacle": obstacle,
        "trial_file": trial_file,
        "cadence": cadence,
        "lateral_path_dev": lateral_dev,
    })


# ======================================================
# BUILD DATAFRAMES
# ======================================================

stride_df = pd.DataFrame(stride_rows)
sw_df = pd.DataFrame(sw_rows)
trial_df = pd.DataFrame(trial_rows)

print(f"\nTotal raw strides: {len(stride_df)}")
print(f"Total raw step widths: {len(sw_df)}")
print(f"Total trials: {len(trial_df)}")


# ======================================================
# STEP 1: REMOVE ENTIRE TRIALS THAT FAIL CADENCE OR
#          LATERAL DEVIATION THRESHOLDS
# ======================================================

print("\n=== Step 1: Trial-level outlier removal (cadence, lateral dev) ===")

trial_df["cadence_fail"] = (
    (trial_df["cadence"] <= trial_thresholds["cadence"][0]) |
    (trial_df["cadence"] >= trial_thresholds["cadence"][1])
)
trial_df["lateral_path_dev_fail"] = (
    (trial_df["lateral_path_dev"] <= trial_thresholds["lateral_path_dev"][0]) |
    (trial_df["lateral_path_dev"] >= trial_thresholds["lateral_path_dev"][1])
)
trial_df["any_trial_fail"] = trial_df["cadence_fail"] | trial_df["lateral_path_dev_fail"]

n_cadence_fail = trial_df["cadence_fail"].sum()
n_lateral_fail = trial_df["lateral_path_dev_fail"].sum()
n_trial_fail = trial_df["any_trial_fail"].sum()
print(f"Cadence failures: {n_cadence_fail}")
print(f"Lateral path dev failures: {n_lateral_fail}")
print(f"Total trials removed: {n_trial_fail}")

# Save excluded trials
excluded_trials = trial_df[trial_df["any_trial_fail"]].copy()


def get_trial_fail_reasons(row):
    reasons = []
    if row["cadence_fail"]:
        reasons.append("cadence")
    if row["lateral_path_dev_fail"]:
        reasons.append("lateral_path_dev")
    return str(reasons)


excluded_trials["fail_reasons"] = excluded_trials.apply(
    get_trial_fail_reasons, axis=1)
excluded_trials.to_csv(
   os.path.join(BASE_DIR, "excluded_trials_cadence_lateral.csv"), index=False)
print("Saved: excluded_trials_cadence_lateral.csv")

# Remove bad trials from ALL dataframes
bad_trial_files = set(trial_df[trial_df["any_trial_fail"]]["trial_file"])
trial_df = trial_df[~trial_df["any_trial_fail"]].copy()
stride_df = stride_df[~stride_df["trial_file"].isin(bad_trial_files)].copy()
sw_df = sw_df[~sw_df["trial_file"].isin(bad_trial_files)].copy()

print(f"Remaining trials: {len(trial_df)}")
print(f"Remaining strides: {len(stride_df)}")
print(f"Remaining step widths: {len(sw_df)}")


# ======================================================
# STEP 2: REMOVE INDIVIDUAL BAD STRIDES
# ======================================================

print("\n=== Step 2: Stride-level outlier removal ===")

stride_df["stride_length_fail"] = (
    (stride_df["stride_length"] <= stride_thresholds["stride_length"][0]) |
    (stride_df["stride_length"] >= stride_thresholds["stride_length"][1])
)
stride_df["stride_time_fail"] = (
    (stride_df["stride_time"] <= stride_thresholds["stride_time"][0]) |
    (stride_df["stride_time"] >= stride_thresholds["stride_time"][1])
)
stride_df["speed_fail"] = (
    (stride_df["speed"] <= stride_thresholds["speed"][0]) |
    (stride_df["speed"] >= stride_thresholds["speed"][1])
)
stride_df["any_stride_fail"] = (
    stride_df["stride_length_fail"] |
    stride_df["stride_time_fail"] |
    stride_df["speed_fail"]
)

for var in ["stride_length", "stride_time", "speed"]:
    print(f"{var}: {stride_df[f'{var}_fail'].sum()} strides removed")

n_stride_fail = stride_df["any_stride_fail"].sum()
print(f"Total strides removed (unique): {n_stride_fail}")

# Save excluded strides
excluded_strides = stride_df[stride_df["any_stride_fail"]].copy()


def get_stride_fail_reasons(row):
    reasons = []
    if row["stride_length_fail"]:
        reasons.append("stride_length")
    if row["stride_time_fail"]:
        reasons.append("stride_time")
    if row["speed_fail"]:
        reasons.append("speed")
    return str(reasons)


excluded_strides["fail_reasons"] = excluded_strides.apply(
    get_stride_fail_reasons, axis=1)
excluded_strides.to_csv(
    os.path.join(BASE_DIR, "excluded_strides_physiological.csv"), index=False)
print("Saved: excluded_strides_physiological.csv")

stride_df = stride_df[~stride_df["any_stride_fail"]].copy()
print(f"Remaining strides: {len(stride_df)}")


# ======================================================
# STEP 3: REMOVE INDIVIDUAL BAD STEP WIDTHS
# ======================================================

print("\n=== Step 3: Step width outlier removal ===")

low, high = step_width_threshold
sw_df["step_width_fail"] = (
    (sw_df["step_width"] <= low) | (sw_df["step_width"] >= high)
)

n_sw_fail = sw_df["step_width_fail"].sum()
print(f"Step widths removed: {n_sw_fail}")

excluded_sw = sw_df[sw_df["step_width_fail"]].copy()
excluded_sw.to_csv(
    os.path.join(BASE_DIR, "excluded_step_widths_physiological.csv"), index=False)
print("Saved: excluded_step_widths_physiological.csv")

sw_df = sw_df[~sw_df["step_width_fail"]].copy()
print(f"Remaining step widths: {len(sw_df)}")


# ======================================================
# STEP 4: REMOVE TRIALS WITH < 2 STRIDES REMAINING
#          (from all dataframes)
# ======================================================

print("\n=== Step 4: Remove trials with < 2 strides remaining ===")

strides_per_trial = stride_df.groupby(
    "trial_file").size().reset_index(name="n_strides")
thin_trials = set(
    strides_per_trial[strides_per_trial["n_strides"] < 2]["trial_file"])

# Also find trials that lost ALL strides
all_remaining_trials = set(trial_df["trial_file"])
trials_with_strides = set(stride_df["trial_file"])
no_strides_left = all_remaining_trials - trials_with_strides

bad_thin_trials = thin_trials | no_strides_left
print(f"Trials with < 2 strides remaining: {len(bad_thin_trials)}")

if len(bad_thin_trials) > 0:
    thin_trial_info = trial_df[trial_df["trial_file"].isin(
        bad_thin_trials)].copy()
    thin_trial_info["fail_reasons"] = "fewer_than_2_strides_after_filtering"
    thin_trial_info.to_csv(
        os.path.join(BASE_DIR, "excluded_trials_too_few_strides.csv"), index=False)
    print("Saved: excluded_trials_too_few_strides.csv")

    stride_df = stride_df[~stride_df["trial_file"].isin(
        bad_thin_trials)].copy()
    sw_df = sw_df[~sw_df["trial_file"].isin(bad_thin_trials)].copy()
    trial_df = trial_df[~trial_df["trial_file"].isin(bad_thin_trials)].copy()

print(f"Remaining trials: {len(trial_df)}")
print(f"Remaining strides: {len(stride_df)}")
print(f"Remaining step widths: {len(sw_df)}")


# ======================================================
# POOL AND AGGREGATE
# ======================================================

print("\n=== Pooling per participant x condition ===")

group_cols = ["pid", "age", "light", "obstacle"]


def cv_pct(x):
    """CV as percentage, ddof=1."""
    if len(x) < 2:
        return np.nan
    return (np.std(x, ddof=1) / np.mean(x)) * 100


# --- Stride-level pooled stats ---
stride_agg = stride_df.groupby(group_cols).agg(
    stride_length_mean=("stride_length", "mean"),
    stride_length_cv_pct=("stride_length", cv_pct),
    stride_time_mean=("stride_time", "mean"),
    stride_time_cv_pct=("stride_time", cv_pct),
    speed_mean=("speed", "mean"),
    n_strides=("stride_length", "size"),
).reset_index()

# --- Step width pooled stats ---
sw_agg = sw_df.groupby(group_cols).agg(
    step_width_mean=("step_width", "mean"),
    step_width_sd=("step_width", lambda x: np.std(
        x, ddof=1) if len(x) > 1 else np.nan),
    n_step_widths=("step_width", "size"),
).reset_index()

# --- Trial-level averaged ---
trial_agg = trial_df.groupby(group_cols).agg(
    cadence=("cadence", "mean"),
    lateral_path_dev=("lateral_path_dev", "mean"),
    n_trials=("cadence", "size"),
).reset_index()


# ======================================================
# MERGE INTO FINAL DATASET
# ======================================================

final_df = stride_agg.merge(sw_agg, on=group_cols, how="outer")
final_df = final_df.merge(trial_agg, on=group_cols, how="outer")

print(f"\nFinal dataset: {len(final_df)} rows (participant x condition)")
print(f"Unique PIDs: {final_df['pid'].nunique()}")


# ======================================================
# SAVE
# ======================================================

out_path = os.path.join(BASE_DIR, "rf_pooled_gait_features.csv")
final_df.to_csv(out_path, index=False)
print(f"\nSaved to: {out_path}")

# --- Descriptives ---
print("\n=== DESCRIPTIVES ===")
desc_cols = [
    "stride_length_mean", "stride_length_cv_pct",
    "stride_time_mean", "stride_time_cv_pct",
    "speed_mean",
    "step_width_mean", "step_width_sd",
    "cadence", "lateral_path_dev",
    "n_strides", "n_step_widths", "n_trials",
]
print(final_df[desc_cols].describe())

# --- Check minimum strides per cell ---
print(f"\nMinimum strides per cell: {final_df['n_strides'].min()}")
print(f"Minimum step widths per cell: {final_df['n_step_widths'].min()}")
print(
    f"Minimum trials per cell (cadence/lat dev): {final_df['n_trials'].min()}")

print("\nCells with fewest strides:")
print(final_df.nsmallest(10, "n_strides")[group_cols + ["n_strides"]])
