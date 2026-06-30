# -*- coding: utf-8 -*-
"""Collect all cleaned trial time series, remove suspicious ones, and count retained trials per condition."""

import os
import re
import pandas as pd
from collections import Counter
import kineticstoolkit.lab as ktk

from helper_functions import (
    extract_light,
    extract_forewarn
)

BASE_DIR = r"D:\Gait_Analysis"
TIME_SERIES_DATA_DIR = os.path.join(BASE_DIR, "time_series_data_x_y_added")
SUSPICIOUS_CSV = os.path.join(BASE_DIR, "time_series_suspicious_trials.csv")


# ======================================================
# 1) Get all TimeSeries trial paths
# ======================================================
def get_all_trials():
    """
    Returns a list of full paths to all .zip TimeSeries files
    in TIME_SERIES_DATA_DIR/PID xx/, excluding practice trial 1.
    """
    all_trials = []

    for pid in os.listdir(TIME_SERIES_DATA_DIR):
        pid_dir = os.path.join(TIME_SERIES_DATA_DIR, pid)
        if not os.path.isdir(pid_dir):
            continue

        for ts_file in os.listdir(pid_dir):
            if not ts_file.lower().endswith(".zip"):
                continue

            # Skip practice trial 1 for every block 
            if "_T1_" in ts_file:
                continue

            all_trials.append(os.path.join(pid_dir, ts_file))

    return all_trials


# ======================================================
# 2) Remove suspicious trials
# ======================================================
def remove_suspicious_trials(all_trials):
    """
    Removes any trial whose filename appears in SUSPICIOUS_CSV['filename'].

    This implementation avoids mutating the list while iterating.
    It returns a new filtered list.
    """
    print("remove_suspicious_trials")

    df = pd.read_csv(SUSPICIOUS_CSV)
    suspicious_names = df["filename"].astype(str).tolist()

    cleaned = []
    for trial_path in all_trials:
        # If any suspicious filename is contained in this trial path, drop it
        is_suspicious = any(name in trial_path for name in suspicious_names)
        if not is_suspicious:
            cleaned.append(trial_path)

    print(f"\t all_trials: {len(all_trials)} -> cleaned: {len(cleaned)}")
    return cleaned


def get_all_non_suspicious_trials():
    """
    Convenience function used by other scripts.
    Returns all TimeSeries trials after suspicious removal.
    Does NOT write any CSVs.
    """
    return remove_suspicious_trials(get_all_trials())


# ======================================================
# 3) Extract existence (present/absent) from TimeSeries events
# ======================================================
def get_existence_from_timeseries(ts_path):
    """
    Reads the TimeSeries file and extracts existence from event names.

    time seriest adds events like:
      START_T12_Present
      END_T12_Absent

    This function returns: "present" or "absent"

    """
    ts = ktk.load(ts_path)

    # Try to find START_T... first, then END_T...
    for ev in ts.events:
        name = str(ev.name)

        m = re.match(r"^START_T\d+_(.+)$", name)
        if m:
            existence = m.group(1).strip().lower()
            if existence in {"present", "absent"}:
                return existence

        m = re.match(r"^END_T\d+_(.+)$", name)
        if m:
            existence = m.group(1).strip().lower()
            if existence in {"present", "absent"}:
                return existence

    raise ValueError(
        f"Could not find Present/Absent START_T... or END_T... event in: {ts_path}"
    )


# ======================================================
# 4) Write retained trial count CSVs
# ======================================================
def write_retained_counts_by_light(all_trials, output_path):
    """
    Writes counts of retained trials per PID × light
    Light is extracted from filename/path 
    Hard-fails if light cannot be extracted
    """
    counter = Counter()

    for ts_path in all_trials:
        pid = os.path.basename(os.path.dirname(ts_path))

        light = extract_light(ts_path)
        if light is None:
            raise ValueError(f"extract_light() returned None for: {ts_path}")

        light = str(light).lower()
        if light not in {"light", "ambient", "dark"}:
            raise ValueError(f"Unexpected light label '{light}' in: {ts_path}")

        counter[(pid, light)] += 1

    rows = [{"pid": pid, "light": light, "num_trials": n}
            for (pid, light), n in counter.items()]

    df = pd.DataFrame(rows).sort_values(["pid", "light"])
    df.to_csv(output_path, index=False)
    print(f"Saved retained counts (PID × light): {output_path}")
    return df


def write_retained_counts_by_existence(all_trials, output_path):
    """
    Writes counts of retained trials per PID × existence (present/absent)
    Existence is extracted from TimeSeries events
    Hard-fails if existence cannot be extracted for any file
    """
    counter = Counter()

    for idx, ts_path in enumerate(all_trials, start=1):

        if idx % 200 == 0:
            print(f"Existence progress: {idx}/{len(all_trials)}")

        pid = os.path.basename(os.path.dirname(ts_path))

        existence = get_existence_from_timeseries(ts_path)
        counter[(pid, existence)] += 1

    rows = [{"pid": pid, "existence": existence, "num_trials": n}
            for (pid, existence), n in counter.items()]

    df = pd.DataFrame(rows).sort_values(["pid", "existence"])
    df.to_csv(output_path, index=False)
    print(f"Saved retained counts (PID × existence): {output_path}")
    return df


def write_retained_counts_by_obstacle_condition(all_trials, output_path):
    """
    Writes counts per PID × obstacle condition:
    expected_present
    expected_absent
    unexpected_present
    unexpected_absent
    """
    counter = Counter()

    for ts_path in all_trials:

        pid = os.path.basename(os.path.dirname(ts_path))

        # expected / unexpected
        forewarn = extract_forewarn(ts_path)
        forewarn = str(forewarn).lower()

        # present / absent
        existence = get_existence_from_timeseries(ts_path)

        obstacle_condition = f"{forewarn}_{existence}"

        counter[(pid, obstacle_condition)] += 1

    rows = [{"pid": pid, "obstacle_condition": cond, "num_trials": n}
            for (pid, cond), n in counter.items()]

    df = pd.DataFrame(rows).sort_values(["pid", "obstacle_condition"])
    df.to_csv(output_path, index=False)

    print(f"Saved retained counts (PID × obstacle_condition): {output_path}")

    return df


def write_retained_counts_by_light_obstacle(all_trials, output_path):
    """
    Writes counts per PID × light × obstacle condition
    Example:
    light + expected_present
    ambient + unexpected_absent
    """
    counter = Counter()

    for ts_path in all_trials:

        pid = os.path.basename(os.path.dirname(ts_path))

        light = extract_light(ts_path)
        light = str(light).lower()

        forewarn = extract_forewarn(ts_path)
        forewarn = str(forewarn).lower()

        existence = get_existence_from_timeseries(ts_path)

        obstacle_condition = f"{forewarn}_{existence}"

        counter[(pid, light, obstacle_condition)] += 1

    rows = [{"pid": pid, "light": light,
             "obstacle_condition": cond,
             "num_trials": n}
            for (pid, light, cond), n in counter.items()]

    df = pd.DataFrame(rows).sort_values(["pid", "light", "obstacle_condition"])
    df.to_csv(output_path, index=False)

    print(f"Saved retained counts (PID × light × obstacle): {output_path}")

    return df


# ======================================================
# Only create CSVs when running this script directly
# ======================================================
if __name__ == "__main__":
    all_clean_trials = get_all_non_suspicious_trials()

    n_t1 = sum("_T1_" in os.path.basename(p) for p in all_clean_trials)
    print("T1 trials remaining:", n_t1)  

    write_retained_counts_by_light(
        all_clean_trials,
        output_path=os.path.join(BASE_DIR, "random_forest_retained_trials_by_light.csv")
    )

    write_retained_counts_by_existence(
        all_clean_trials,
        output_path=os.path.join(BASE_DIR, "random_forest_retained_trials_by_existence.csv")
    )

    write_retained_counts_by_obstacle_condition(
        all_clean_trials,
        output_path=os.path.join(BASE_DIR, "random_forest_retained_trials_by_obstacle_condition.csv")
    )

    write_retained_counts_by_light_obstacle(
        all_clean_trials,
       output_path=os.path.join(BASE_DIR, "random_forest_retained_trials_by_light_obstacle.csv")
    )

    print("Done")
