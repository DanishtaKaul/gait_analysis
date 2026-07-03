# -*- coding: utf-8 -*-

"""Detect heel strikes from motion-capture data, quality check, and save per-trial time series."""


import os
import pandas as pd
import numpy as np
import kineticstoolkit.lab as ktk
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, detrend
from PyQt5 import QtWidgets

from helper_functions import (
    findTrackersFolder,
    getSkippedTrialsDict,
    max_nan_gap,
    extract_light
)

# ======================================================
# SETTINGS
# ======================================================

BASE_PATH = r"D:\motion_tracking_data"
BASE_DIR = r"D:\Gait_Analysis"

AP_AXIS = "Pos[2]"          # anterior–posterior axis
MAX_GAP = 15               # frames
PROMINENCE = 0.3           # metres
MIN_PEAK_DISTANCE = 120    # samples (~0.5 s @ 240 Hz)

KEEP_MARKERS = [
    ":Skeleton 001",   # pelvis
    "LFoot",            # left ankle
    "RFoot"
]

# -------- TEST MODE --------
TEST_MODE = False
TEST_PID = "PID 13"
TEST_BLOCK_CONTAINS = "LIGHT"
# ---------------------------

SKIP_PARTICIPANTS = [
    'PID 10',
    'PID 11',
    'PID 13',
    'PID 14',
    'PID 15',
    'PID 16',  # Excluded
    'PID 17',
    'PID 18',
    'PID 19',
    'PID 20',
    'PID 22',
    'PID 23',
    'PID 24',
    'PID 25',
    'PID 26',
    'PID 27',
    'PID 28',
    'PID 29',
    'PID 3',
    'PID 30',
    'PID 31',
    'PID 32',
    'PID 33',
    'PID 35',
    'PID 36',
    'PID 38',
    'PID 4',
    'PID 40',  # Excluded
    'PID 41',
    'PID 42',  # Excluded
    'PID 43',
    'PID 45',
    'PID 46',
    'PID 49',
    'PID 5',
    'PID 50',
    'PID 52',
    'PID 55',
    'PID 57',
    'PID 58',
    'PID 6',
    'PID 7',
    'PID 8',
    'PID 9',
]

# Change to "" to start from first block
START_FROM_BLOCK = "ambient expected"

TS_BASE_OUT = os.path.join(BASE_DIR, "time_series_data")


# ======================================================
# STORAGE
# ======================================================


all_trials_to_skip = getSkippedTrialsDict()


# ======================================================
# MAIN LOOP
# ======================================================

for ppid in os.listdir(BASE_PATH):

    if TEST_MODE and ppid.upper() != TEST_PID:
        continue
    if ppid.upper() in SKIP_PARTICIPANTS:
        print(f'{ppid} in skip list. Skipping...')
        continue

    ppid_dir = os.path.join(BASE_PATH, ppid)

    for csv_file in os.listdir(ppid_dir):

        csv_path = os.path.join(ppid_dir, csv_file)
        block_id = os.path.splitext(csv_file)[0].upper()

        if START_FROM_BLOCK.upper() not in block_id.upper():
            print(
                f'{block_id} is not in {START_FROM_BLOCK}. Skipping...')
            continue
        print(
            f'{block_id} is in {START_FROM_BLOCK}. Starting!')
        START_FROM_BLOCK = ""

        if TEST_MODE and TEST_BLOCK_CONTAINS not in block_id:
            continue

        trials_to_skip = all_trials_to_skip.get(block_id, set())
        light = extract_light(block_id)

        # -----------------------------
        # LOAD CSV
        # -----------------------------
        df = pd.read_csv(
            csv_path,
            on_bad_lines="skip",
            skiprows=[0],
            low_memory=False
        )

        df.drop(df.index[1], inplace=True)

        df.columns = df.iloc[0:3, :].apply(
            lambda x: " ".join(x.dropna().astype(str)), axis=0
        )
        df.drop(df.index[0:3], inplace=True)
        df.reset_index(drop=True, inplace=True)

        df = df[df.columns.drop(list(df.filter(regex="Unlabeled")))]

        df.columns = (
            df.columns
            .str.replace("Position X", "Pos[0]")
            .str.replace("Position Y", "Pos[1]")
            .str.replace("Position Z", "Pos[2]")
        )

        df = df.rename(columns={
            "Time (Seconds)": "Time",
            "Name Time (Seconds)": "Time"
        })

        df = df.drop(df.columns[0], axis=1)
        df = df.astype(float)
        df = df.set_index("Time")

        pattern = "|".join(KEEP_MARKERS)
        df = df.loc[:, df.columns.str.contains(pattern)]

        # -----------------------------
        # WALK WINDOWS
        # -----------------------------
        trackers_path = findTrackersFolder(ppid, block_id)

        trial_info = pd.read_csv(
            os.path.join(trackers_path, "trial_results.csv")
        )

        # Sort via time
        trial_info = trial_info.sort_values(
            "start_time").reset_index(drop=True)

        trial_windows = []

        for i in range(len(trial_info)):
            trial_num = int(trial_info.loc[i, "trial_num"])

            if trial_num == 1:
                continue
            start = trial_info.loc[i, "start_time"]

            # end = start of next trial (if exists)
            if i < len(trial_info) - 1:
                end = trial_info.loc[i + 1, "start_time"]
            else:
                end = df.index.max()

            trial_windows.append((trial_num, start, end))

        # -----------------------------
        # GAP CHECK + INTERPOLATION
        # -----------------------------
        discard = False
        for col in df.columns:
            if max_nan_gap(df[col]) > MAX_GAP:
                discard = True
                break
            df[col] = df[col].interpolate(
                method="linear",
                limit=MAX_GAP,
                limit_direction="both"
            )

        if discard:
            continue

        # ======================================================
        # EXTRACT PELVIS AND FOOT POSITIONS
        # ======================================================

        time = df.index.values

        pelvis_col = [
            c for c in df.columns
            if ":Skeleton 001" in c and AP_AXIS in c
        ][0]

        lfoot_col = [
            c for c in df.columns
            if "LFoot" in c and AP_AXIS in c
        ][0]

        rfoot_col = [
            c for c in df.columns
            if "RFoot" in c and AP_AXIS in c
        ][0]

        pelvis_ap = df[pelvis_col].values
        lfoot_ap = df[lfoot_col].values
        rfoot_ap = df[rfoot_col].values
        # -----------------------------
        # HEEL STRIKE DETECTION
        # -----------------------------

        ap_rel = detrend(lfoot_ap - pelvis_ap, type="linear")
        rp_rel = detrend(rfoot_ap - pelvis_ap, type="linear")

        # KTK TimeSeries (events only)
        ts = ktk.TimeSeries(
            time=time,
            data={
                "pelvis_ap": pelvis_ap,
                "lfoot_ap": lfoot_ap,
                "rfoot_ap": rfoot_ap,
                "LFoot_relPelvis": ap_rel,
                "RFoot_relPelvis": rp_rel
            }
        )
        

        peaks, _ = find_peaks(
            ap_rel,
            prominence=PROMINENCE,
            distance=MIN_PEAK_DISTANCE
        )

        r_peaks, _ = find_peaks(
            rp_rel,
            prominence=PROMINENCE,
            distance=MIN_PEAK_DISTANCE
        )

        # Append peaks to the time series
        # LEFT heel strikes
        for idx in peaks:
            ts = ts.add_event(time[idx], "HS_L")

        # RIGHT heel strikes
        for idx in r_peaks:
            ts = ts.add_event(time[idx], "HS_R")

        # Identifying groups
        # Combine L and R heel strikes
        all_peaks = np.sort(
            np.concatenate([peaks, r_peaks])
        )

        all_peak_times = time[all_peaks]

        GROUP_GAP_SEC = 1.5

        groups = []
        start_idx = 0
        MIN_STEPS_PER_GROUP = 4

        # Create groups
        for i in range(1, len(all_peak_times)):
            if all_peak_times[i] - all_peak_times[i - 1] > GROUP_GAP_SEC:
                end_idx = i - 1

                # number of steps in this group
                n_steps = end_idx - start_idx + 1

                if n_steps >= MIN_STEPS_PER_GROUP:
                    groups.append((start_idx, end_idx))

                start_idx = i

        end_idx = len(all_peak_times) - 1
        n_steps = end_idx - start_idx + 1

        if n_steps >= MIN_STEPS_PER_GROUP:
            groups.append((start_idx, end_idx))

        group_windows = []

        for g_start, g_end in groups:
            w_start = all_peak_times[g_start]
            w_end = all_peak_times[g_end]

            # Add 1 sec of padding to all groups
            PAD_SEC = 1.0
            t_min = time[0]
            t_max = time[-1]

            w_start = max(w_start - PAD_SEC, t_min)
            w_end = min(w_end + PAD_SEC,   t_max)

            group_windows.append((w_start, w_end))

        group_windows = group_windows[-41:]

        trial_to_existance = dict(
            zip(trial_info["trial_num"], trial_info["ExistanceLevel"])
        )

        for w_start, w_end in reversed(group_windows):
            existance = trial_to_existance.get(trial_num, "UNKNOWN")

            if existance == "UNKNOWN":
                continue
            ts = ts.add_event(w_start, f"START_T{trial_num}_{existance}")
            ts = ts.add_event(w_end,   f"END_T{trial_num}_{existance}")
            trial_num -= 1

        start_trial_num = int(trial_info["trial_num"].iloc[-1])
        trial_num = start_trial_num

        trial_to_existance = dict(
            zip(trial_info["trial_num"], trial_info["ExistanceLevel"])
        )

        pid_dir = os.path.join(TS_BASE_OUT, ppid)
        os.makedirs(pid_dir, exist_ok=True)

        print(f"Viewing {ppid} | {trials_to_skip}")

        for i, (w_start, w_end) in enumerate(group_windows, start=1):

            if i in trials_to_skip:
                print(f"[SKIP] Trial {i} (marked to skip)")
                continue

            existance = trial_to_existance.get(trial_num, "UNKNOWN")


            # -----------------------------
            # Launch editing GUI 
            # -----------------------------
            print(f"\n Showing {ppid} {block_id} T{i}_{existance}")
            ts_group = ts.ui_edit_events(
                name=["HS_L", "HS_R"],
                data_keys=["LFoot_relPelvis", "RFoot_relPelvis"]
            )
            plt.close()

            # -----------------------------
            # Save edited TimeSeries
            # -----------------------------
            ts_out = os.path.join(
                pid_dir,
                f"{block_id}_T{i}_{existance}.zip"
            )
            # ktk.save(ts_out, ts_group)

            print(f"[SAVED AFTER QC] {ts_out}")
            trial_num -= 1

    
