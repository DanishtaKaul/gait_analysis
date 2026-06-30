# -*- coding: utf-8 -*-

"""Count heel strikes per trial and flag trials with too many, too few, or duplicate steps"""


import kineticstoolkit.lab as ktk
import pandas as pd
import numpy as np
import os

# ---------------- PATH ----------------
BASE_DIR = r"D:\Gait_Analysis"
TS_DIR_PATH = os.path.join(BASE_DIR, "time_series_data_x_y_added")
MAX_STEPS = 15
MIN_STEPS = 6

trial_id = ""

log_steps = []  # For logging steps outside of step bounds

for file in os.listdir(TS_DIR_PATH):

    ts_path = os.path.join(TS_DIR_PATH, file)
    if os.path.isfile(ts_path):
        continue

    for ts_file in os.listdir(ts_path):

        if trial_id and trial_id not in ts_file:
            print(f'\n TRIAL ID {trial_id} : Skipping {ts_file}')
            continue

        ts_file_path = os.path.join(ts_path, ts_file)

        # 1) Load TimeSeries
        ts = ktk.load(ts_file_path)

        count = 0
        last_heel_strike = ""  # Track the last HEEL STRIKE specifically
        has_duplicate_event = False

        for event in ts.events:
            # Check if current event is a heel strike
            if event.name in ["HS_L", "HS_R"]:

                # Compare current heel strike to the PREVIOUS heel strike found
                if event.name == last_heel_strike:
                    log_steps.append([
                        ts_file,
                        None,
                        f'Duplicate {event.name} detected'
                    ])
                    has_duplicate_event = True
                    break

                # If not a duplicate, count it and update tracker
                count += 1
                last_heel_strike = event.name

        if has_duplicate_event:
            print(f'Duplicate {last_heel_strike} found. Skipping {ts_file}')
            continue

        print(f'{ts_file} step count {count}')

        if count > MAX_STEPS or count < MIN_STEPS:
            log_steps.append(
                [ts_file, count, f"Step count too {'high' if count > MAX_STEPS else 'low'}"])

            print(f'Plotting {ts_file}, step count {count}')
            ts_edit = ts.ui_edit_events(
                name=["HS_L", "HS_R"],
                data_keys=["LFoot_relPelvis", "RFoot_relPelvis"]
            )

print('\n Suspicious trials:')
for row in log_steps:
    # row[0] is filename, row[1] is count, row[2] is reason
    print(f'{row[0]} | Count: {row[1]} | Reason: {row[2]}')

"""
SAVE TO CSV
"""
if log_steps:
    df_log = pd.DataFrame(log_steps, columns=[
                          "filename", "step_count", "reason"])

    # Save to CSV
    out_csv = os.path.join(BASE_DIR, "time_series_suspicious_trials.csv")
    df_log.to_csv(out_csv, index=False)
    print(f"\nSaved {len(df_log)} suspicious trials to: {out_csv}")
else:
    print("\nNo suspicious trials found to save.")
