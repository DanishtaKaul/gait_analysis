# -*- coding: utf-8 -*-

"""
Merge each trial's metadata with its tracker files into one CSV per trial.
For each participant, this reads trial_results.csv and the matching tracker
files in the Unity folder, then combines them into a single CSV per trial.
"""


import numpy as np
import pandas as pd
import os
import config

BASE_DIR = r"D:\Gait_Analysis"

def main():
    grand_dir = os.path.join(BASE_DIR, "Grand_CSVs")
    os.makedirs(grand_dir, exist_ok=True)

    for participant_root in config.experiments:

        # Get the paths to trial_results and tracker directories
        all_paths = navigate_experiment(participant_root)

        all_csv = get_grand_csvs(all_paths)

        print(f"Saving all csvs.")

        # Extract participant id 
        participant_pd = os.path.basename(os.path.normpath(participant_root))

        # Use underscore version for folder & filename: "PID 3" -> "PID_3"
        pid_dir_name = participant_pd.replace(" ", "_")

        # Ensure per-participant directory exists
        pid_dir = os.path.join(grand_dir, pid_dir_name)
        os.makedirs(pid_dir, exist_ok=True)

        # Save each merged CSV as: PID_3_grand_T{trial_num}.csv
        for i, df in enumerate(all_csv):
            if df.empty or 'trial_num' not in df.columns:
                continue
            try:
                ppid = str(df['ppid'].iloc[0]).replace(" ", "_")
                trial_num = int(df['trial_num'].iloc[0])
            except Exception:
                ppid = str(df['ppid'].iloc[0]).replace(" ", "_")
                trial_num = str(df['trial_num'].iloc[0])

            if trial_num == 1 or trial_num == '1':  # Skip the first csv
                continue

            out_name = f"{ppid}_grand_T{trial_num}.csv"
            out_path = os.path.join(pid_dir, out_name)

            print(f'saving {trial_num}, here {out_path}')
            df.to_csv(out_path, index=False)


def get_grand_csvs(all_paths):
    """
    Returns
    -------
    all_csv : list of DataFrames
        One merged DataFrame per (trial × tracker file).
    """
    print("Gathering grand CSVs")
    all_csv = []

    for block in all_paths:
        tracker_path = block['trackers']
        trial_results = block['trial_results']

        trial_result_attributes = [
            "ppid",
            "trial_num",
            "start_time",
            "end_time",
            "SpawnDistance",
            "LightCondition",
            "ExistanceLevel",
            "ChallengeLevel",
            "ForewarningLevel"
        ]

        tr_csv = pd.read_csv(trial_results, usecols=trial_result_attributes)

        for _, row in tr_csv.iterrows():
            trial_num = row['trial_num']

            file_prefix = [
                'LeftFootHeel',
                'LeftFootToe',
                'obstacle',
                'pelvis',
                'RightFootHeel',
                'RightFootToe',
            ]

            # Format trial number as two digits (1 -> '01', 15 -> '15')
            formatted_trial_num = str(trial_num).zfill(2)
            file_suffix = f'_movement_T0{formatted_trial_num}.csv'

            trial_tracker_paths = []
            for prefix in file_prefix:
                file_name = prefix + file_suffix
                current_trial_tracker_path = os.path.join(
                    tracker_path, file_name)
                trial_tracker_paths.append(
                    {'prefix': prefix, 'path': current_trial_tracker_path})

            # Use only the current trial's metadata row for merging
            base_row = tr_csv[tr_csv['trial_num'] == trial_num].copy()

            # per-trial accumulator
            acc = None

            for path in trial_tracker_paths:

                if path['prefix'] == "obstacle":
                    tracker_attributes = [
                        {'name': 'time',
                            'rename': path['prefix'] + '_time'},
                        {'name': 'obstacle_crossing',
                            'rename': path['prefix'] + '_obstacle_crossing'},
                        {'name': 'mid_point_crossing',
                            'rename': path['prefix'] + '_mid_point_crossing'}
                    ]
                else:
                    tracker_attributes = [
                        {'name': 'state', 'rename': path['prefix'] + '_state'},
                        {'name': 'time',  'rename': path['prefix'] + '_time'},
                    ]

                csv = pd.read_csv(path['path'], usecols=[
                                  att['name'] for att in tracker_attributes])

                # align rows across trackers
                csv = csv.reset_index(drop=True)
                csv['sample_idx'] = csv.index

                # Rename safely
                rename_map = {att['name']: att['rename']
                              for att in tracker_attributes}
                existing_map = {k: v for k,
                                v in rename_map.items() if k in csv.columns}
                csv = csv.rename(columns=existing_map)

                # Ensure merge on trial_num
                csv['trial_num'] = trial_num

                # Merge into per-trial accumulator
                if acc is None:
                    acc = base_row.merge(csv, on="trial_num", how="right")
                else:
                    acc = acc.merge(
                        csv, on=["trial_num", "sample_idx"], how="outer")

            # Collect result once per trial
            df = acc
            all_csv.append(df)

    return all_csv


def navigate_experiment(experiment_root):
    """

    Navigates an experiment root and returns the trial_results_path and tracker directory path 

    Parameters
    ----------
    experiment_root : STRING

    Returns
    -------
    trial_results_paths : ARRAY
    tracker_paths : ARRAY

    """

    print(f"Navigating experiment: {experiment_root}")
    unity_dir = None

    # Find Unity directories.
    for item in os.listdir(experiment_root):
        item_path = os.path.join(experiment_root, item)
        if os.path.isdir(item_path):
            lower_item = item.lower()
            if 'unity' in lower_item:
                unity_dir = item_path

    if unity_dir is None:
        print(
            "No Unity folder (with 'unity' in its name) found in experiment root.")
        return

    # Define conditions.
    light_conditions = ['light', 'ambient', 'dark']
    obstacle_conditions = ['expected', 'unexpected']

    all_paths = []
    # Process each combination of conditions.
    for light in light_conditions:
        for obstacle in obstacle_conditions:

            # Search for the appropriate Unity block folder.
            unity_block_dir = None
            for folder in os.listdir(unity_dir):
                folder_lower = folder.lower()
                folder_path = os.path.join(unity_dir, folder)
                if os.path.isdir(folder_path) and (light in folder_lower):
                    if obstacle == 'expected' and 'unexpected' in folder_lower:
                        continue
                    elif obstacle == 'unexpected' and 'unexpected' not in folder_lower:
                        continue
                    unity_block_dir = folder_path
                    break

            if unity_block_dir is None:
                logger.info(
                    f"No Unity block folder found for condition: {light} & {obstacle}")
                continue

            # Locate the subfolder starting with 'S'.
            s_folder = None
            for folder in os.listdir(unity_block_dir):
                folder_path = os.path.join(unity_block_dir, folder)
                if os.path.isdir(folder_path) and folder.lower().startswith('s'):
                    s_folder = folder_path
                    break

            if s_folder is None:
                logger.info(
                    f"No subfolder starting with 'S' found in Unity block folder for condition: {light} & {obstacle}")
                continue

            # Check if trial_results.csv exists.
            trial_results_path = os.path.join(s_folder, "trial_results.csv")
            if not os.path.exists(trial_results_path):
                logger.info(
                    f"trial_results.csv not found in {s_folder} for condition: {light} & {obstacle}")
                continue

            trackers_path = os.path.join(s_folder, "trackers")

            # Stores matched metadata files so they can be processed later all at once

            all_paths.append({
                "trial_results": trial_results_path,
                "trackers": trackers_path
            })

    return all_paths


if __name__ == "__main__":
    main()
