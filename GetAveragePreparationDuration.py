# -*- coding: utf-8 -*-
"""Compute preparation and crossing durations per trial from grand CSVs; flag outliers and save per-condition averages."""

import os
import pandas as pd
import numpy as np
import config

BASE_DIR = r"D:\Gait_Analysis"


def addSkippedTrial(st, reason, trial_num, ppid, path, ppid_condition):
    st.append(
        {
            'reason': reason,
            'trial_num': trial_num,
            'ppid': ppid,
            'ppid_condition': ppid_condition,
            'file_path': path,
        }
    )


def grandCSVLoop(grand_csv_path, save=True,  save_trial_flags=False, save_skipped=True):
    """
    Walk all grand CSVs, compute trial-level prep_time, then:
      - collect raw pairs as (ppid_condition, duration)
      - compute within-group (participant×condition) mean/SD
      - flag trial outliers using ±sd_k*SD
      - save group averages + % flagged to CSV
      - save trial-level flags for auditing
    """
    print("Looping over all grand CSV files")

    skipped_trial_ids = set()
    # list of dicts: {"trial_id": (...), "ppid_condition": "..."}
    valid_trials = []

    def make_trial_id(ppid, trial_num, file_path):
        return (str(ppid).strip(), int(trial_num), str(file_path).strip())

    prep_raw_pairs = []
    cross_raw_pairs = []
    skipped_trials = []
    trial_nums = []

    pid_skip_list = ['PID 12', 'PID 16', 'PID 21', 'PID 40']

    for pid_dir in os.listdir(grand_csv_path):
        path = os.path.join(grand_csv_path, pid_dir)
        print(f'Reading {path}')
        if not os.path.isdir(path):
            continue

        for csv_name in os.listdir(path):
            csv_path = os.path.join(path, csv_name)
            if not csv_path.lower().endswith(".csv"):
                continue

            df = pd.read_csv(csv_path)

            
            # Required fields
            # already includes LIGHT + EXPECTED/UNEXPECTED
            ppid = str(df['ppid'].iloc[2]).strip()
            trial_num = int(df['trial_num'].iloc[2])

            if ppid in pid_skip_list:
                print(f'Participant in skiplist, skipping {ppid}...')
                continue

            # Existence level (the ONLY missing part to add)
            existance = str(df['ExistanceLevel'].iloc[2]).strip()
            existence_label = existance.upper()                    # "PRESENT" / "ABSENT"

            # This is the condition label in skipped_trials.csv
            ppid_condition = f"{ppid} {existence_label}"

            # Walk start = first row where state is "Walk" or "Return Walk"
            walk_mask = df['LeftFootHeel_state'].isin(
                ['Walk', 'ReturnWalk'])

            df_walk_state = df[walk_mask]
            start_walking_time = df_walk_state['obstacle_time'].iloc[0]

            # Crossing selector based on existence
            selector = (
                'obstacle_obstacle_crossing'
                if str(existance).strip().lower() == 'present'
                else 'obstacle_mid_point_crossing'
            )

            df_not_null_state = df[df[selector].notna()]

            if len(df_not_null_state) <= 0:
                # Nothing was recorded in this trial; there was no crossing time
                reason = f'No values exist in {selector} '
                print(
                    f"Error: No values exist in {selector} ")

                
                trial_id = make_trial_id(ppid, trial_num, csv_path)

                if trial_id not in skipped_trial_ids:
                    skipped_trial_ids.add(trial_id)
                    addSkippedTrial(skipped_trials, reason,
                                    trial_num, ppid, csv_path, ppid_condition)

                continue

            start_crossing_time = df_not_null_state['obstacle_time'].iloc[0]
            end_crossing_time = df_not_null_state['obstacle_time'].iloc[-1]

            # Duration
            prep_time = float(start_crossing_time - start_walking_time)
            cross_time = float(end_crossing_time - start_crossing_time)

            def printWalkings():
                print(f'start_walking_time:{start_walking_time}')
                print(f'start_crossing_time:{start_crossing_time}')
                print(f'end_crossing_time:{end_crossing_time}')

            # Rules to skip trials
            if start_walking_time == None or start_crossing_time == None or end_crossing_time == None:
                reason = f'start_walking_time or start_crossing_time or end_crossing_time is missing'
                print(f'    {ppid}: {reason}')
                printWalkings()
                trial_id = make_trial_id(ppid, trial_num, csv_path)

                if trial_id not in skipped_trial_ids:
                    skipped_trial_ids.add(trial_id)
                    addSkippedTrial(skipped_trials, reason,
                                    trial_num, ppid, csv_path, ppid_condition)

                continue

            if end_crossing_time < start_crossing_time or start_crossing_time < start_walking_time:
                reason = f'start_walking_time ({start_walking_time:.3f}) or start_crossing_time ({start_crossing_time:.3f}) or end_crossing_time ({end_crossing_time:.3f}) is sequentially out of order'
                print(f'    {ppid}: {reason}')
                printWalkings()
                trial_id = make_trial_id(ppid, trial_num, csv_path)

                if trial_id not in skipped_trial_ids:
                    skipped_trial_ids.add(trial_id)
                    addSkippedTrial(skipped_trials, reason,
                                    trial_num, ppid, csv_path, ppid_condition)

                continue

            if prep_time > 9:
                reason = f'Prep time ({prep_time:.3f}) is too long (> 9) '
                print(f'    {ppid}: {reason}')
                trial_id = make_trial_id(ppid, trial_num, csv_path)

                if trial_id not in skipped_trial_ids:
                    skipped_trial_ids.add(trial_id)
                    addSkippedTrial(skipped_trials, reason,
                                    trial_num, ppid, csv_path, ppid_condition)

                continue

            if cross_time > 1.05:
                reason = f'Cross time({cross_time:.3f}), is too long (> 1.05) '
                print(f'    {ppid}: {reason}')
                trial_id = make_trial_id(ppid, trial_num, csv_path)

                if trial_id not in skipped_trial_ids:
                    skipped_trial_ids.add(trial_id)
                    addSkippedTrial(skipped_trials, reason,
                                    trial_num, ppid, csv_path, ppid_condition)

                continue

            
            trial_nums.append(trial_num)
            # ('<ppid> <condition>', <duration>)

            composite = ppid_condition

            trial_id = make_trial_id(ppid, trial_num, csv_path)

            valid_trials.append(
                {"trial_id": trial_id, "ppid_condition": ppid_condition})

            prep_raw_pairs.append({
                "key": composite,
                "val": prep_time,
                "trial_num": int(trial_num),
                "ppid": ppid,
                "file_path": csv_path,
            })
            cross_raw_pairs.append({
                "key": composite,
                "val": cross_time,
                "trial_num": int(trial_num),
                "ppid": ppid,
                "file_path": csv_path,
            })

            
            
    print(
        f"Raw (ppid_condition, duration) pairs collected: {len(prep_raw_pairs)}")
    print(
        f"Raw (ppid_condition, duration) pairs collected: {len(cross_raw_pairs)}")

    def getGroupStats(raw_pairs, sd_k=2.5, label="prep", skipped_trials=None, skipped_trial_ids=None):
        """
        raw_pairs: list[dict] with keys:
            - "key" (group key, e.g. 'PID 3 LIGHT EXPECTED PRESENT')
            - "val" (duration float)
            - "trial_num" (int)
            - "ppid" (string)
            - "file_path" (string)
        sd_k: SD multiplier for outlier bounds
        label: metric label used for column names + messages ("prep" or "cross")
        skipped_trials: list to append outlier records into
        """

        # ---- Extract arrays from dicts ----
        keys = np.array([d["key"] for d in raw_pairs], dtype=str)
        vals = np.array([d["val"] for d in raw_pairs], dtype=float)

        trial_nums = np.array([int(d["trial_num"])
                              for d in raw_pairs], dtype=int)
        ppids = np.array([d["ppid"] for d in raw_pairs], dtype=str)
        paths = np.array([d["file_path"] for d in raw_pairs], dtype=str)

        # ---- Group stats ----
        unique_keys, inv = np.unique(keys, return_inverse=True)

        counts = np.bincount(inv)
        sum_vals = np.bincount(inv, weights=vals)
        sumsq_vals = np.bincount(inv, weights=vals * vals)

        means = sum_vals / counts
        variances = np.where(
            counts > 1,
            (sumsq_vals - counts * (means ** 2)) / (counts - 1),
            np.nan
        )
        sds = np.sqrt(variances)

        lowers = means - sd_k * sds
        uppers = means + sd_k * sds

        # ---- Map group stats back to each trial ----
        means_per_trial = means[inv]
        sds_per_trial = sds[inv]
        lower_per_trial = lowers[inv]
        upper_per_trial = uppers[inv]
        counts_per_trial = counts[inv]

        # Flag only if SD is defined (n>=2)
        flags = (counts_per_trial > 1) & (
            (vals < lower_per_trial) | (vals > upper_per_trial))

        # ---- Log outliers into skipped_trials ----
        if skipped_trials is not None:
            flagged_indices = np.where(flags)[0]
            for idx in flagged_indices:
                val = float(vals[idx])
                lower = float(lower_per_trial[idx])
                upper = float(upper_per_trial[idx])

                reason = (
                    f"The trial's {label}_time ({val:.3f}) is outside SD bounds | "
                    f"Lower {lower:.3f}, Upper {upper:.3f}"
                )

                trial_id = (str(ppids[idx]).strip(), int(
                    trial_nums[idx]), str(paths[idx]).strip())

                if (skipped_trial_ids is None) or (trial_id not in skipped_trial_ids):
                    if skipped_trial_ids is not None:
                        skipped_trial_ids.add(trial_id)

                    addSkippedTrial(
                        skipped_trials,
                        reason,
                        int(trial_nums[idx]),
                        str(ppids[idx]),
                        str(paths[idx]),
                        str(keys[idx]),  # ppid_condition
                    )

        # ---- Trial-level audit table ----
        raw_trials_df = pd.DataFrame({
            "ppid_condition": keys,
            f"{label}_time": vals,
            "mean_group": means_per_trial,
            "lower": lower_per_trial,
            "upper": upper_per_trial,
            "is_outlier": flags,
            "trial_num": trial_nums,
            "ppid": ppids,
            "file_path": paths,
        })

        # ---- Per-group summary table ----
        flagged_counts = np.bincount(inv, weights=flags.astype(int))

        avg_df = pd.DataFrame({
            "ppid_condition": unique_keys,
            "n_trials": counts,
            f"avg_{label}_time": means,
            f"sd_{label}_time": sds,
            "lower": lowers,
            "upper": uppers,
            "num_flagged_trials": flagged_counts,
        }).sort_values("ppid_condition")

        return avg_df, raw_trials_df

    prep_avg_df, prep_raw_trials_df = getGroupStats(
        prep_raw_pairs, sd_k=2.5, label="prep",
        skipped_trials=skipped_trials,
        skipped_trial_ids=skipped_trial_ids
    )

    cross_avg_df, cross_raw_trials_df = getGroupStats(
        cross_raw_pairs, sd_k=2.5, label="cross",
        skipped_trials=skipped_trials,
        skipped_trial_ids=skipped_trial_ids
    )

    # ---- Final retained counts: valid trials minus anything skipped (rules or SD) ----
    valid_df = pd.DataFrame(valid_trials)

    if not valid_df.empty:
        valid_df["is_skipped"] = valid_df["trial_id"].apply(
            lambda tid: tid in skipped_trial_ids)
        retained_df = valid_df[~valid_df["is_skipped"]]

        retained_counts = (
            retained_df.groupby("ppid_condition")
            .size()
            .reset_index(name="n_retained")
            .sort_values("ppid_condition")
        )

        retained_counts_out = os.path.join(BASE_DIR, "Summaries", "retained_counts_by_condition.csv")
        retained_counts.to_csv(retained_counts_out, index=False)
        print(f"Saved retained counts: {retained_counts_out}")
    else:
        print("No valid trials collected (valid_trials is empty).")

    if save_skipped and skipped_trials:
        skipped_trials_out = os.path.join(BASE_DIR, "Logs", "skipped_trials_timing.csv")

        skipped_df = pd.DataFrame(skipped_trials)
        skipped_df.to_csv(skipped_trials_out, index=False)
        print(f"Saved skipped trials to: {skipped_trials_out}")

    if save:
        prep_out_path = os.path.join(BASE_DIR, "Summaries", "prep_avg_by_ppid.csv")
        prep_avg_df.to_csv(prep_out_path, index=False)
        cross_out_path = os.path.join(BASE_DIR, "Summaries", "cross_avg_by_ppid.csv")
        cross_avg_df.to_csv(cross_out_path, index=False)
        print(f"\nSaved averaged object: {prep_out_path}")

        if save_trial_flags:
            prep_flags_path = os.path.join(
                grand_csv_path, "prep_trials_with_flags.csv")
            prep_raw_trials_df.to_csv(prep_flags_path, index=False)

            cross_flags_path = os.path.join(
                grand_csv_path, "cross_trials_with_flags.csv")
            cross_raw_trials_df.to_csv(cross_flags_path, index=False)
            print(f"Saved trial-level flags: {prep_flags_path}")

    return prep_raw_pairs, prep_avg_df, cross_raw_pairs, cross_avg_df


def main():
    print('Running AveragePrepDuration by PPID')
    path = os.path.join(BASE_DIR, "Grand_CSVs")

    p_raw_pairs, p_avg_df, c_raw_pairs, c_avg_df = grandCSVLoop(
        path, save=True)

    # Preview of the requested list format
    if p_raw_pairs and c_raw_pairs:
        preview_lines = [
            f"'{d['key']}' : {d['val']:.3f}" for d in p_raw_pairs[:10]]
        c_preview_lines = [
            f"'{d['key']}' : {d['val']:.3f}" for d in c_raw_pairs[:10]]

        print("\nRaw (ppid : duration) examples:")
        print("[\n  " + ",\n  ".join(preview_lines) + "\n]")
        print("[\n  " + ",\n  ".join(c_preview_lines) + "\n]")

    if p_avg_df is not None and c_avg_df is not None:
        print("\nAveraged prep time per ppid (first 10 rows):")
        print(p_avg_df.head(10).to_string(index=False))
        print("\nAveraged cross time per ppid (first 10 rows):")
        print(c_avg_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
