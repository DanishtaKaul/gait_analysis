"""Add medial-lateral (x) and vertical (y) position columns to the saved heel-strike time series."""

import os
import glob
import kineticstoolkit.lab as ktk
from helper_functions import (
    extract_forewarn, extract_light, preprocess_motion_tracking_file)


BASE_DIR = r"D:\Gait_Analysis"
BASE_PATH = r"D:\motion_tracking_data"

# --- Configuration ---
COLUMN_MAP = {
    "pel_ml":   "Skeleton 001:Skeleton 001 Pos[0]",
    "pel_y":    "Skeleton 001:Skeleton 001 Pos[1]",
    "rfoot_ml": "Skeleton 001:RFoot Pos[0]",
    "rfoot_y":  "Skeleton 001:RFoot Pos[1]",
    "lfoot_ml": "Skeleton 001:LFoot Pos[0]",
    "lfoot_y":  "Skeleton 001:LFoot Pos[1]",
}


def get_start_end_time(ts):
    if not ts.events:
        return None, None
    return ts.events[0].time, ts.events[-1].time


if __name__ == "__main__":
    TS_PATH = os.path.join(BASE_DIR, "time_series_data_x_y_added")

    # 1. Flatten the file list for better progress tracking
    ts_files = glob.glob(os.path.join(TS_PATH, "**", "*.zip"), recursive=True)
    total = len(ts_files)

    # 2. Setup a simple cache
    current_csv_path = None
    cached_df = None

    for i, ts_path in enumerate(ts_files):
        ts_file_name = os.path.basename(ts_path)
        print(f'\t Processing {ts_file_name}')
        # Progress logging
        if i % 10 == 0:
            print(f"PROGRESS: {(i/total)*100:.1f}% ({i}/{total})")

        # Load TimeSeries
        ts = ktk.load(ts_path)
        t_start, t_end = get_start_end_time(ts)
        if t_start is None:
            print(f"WARNING: No events found in {ts_file_name}, skipping.")
            continue

        # Identify required CSV
        pid = os.path.basename(os.path.dirname(ts_path))
        ts_filename = os.path.basename(ts_path)

        # Build CSV path
        light = extract_light(ts_filename)
        forewarn = extract_forewarn(ts_filename)
        csv_path = os.path.join(BASE_PATH, pid,
                                f"{pid.upper()} {light.upper()} {forewarn.upper()}.csv")

        # 3. CACHING LOGIC: Only load if it's a new file
        if csv_path != current_csv_path:
            if os.path.exists(csv_path):
                print(f"\n --- Loading CSV: {os.path.basename(csv_path)}")
                cached_df, _ = preprocess_motion_tracking_file(csv_path)
                current_csv_path = csv_path
            else:
                print(f"!!! Missing: {csv_path}")
                continue

        # 4. Splicing with reindex 
        try:
            
            spliced = cached_df.reindex(ts.time, method='nearest')

            for ts_key, csv_col in COLUMN_MAP.items():
                ts.data[ts_key] = spliced[csv_col].values

            ktk.save(ts_path, ts)
            print(f'\t Saved {ts_file_name}')
        except Exception as e:
            print(f"Error on {ts_filename}: {e}")

    print("Processing Complete")
