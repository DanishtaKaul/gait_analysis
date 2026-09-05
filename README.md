# Effect of Dim Lighting on Walking and Obstacle Navigation

This repository contains the gait analysis code for a study investigating the effects of dim lighting on walking and obstacle navigation in young and older adults. The pipeline processes motion-capture data into gait features and analyses how lighting and obstacle conditions affect walking performance.

**No participant data is included in this repository**. The scripts
read from local data folders that are not committed. Paths are set at the top of each
script and in `config.py`.

## Pipeline order

Run in this order - each script writes files the next one reads.

1. `GetGrandCSVFiles.py` - merge trial metadata with tracker CSVs into one CSV per trial.
2. `GetAveragePreparationDuration.py` - compute preparation and crossing times per trial; flag outliers and log skipped trials.
3. `detect_heel_strikes.py` - detect heel strikes, with a manual check, and save a per-trial time series.
4. `from_motion_tracking_data_splice_x_y_to_time_series.py` - add x and y position channels to the time series.
5. `min_max_step_count_check.py` - flag trials with too many, too few, or duplicate steps.
6. `random_forest_dataset.py` - build the cleaned trial list and retained-trial counts.
7. `pooled_gait_features.py` - extract gait measures, remove outliers, and pool into one row per participant and condition. Measures: speed, stride length, stride time, step width, cadence, lateral deviation, stride length CV, stride time CV, and step width SD.
Steps 1 to 5 also produce the time series used by the behavioural analysis in the [eeg_analysis](https://github.com/DanishtaKaul/eeg_analysis) repository.
## Analysis

- `gait_mixed_models.R` - linear mixed-effects models.
- `rf_age_condition_level_model.py` - random forest classifying age group from gait features.

## Dependencies

- `config.py` - constants and data paths.
- `helper_functions.py` - shared functions used across the pipeline.

## Author

Danishta Kaul
