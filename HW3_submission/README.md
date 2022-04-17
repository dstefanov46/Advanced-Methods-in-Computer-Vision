## Exercise 3: Correlation filters and tracking evaluation

### Commands to reproduce the results from the first part of the assignment
python evaluate_tracker.py --workspace_path workspace-dir --tracker sigma_0.5 && python evaluate_tracker.py --workspace_path workspace-dir --tracker sigma_0.75 && python evaluate_tracker.py --workspace_path workspace-dir --tracker sigma_1 && python evaluate_tracker.py --workspace_path workspace-dir --tracker sigma_3 && python evaluate_tracker.py --workspace_path workspace-dir --tracker sigma_5

python compare_trackers.py --workspace_path workspace-dir --trackers sigma_0.5 sigma_0.75 sigma_1 sigma_3 sigma_5 --sensitivity 100

### Commands to reproduce the results from the second part of the assignment 
python evaluate_tracker.py --workspace_path workspace-dir --tracker factor_1 && python evaluate_tracker.py --workspace_path workspace-dir --tracker factor_1.1 && python evaluate_tracker.py --workspace_path workspace-dir --tracker factor_1.2 && python evaluate_tracker.py --workspace_path workspace-dir --tracker factor_1.3 && python evaluate_tracker.py --workspace_path workspace-dir --tracker factor_1.5 && python evaluate_tracker.py --workspace_path workspace-dir --tracker factor_2

python compare_trackers.py --workspace_path workspace-dir --trackers factor_1 factor_1.1 factor_1.2 factor_1.3 factor_1.5 factor_2 --sensitivity 100

- There has also been a change to the file plot_styles.py as it only had 5 types of markers, and in the 
second part of our assignment we needed a plot with 6 trackers.