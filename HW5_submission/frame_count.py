# This script is used to count the frames used for re-detection for each of the different test cases (num_samples = [5,
# 10, 20, 30, 50])

import os
import numpy as np

cd = os.getcwd()
num_samples = [5, 10, 20, 30, 50]
frame_counts = dict.fromkeys(num_samples)

for num in num_samples:
    num_frames = 0
    curr_path = cd + f'/car9_results/thresh_5_{num}/car9_scores.txt'
    file = open(curr_path, 'r')
    scores = file.readlines()
    for score in scores:
        score = score.replace('\n', '')
        if float(score) == 0:
            num_frames += 1
    frame_counts[num] = num_frames

print(frame_counts)
