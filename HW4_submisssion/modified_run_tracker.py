# Script for testing on chosen sequences from VOT14

# Importing libraries
import time
import cv2
import os
import numpy as np

from sequence_utils import VOTSequence
from particle_filter import PFTracker_1

# Setting parameters to test different motion models (Section C of Assignment)
models = ['RW', 'NCV']
qr_combs = {'RW': [[5*10**(-5), 0.05], [2*10**(-5), 0.1], [10**(-4), 2.5]],
            'NCV': [[5*10**(-5), 0.05], [2*10**(-5), 0.1], [10**(-4), 2.5]]}
n_parts = [100]

# In order to replicate results from Section D of Assignment, uncomment the following 3 lines
# models = ['NCV']
# qr_combs = {'NCV': [[5*10**(-5), 0.05]]}
# n_parts = [25, 80, 100, 200]

# In order to replicate results from Section B of Assignment, uncomment the following 3 lines
# models = ['NCV']
# qr_combs = {'NCV': [[10**(-4), 1]]}
# n_parts = [100]

sequences = ['bolt', 'basketball', 'sphere', 'ball', 'david', 'polarbear']

# 'for' loop which produces the results from either Section C or Section D in the report
for model in models:
    for parts in n_parts:
        print(f"Currently, we're using {parts} particles.")
        for comb in qr_combs[model]:
            print(f'Testing {model} with parameters q={comb[0]} and r={comb[1]}.')
            total_frames = 0
            total_failures = 0
            total_iou = []
            for seq in sequences:
                # set the path to directory where you have the sequences
                dataset_path = os.getcwd()
                sequence = seq  # choose the sequence you want to test

                # visualization and setup parameters
                win_name = 'Tracking window'
                reinitialize = True
                show_gt = True
                video_delay = 15
                font = cv2.FONT_HERSHEY_PLAIN

                # create sequence object
                sequence = VOTSequence(dataset_path, sequence)
                init_frame = 0
                n_failures = 0
                iou_score = []
                tracker = PFTracker_1(model, comb[0], comb[1], parts)

                time_all = 0

                # initialize visualization window
                sequence.initialize_window(win_name)
                # tracking loop - goes over all frames in the video sequence
                frame_idx = 0
                while frame_idx < sequence.length() - 1:
                    img = cv2.imread(sequence.frame(frame_idx))
                    # initialize or track
                    if frame_idx == init_frame:
                        # initialize tracker (at the beginning of the sequence or after tracking failure)
                        t_ = time.time()
                        tracker.initialize(img, sequence.get_annotation(frame_idx, type='rectangle'))
                        time_all += time.time() - t_
                        predicted_bbox = sequence.get_annotation(frame_idx, type='rectangle')
                    else:
                        # track on current frame - predict bounding box
                        t_ = time.time()
                        predicted_bbox = tracker.track(img)
                        time_all += time.time() - t_

                    # calculate overlap (needed to determine failure of a tracker)
                    gt_bb = sequence.get_annotation(frame_idx, type='rectangle')
                    o = sequence.overlap(predicted_bbox, gt_bb)
                    iou_score.append(o)
                    total_iou.append(o)

                    # draw ground-truth and predicted bounding boxes, frame numbers and show image
                    if show_gt:
                        sequence.draw_region(img, gt_bb, (0, 255, 0), 1)
                    sequence.draw_region(img, predicted_bbox, (0, 0, 255), 2)
                    sequence.draw_text(img, '%d/%d' % (frame_idx + 1, sequence.length()), (25, 25))
                    sequence.draw_text(img, 'Fails: %d' % n_failures, (25, 55))
                    sequence.show_image(img, video_delay)

                    if o > 0 or not reinitialize:
                        # increase frame counter by 1
                        frame_idx += 1
                    else:
                        # increase frame counter by 5 and set re-initialization to the next frame
                        frame_idx += 5
                        init_frame = frame_idx
                        n_failures += 1

                iou_score = np.mean(np.array(iou_score))

                sens = 100
                robust = np.exp(-sens * (n_failures / frame_idx))

                print('Tracking speed: %.1f FPS' % (sequence.length() / time_all))
                print('Tracker failed %d times' % n_failures)
                print(f'The accuracy (IoU) of the tracker on sequence {seq} is: {iou_score}.')
                print(f'The robustness of the tracker on sequence {seq} is: {robust}.')

                total_failures += n_failures
                total_frames += frame_idx

            print(f'Total number of faliures of the tracker on all sequences is: {total_failures}.')
            total_robust = np.exp(-sens * (total_failures / total_frames))
            print(f'The robustness of the tracker on all sequences is: {total_robust}.')
            total_iou = np.mean(np.array(total_iou))
            print(f'The accuracy (IoU) of the tracker on all sequences is: {total_iou}.')
