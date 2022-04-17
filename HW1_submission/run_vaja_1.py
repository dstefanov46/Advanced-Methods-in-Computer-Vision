# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
import cProfile
import pstats
from pstats import SortKey

# Importing helper functions
from ex1_utils import rotate_image, show_flow, normalize_image
from of_methods import lucas_kanade, horn_schunck

# Generating images of Gaussian noise
im1 = np.random.rand(200, 200).astype(np.float32)
im2 = im1.copy()
im2 = rotate_image(im2, 1)

# Importing 3x2 images to test functioning of algorithms
# im1 = cv2.imread('./disparity/cporta_left.png', flags=cv2.IMREAD_GRAYSCALE)
# im2 = cv2.imread('./disparity/cporta_right.png', flags=cv2.IMREAD_GRAYSCALE)
# im1 = cv2.imread('./disparity/office_left.png', flags=cv2.IMREAD_GRAYSCALE)
# im2 = cv2.imread('./disparity/office_right.png', flags=cv2.IMREAD_GRAYSCALE)
# im1 = cv2.imread('./disparity/office2_left.png', flags=cv2.IMREAD_GRAYSCALE)
# im2 = cv2.imread('./disparity/office2_right.png', flags=cv2.IMREAD_GRAYSCALE)

# Defining a list needed for creation of Figure 2 in the report
imgs = [('./disparity/cporta_left.png', './disparity/cporta_right.png'),
        ('./disparity/cporta_left.png', './disparity/cporta_right.png'),
        ('./disparity/office_left.png', './disparity/office_right.png'),
        ('./disparity/office2_left.png', './disparity/office2_right.png')]

# Calculating optical flow
# U_lk, V_lk = lucas_kanade(im1, im2, 3)
# U_hs, V_hs = horn_schunck(im1, im2, 1000, 0.5)
# print(U_hs, V_hs, conv_iter)

# Showing results of Lucas-Kanade method
# fig1, ((ax1_11, ax1_12), (ax1_21, ax1_22)) = plt.subplots(2, 2)
# ax1_11.imshow(im1)
# ax1_12.imshow(im2)
# show_flow(U_lk, V_lk, ax1_21, type='angle')
# show_flow(U_lk, V_lk, ax1_22, type='field', set_aspect=True)
# fig1.suptitle('Lucas-Kanade Optical Flow')

# Showing results of Horn-Schunck method
# fig2, ((ax1_11, ax1_12), (ax1_21, ax1_22)) = plt.subplots(2, 2)
# ax1_11.imshow(im1)
# ax1_12.imshow(im2)
# show_flow(U_hs, V_hs, ax1_21, type='angle')
# show_flow(U_hs, V_hs, ax1_22, type='field', set_aspect=True)
# fig2.suptitle('Horn-Schunk Optical Flow')

# Generation of Figure 1 in the report
fig, ((ax_11, ax_12), (ax_21, ax_22), (ax_31, ax_32)) = plt.subplots(3, 2, figsize=(6.4, 4.8))
fig.tight_layout(pad=0.05)
ax_11.imshow(im1)
ax_12.imshow(im2)
ax_11.title.set_text('Random noise')
ax_12.title.set_text('Rotated image')
U_lk, V_lk = lucas_kanade(im1, im2, 3)
show_flow(U_lk, V_lk, ax_21, type='angle')
show_flow(U_lk, V_lk, ax_22, type='field', set_aspect=True)
ax_21.title.set_text('Lucas-Kanade: Angle')
ax_22.title.set_text('Lucas-Kanade: Motion Field')
U_hs, V_hs = horn_schunck(im1, im2, 1000, 0.5)
show_flow(U_hs, V_hs, ax_31, type='angle')
show_flow(U_hs, V_hs, ax_32, type='field', set_aspect=True)
ax_31.title.set_text('Horn-Schunck: Angle')
ax_32.title.set_text('Horn-Schunck: Motion Field')

# Generation of Figure 2 in the report
fig, ax = plt.subplots(4, 2, figsize=(8, 12.8))
for i in range(4):
    im1 = cv2.imread(imgs[i][0], flags=cv2.IMREAD_GRAYSCALE)
    im2 = cv2.imread(imgs[i][1], flags=cv2.IMREAD_GRAYSCALE)
    if i == 0:
        ax[i][0].imshow(im1)
        # Drawing vertical dashed line
        ax[i][0].plot([24, 25], [160, 420], ls='--', color='red', linewidth=3)
        ax[i][1].imshow(im2)
        # Drawing vertical dashed line
        ax[i][1].plot([49, 50], [160, 420], ls='--', color='red', linewidth=3)
    elif i == 1:
        U_lk, V_lk = lucas_kanade(im1, im2, 3)
        U_hs, V_hs = horn_schunck(im1, im2, 1000, 0.5)
        show_flow(U_lk, V_lk, ax[i][0], type='field', set_aspect=True)
        show_flow(U_hs, V_hs, ax[i][1], type='field', set_aspect=True)
    else:
        U_hs, V_hs = horn_schunck(im1, im2, 1000, 0.5)
        ax[i][0].imshow(im1)
        show_flow(U_hs, V_hs, ax[i][1], type='field', set_aspect=True)

plt.show()

# The code which was used to measure the time of algorithm execution for different parameters
# cProfile.run('lucas_kanade(im1, im2, 3)', 'lucas_kanade')
# p = pstats.Stats('lucas_kanade')
# p.sort_stats(SortKey.TIME).print_stats(10)
