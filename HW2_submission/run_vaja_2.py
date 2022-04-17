import numpy as np
import matplotlib.pyplot as plt

from ex1_utils import gausssmooth
from ex2_utils import generate_responses_1, generate_responses_2, get_patch, mode_seeking

# Testing how our mode seeking function works on the output of the get_responses_1() function
print('# Testing how our mode seeking function works on the output of the get_responses_1() function')
im = generate_responses_1()
# print(im.shape, len(im.shape))
# print(np.argwhere(im == im.max()))
fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8))
ax.imshow(im)
plt.show()

kern_sizes = [5, 9, 15, 21]
points = [np.array([50, 50]),
          np.array([65, 50]),
          np.array([20, 80]),
          np.array([80, 20])]
kern_shapes = ['square', 'elliptical']
stop_crit = {5: 0.01,
             9: 0.05,
             15: 0.1,
             21: 0.3}

for shape in kern_shapes:
    for size in kern_sizes:
        for point in points:
            max_pos, n_iters = mode_seeking(im, size, point, shape, stop_crit[size], 1)
            print(f'{shape} kernel of size {size} starting at point {list(point)} found max at {max_pos} in {n_iters} iterations.')

print('')
print('')

# Testing how our mode seeking function works on our own probability density function
print('Testing how our mode seeking function works on our own probability density function')
im = generate_responses_2()
fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8))
ax.imshow(im)
plt.show()

kern_sizes = [9, 15, 21]
points = [np.array([20, 50]),
          np.array([37, 70]),
          np.array([87, 87])]
kern_shapes = ['square', 'elliptical']
stop_crit = {9: 0.05,
             15: 0.1,
             21: 0.3}

for shape in kern_shapes:
    for size in kern_sizes:
        for point in points:
            max_pos, n_iters = mode_seeking(im, size, point, shape, stop_crit[size], 1)
            print(f'{shape} kernel of size {size} starting at point {list(point)} found max at {max_pos} in {n_iters} iterations.')

# To reproduce the results we obtained when testing on the video sequences, see file modified_run_tracker.py

