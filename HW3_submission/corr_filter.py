# Importing libraries
import cv2
import numpy as np
from numpy.fft import fft2, ifft2

import sys
# The path needs to be changed to path on local machine
sys.path.insert(1, 'C:/Users/User/Desktop/Fakultet/2.Semester/AMCV/HW3/toolkit-dir/utils')

from tracker import Tracker

from ex3_utils import create_cosine_window, create_gauss_peak, get_patch, normalize_image

# Correlation filter tracker class
class CFTracker_1(Tracker):

    def __init__(self):
        self.enlarge_factor = 1
        self.lambda_ = 10 ** (-3)
        self.alpha = 0
        self.sigma = 1

    def name(self):
        # return 'sigma_0.5'     # this line is used when testing for different alphas and sigmas
        return 'factor_1'

    def initialize(self, image, region):

        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = normalize_image(image)

        left = int(region[0])
        top = int(region[1])
        self.width = int(region[2] * self.enlarge_factor)
        self.height = int(region[3] * self.enlarge_factor)

        self.position = np.array((left + int(self.width / 2), top + int(self.height / 2)))
        f = get_patch(image, self.position, (self.width, self.height))[0]

        self.cos_window = create_cosine_window((self.width, self.height))
        f = f * self.cos_window
        f_fft = np.conjugate(fft2(f))

        g = create_gauss_peak((self.width, self.height), self.sigma)
        self.g_fft = fft2(g)

        min_width = min(self.g_fft.shape[1], f_fft.shape[1])
        min_height = min(self.g_fft.shape[0], f_fft.shape[0])
        self.g_fft = self.g_fft[: min_height, : min_width]
        f_fft = f_fft[: min_height, : min_width]

        self.h_fft = (self.g_fft * f_fft) / (np.conjugate(f_fft) * f_fft + self.lambda_)

    def track(self, image):

        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = normalize_image(image)

        f = get_patch(image, self.position, (self.width, self.height))[0]
        # this next line is only due one error with test sequence diving when alpha=0.2 and sigma=3
        f = f[: self.height, : self.width]
        f_fft = fft2(f * self.cos_window)

        peak_matrix = ifft2(f_fft * self.h_fft)
        y_max, x_max = np.unravel_index(np.argmax(peak_matrix, axis=None), peak_matrix.shape)
        if x_max > (self.width / 2):
            x_max = x_max - self.width
        if y_max > (self.height / 2):
            y_max = y_max - self.height
        self.position = self.position + np.array([x_max, y_max])

        f = get_patch(image, self.position, (self.width, self.height))[0]
        # this next ine is only due one error with test sequence diving when alpha=0.2 and sigma=3
        f = f[: self.height, : self.width]

        f_fft = fft2(f * self.cos_window)
        h_fft_new = (self.g_fft * np.conjugate(f_fft)) / (f_fft * np.conjugate(f_fft) + self.lambda_)
        self.h_fft = self.alpha * self.h_fft + (1 - self.alpha) * h_fft_new

        tracker_width = self.width / self.enlarge_factor
        tracker_height = self.height / self.enlarge_factor

        left = max(0, self.position[0] - int(tracker_width / 2))
        top = max(0, self.position[1] - int(tracker_height / 2))

        return [left, top, tracker_width, tracker_height]

# The following 4 classes inherit from CFTracker_1.
# This kind of class structure was most convenient for testing.
# We would just change parameter alpha in CFTracker_1 and be able to test for a particular alpha and all of the chosen
# sigmas at once.
# So, in order to test for different sigmas, just set a value for alpha in line 20 and uncomment lines 95 - 130.
# class CFTracker_2(CFTracker_1):
#
#     def __init__(self):
#         CFTracker_1.__init__(self)
#         self.sigma = 0.75
#
#     def name(self):
#         return 'sigma_0.75'
#
# class CFTracker_3(CFTracker_1):
#
#     def __init__(self):
#         CFTracker_1.__init__(self)
#         self.sigma = 1
#
#     def name(self):
#         return 'sigma_1'
#
# class CFTracker_4(CFTracker_1):
#
#     def __init__(self):
#         CFTracker_1.__init__(self)
#         self.sigma = 3
#
#     def name(self):
#         return 'sigma_3'
#
#
# class CFTracker_5(CFTracker_1):
#
#     def __init__(self):
#         CFTracker_1.__init__(self)
#         self.sigma = 5
#
#     def name(self):
#         return 'sigma_5'

# Here we have the same structure as above.
# We just change sigma and alpha in CFTracker_1 and are able to test for all the enlargement factors at once.
class CFTracker_2(CFTracker_1):

    def __init__(self):
        CFTracker_1.__init__(self)
        self.enlarge_factor = 1.1

    def name(self):
        return 'factor_1.1'

class CFTracker_3(CFTracker_1):

    def __init__(self):
        CFTracker_1.__init__(self)
        self.enlarge_factor = 1.2

    def name(self):
        return 'factor_1.2'

class CFTracker_4(CFTracker_1):

    def __init__(self):
        CFTracker_1.__init__(self)
        self.enlarge_factor = 1.3

    def name(self):
        return 'factor_1.3'

class CFTracker_5(CFTracker_1):

    def __init__(self):
        CFTracker_1.__init__(self)
        self.enlarge_factor = 1.5

    def name(self):
        return 'factor_1.5'

class CFTracker_6(CFTracker_1):

    def __init__(self):
        CFTracker_1.__init__(self)
        self.enlarge_factor = 2

    def name(self):
        return 'factor_2'

