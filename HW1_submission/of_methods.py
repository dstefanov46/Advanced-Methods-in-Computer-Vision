# Importing libraries
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import math

# Importing helper functions
from ex1_utils import gaussderiv, gausssmooth, show_flow, normalize_image

# Definition of lucas_kanade function
def lucas_kanade(im1, im2, N):
    # Image normalization
    im1 = normalize_image(im1)
    im2 = normalize_image(im2)

    # Defining initial variables
    conv_mask = np.ones((N, N))

    # Smoothing of both images
    im1 = gausssmooth(im1, 1)
    im2 = gausssmooth(im2, 1)

    # Computation of temporal and spatial derivatives (and some improvement on them)
    it = gausssmooth(im2 - im1, 1)
    ix = (gaussderiv(im1, 1)[0] + gaussderiv(im2, 1)[0]) / 2
    iy = (gaussderiv(im1, 1)[1] + gaussderiv(im2, 1)[1]) / 2

    # Computation of sums for the formulas for u and v
    ix2 = signal.convolve2d(ix ** 2, conv_mask, mode='same', boundary='symm')
    iy2 = signal.convolve2d(iy ** 2, conv_mask, mode='same', boundary='symm')
    ixy = signal.convolve2d(ix * iy, conv_mask, mode='same', boundary='symm')
    ixt = signal.convolve2d(ix * it, conv_mask, mode='same', boundary='symm')
    iyt = signal.convolve2d(iy * it, conv_mask, mode='same', boundary='symm')

    # Computation of the determinant
    D = ix2 * iy2 - ixy ** 2

    # Replacing zeros with a small value
    D[D == 0] = math.pow(10, -10)

    # Computation of trace of structure tensor for Harris detector
    tr = ix2 + iy2

    # Computation of Harris detector response
    k = 0.05
    resp = D - k * tr**2

    # Dividing pixels into reliable and non-reliable optical flow
    resp[resp >= - 0.5 * math.pow(10, -2)] = 0
    resp[resp < - 0.5 * math.pow(10, -2)] = 1
    # fig, ax = plt.subplots(1, 1)
    # ax.imshow(resp)

    # Computation of u and v
    u = - (iy2 * ixt - ixy * iyt) / D
    v = - (ix2 * iyt - ixy * ixt) / D

    return u, v

# Definition of function horn_schunck
def horn_schunck(im1, im2, n_iters, lmbd):
    # Image normalization
    im1 = normalize_image(im1)
    im2 = normalize_image(im2)

    # Smoothing the images first
    im1 = gausssmooth(im1, 1)
    im2 = gausssmooth(im2, 1)

    # Defining initial variables
    ua = np.zeros(im1.shape)
    va = np.zeros(im1.shape)
    ld = np.array([[0, 0.25, 0],
                   [0.25, 0, 0.25],
                   [0, 0.25, 0]])
    lmbd = lmbd * np.ones(im1.shape)

    # Computing and smoothing derivatives
    # Derivatives as suggested in the lectures
    # it = signal.convolve2d(im2 - im1, 0.25 * np.ones((2, 2)), mode='same', boundary='symm')
    # ix_mask = np.array([[-0.5, 0.5],
    #                     [-0.5, 0.5]])
    # ix = signal.convolve2d(im1, ix_mask, mode='same', boundary='symm')
    # iy_mask = np.array([[-0.5, -0.5],
    #                     [0.5, 0.5]])
    # iy = signal.convolve2d(im1, iy_mask, mode='same', boundary='symm')

    # Derivatives by the assignment instructions
    it = gausssmooth(im2 - im1, 1)
    ix = (gaussderiv(im1, 1)[0] + gaussderiv(im2, 1)[0]) / 2
    iy = (gaussderiv(im1, 1)[1] + gaussderiv(im2, 1)[1]) / 2

    # Initializing P and D
    p = ix * ua + iy * va + it
    d = lmbd + ix**2 + iy**2

    # Storing new values for u and v for later in the function
    # u_vals, v_vals = [], []

    # Iteration loop
    for i in range(n_iters):
        # Calculation of u and v
        u = ua - ix * p / d
        v = va - iy * p / d

        # Adding new u, v to the storage array
        # u_vals.append(u)
        # v_vals.append(v)

        # Recalculating ua, va and P
        ua = signal.convolve2d(u, ld, mode='same', boundary='symm')
        va = signal.convolve2d(v, ld, mode='same', boundary='symm')
        p = ix * ua + iy * va + it

        # Plots on every 100 iterations to see the progression of the algorithm
        # if (i % 100) == 0:
        #     fig1, (ax1, ax2) = plt.subplots(1, 2)
        #     show_flow(u, v, ax1, type='angle')
        #     show_flow(u, v, ax2, type='field', set_aspect=True)

    # See where the algorithm converges
    # conv_iter = 0
    # for i in range(len(u_vals) - 1):
    #     if np.linalg.norm(u_vals[i + 1] - u_vals[i], 'fro') >= math.pow(10, -3) or \
    #        np.linalg.norm(v_vals[i + 1] - v_vals[i], 'fro') >= math.pow(10, -3):
    #         pass
    #     else:
    #         conv_iter = i + 1
    #         break

    return u, v #, conv_iter



