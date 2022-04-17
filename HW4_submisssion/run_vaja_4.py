import numpy as np
import math
import matplotlib.pyplot as plt

from ex4_utils import kalman_step
from dynamic_models import NCV, RW, NCA

trajects = ['spiral', 'rectangle', 'RW']

for traject in trajects:
    if traject == 'spiral':
        N = 40
        v = np.linspace(5 * math.pi, 0, N)
        x = np.cos(v) * v
        y = np.sin(v) * v
    elif traject == 'rectangle':
        N = 14
        x, y = np.zeros(N), np.zeros(N)

        # Constant velocity part
        x[: 6] = np.arange(5, 11, 1)
        y[: 6] = 5 * np.ones(6)
        x[6:9] = 10 * np.ones(3)
        y[6:9] = np.arange(6, 9, 1)

        # Constant acceleration part
        x[9:12] = np.array([10 - (1 + 0.33), 10 - (2 + 0.33 * 3), 10 - (3 + 0.33 * 6)])
        y[9:12] = 8 * np.ones(3)
        x[12] = 5
        y[12] = 8 - (1 + 0.33 * 4)

        # Coming back to starting point
        x[13], y[13] = 5, 5

    else:
        np.random.seed(0)
        N = 15
        x, y = np.arange(N), np.zeros(N)
        steps = [-1, 0, 1]

        # Generating a random movement
        gener_steps = np.random.choice(a=steps, size=N-1)
        y[1:] = gener_steps
        y = y.cumsum(axis=0)

    sx = np.zeros((x.size, 1), dtype=np.float32).flatten()
    sy = np.zeros((y.size, 1), dtype=np.float32).flatten()

    sx[0] = x[0]
    sy[0] = y[0]

    # Parameter combinations
    qr_combs = [[100, 1], [5, 1], [1, 1], [1, 5], [1, 100]]
    models = ['RW', 'NCV', 'NCA']
    if traject == 'spiral':
        fig, (ax_1, ax_2, ax_3) = plt.subplots(3, 5, figsize=(15, 15))
    elif traject == 'rectangle':
        fig, (ax_1, ax_2, ax_3) = plt.subplots(3, 5, figsize=(17, 8))
    else:
        fig, (ax_1, ax_2, ax_3) = plt.subplots(3, 5, figsize=(15, 15))

    plt.subplots_adjust(hspace=0.3)

    for model in models:
        for (i, comb) in enumerate(qr_combs):

            if model == 'RW':
                A, C, Q_i, R_i = RW(q_val=comb[0], r_val=comb[1])
                temp_ax = ax_1
            elif model == 'NCV':
                A, C, Q_i, R_i = NCV(q_val=comb[0], r_val=comb[1])
                temp_ax = ax_2
            else:
                A, C, Q_i, R_i = NCA(q_val=comb[0], r_val=comb[1])
                temp_ax = ax_3

            state = np.zeros((A.shape[0], 1), dtype=np.float32).flatten()
            state[0] = x[0]
            state[1] = y[0]
            covariance = np.eye(A.shape[0], dtype=np.float32)

            for j in range(1, x.size):
                state, covariance, _, _ = kalman_step(A, C, Q_i, R_i,
                                                      np.reshape(np.array([x[j], y[j]]), (-1, 1)),
                                                      np.reshape(state, (-1, 1)),
                                                      covariance)
                sx[j] = state[0]
                sy[j] = state[1]

            temp_ax[i].plot(x, y, c='red', linestyle='-', marker='o', markersize=5)
            temp_ax[i].plot(sx, sy, c='blue', linestyle='-', marker='o', markersize=5)
            temp_ax[i].set_title(f'{model}: q={comb[0]}, r={comb[1]}')

    plt.show()








