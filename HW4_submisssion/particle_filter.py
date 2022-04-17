# Importing libraries
import numpy as np
from gensim.matutils import hellinger

from ex4_utils import get_patch, create_epanechnik_kernel, extract_histogram, sample_gauss, get_equal_sizes
from dynamic_models import NCV, RW, NCA

# Particle filter tracker class
class PFTracker_1:

    def __init__(self, model, q_val, r_val, n_parts):
        self.alpha = 0.1
        self.sigma = 0.5
        self.n_parts = n_parts
        self.q = q_val
        self.r = r_val
        self.model = model

    def initialize(self, image, region):

        left = int(region[0])
        top = int(region[1])
        self.width = int(region[2])
        self.height = int(region[3])

        self.position = np.array((left + int(self.width / 2), top + int(self.height / 2)))
        template = get_patch(image, self.position, (self.width, self.height))[0]

        self.kernel = create_epanechnik_kernel(self.width, self.height, self.sigma)
        if template.shape[: 2] != self.kernel.shape:
            template, self.kernel = get_equal_sizes(template, self.kernel)

        self.tar_hist = extract_histogram(template, 16, self.kernel)
        self.tar_hist = self.tar_hist / self.tar_hist.sum()

        # We generate different matrices based on the motion model we use
        if self.model == 'NCV':
            self.A, self.C, self.Q, self.R = NCV(q_val=self.q * (self.width * self.height), r_val=self.r)
        elif self.model == 'RW':
            self.A, self.C, self.Q, self.R = RW(q_val=self.q * (self.width * self.height), r_val=self.r)
        else:
            self.A, self.C, self.Q, self.R = NCA(q_val=self.q * (self.width * self.height), r_val=self.r)

        # We generate matrix with the state of every particle
        part_pos = np.array([self.position for _ in range(self.n_parts)])
        if self.model == 'NCA':
            self.particles = np.hstack([part_pos, np.zeros((len(part_pos), 4))])
        else:
            self.particles = np.hstack([part_pos, np.zeros((len(part_pos), 2))])
        for i in range(self.n_parts):
            self.particles[i, :] += sample_gauss(np.zeros(len(self.Q)), self.Q, 1).flatten()

        if self.model == 'NCA':
            self.particles[:, 2:] = np.zeros((self.n_parts, 4))
        else:
            self.particles[:, 2:] = np.zeros((self.n_parts, 2))

        # We initialize the weights for the particles
        self.part_weigths = np.ones(self.n_parts)

    def track(self, image):

        weights_norm = self.part_weigths / np.sum(self.part_weigths)
        weights_cumsumed = np.cumsum(weights_norm)
        rand_samples = np.random.rand(self.n_parts, 1)
        sampled_idxs = np.digitize(rand_samples, weights_cumsumed)
        self.particles = self.particles[sampled_idxs.flatten(), :]

        # We perform the movement of the particles
        self.particles = np.matmul(self.A, self.particles.T)
        for i in range(self.n_parts):
            self.particles[:, i] += sample_gauss(np.zeros(len(self.Q)), self.Q, 1).flatten()
        self.particles = self.particles.T

        # The weights of the particles are updated
        for i in range(self.n_parts):
            temp_patch = get_patch(image, self.particles[i, : 2], (self.width, self.height))[0]
            if temp_patch.shape[: 2] != self.kernel.shape:
                temp_patch, self.kernel = get_equal_sizes(temp_patch, self.kernel)
            temp_hist = extract_histogram(temp_patch, 16, self.kernel)
            temp_hist = temp_hist / temp_hist.sum()
            hell_dist = hellinger(temp_hist, self.tar_hist)
            self.part_weigths[i] = np.exp(-0.5 * hell_dist ** 2 / self.sigma ** 2)

        self.part_weigths = self.part_weigths / self.part_weigths.sum()

        # The target position is updated
        self.position = np.sum(self.particles[:, : 2] * self.part_weigths.reshape(-1, 1), axis=0)

        # New temporary target histogram is obtained
        new_template = get_patch(image, self.position, (self.width, self.height))[0]
        if new_template.shape[: 2] != self.kernel.shape:
            new_template, self.kernel = get_equal_sizes(new_template, self.kernel)
        new_hist = extract_histogram(new_template, 16, self.kernel)
        new_hist = new_hist / new_hist.sum()

        # Target histogram is updated
        self.tar_hist = (1 - self.alpha) * self.tar_hist + self.alpha * new_hist

        left = max(0, self.position[0] - int(self.width / 2))
        top = max(0, self.position[1] - int(self.height / 2))

        return [left, top, self.width, self.height]


