import numpy as np

from ex2_utils import Tracker, extract_histogram, create_epanechnik_kernel, backproject_histogram, get_patch

# Parameters for mean-shift tracker
class MSParams():
    def __init__(self):
        self.enlarge_factor = 2

# Mean-shift tracker class
class MeanShiftTracker(Tracker):

    def initialize(self, image, region):

        left = int(region[0])
        top = int(region[1])
        self.width = int(region[2])
        self.height = int(region[3])

        self.position = np.array((left + int(self.width / 2), top + int(self.height / 2)))
        template = get_patch(image, self.position, (self.width, self.height))[0]

        self.kernel = create_epanechnik_kernel(self.width, self.height, 1)
        template = template[:,
                            : min(template.shape[0], self.kernel.shape[0]),
                            : min(template.shape[1], self.kernel.shape[1])]
        self.kernel = self.kernel[: min(template.shape[0], self.kernel.shape[0]),
                                  : min(template.shape[1], self.kernel.shape[1])]

        self.q = extract_histogram(template, 16, self.kernel)
        self.q = self.q / (self.q).sum()

        x_coeff = np.arange(- 1 * (self.width - 1) / 2, (self.width - 1) / 2 + 1)
        y_coeff = np.arange(- 1 * (self.height - 1) / 2, (self.height - 1) / 2 + 1)
        self.x, self.y = np.meshgrid(x_coeff, y_coeff)

        self.eps = 10 ** (- 5)
        self.alpha = 0 # 0.1
        # self.stop_crit = 10

    def track(self, image):

        # shift_dist = 10
        # while shift_dist >= self.stop_crit:
        for i in range(10):
            template = get_patch(image, self.position, (self.width, self.height))[0]
            template = template[:,
                                : min(template.shape[0], self.kernel.shape[0]),
                                : min(template.shape[1], self.kernel.shape[1])]

            p = extract_histogram(template, 16, self.kernel)
            p = p / p.sum()

            p_eps = p + self.eps
            v = np.sqrt(self.q / p_eps)

            w = backproject_histogram(template, v, 16)

            w = w[: min(w.shape[0], self.x.shape[0]),
                  : min(w.shape[1], self.x.shape[1])]
            x = self.x[: min(w.shape[0], self.x.shape[0]),
                       : min(w.shape[1], self.x.shape[1])]
            y = self.y[: min(w.shape[0], self.x.shape[0]),
                       : min(w.shape[1], self.x.shape[1])]
            shift = np.array(((x * w).sum() / w.sum(),
                              (y * w).sum() / w.sum()))
            # shift_dist = np.linalg.norm(shift)

            self.position = self.position + shift

        q_tilde = extract_histogram(template, 16, self.kernel)
        q_tilde = q_tilde / q_tilde.sum()

        self.q = (1 - self.alpha) * self.q + self.alpha * q_tilde

        left = max(0, self.position[0] - int(self.width / 2))
        top = max(0, self.position[1] - int(self.height / 2))

        return [left, top, self.width, self.height]
