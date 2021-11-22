import cv2
import numpy as np
import random

import tqdm


class DataGenerator:

    def _read_value(self, var_config:dict):
        if var_config['is_static']:
            return var_config['static_value']

        else:
            dynamic = var_config['dynamic']
            if dynamic['is_int']:
                return random.randint(dynamic['range_min'], dynamic['range_max'])

            else:
                return dynamic['range_min'] + (random.random() * (dynamic['range_max'] - dynamic['range_min']))

    def update_vars(self):
        self.number_of_transmitters = self._read_value(self.configuration['transmitters'])
        self.beta = self._read_value(self.configuration['beta'])
        self.noise = self._read_value(self.configuration['noise'])
        self.dimension = self.configuration['dimension']

    def generate_new_problem(self):
        self.update_vars()
        self.transmitters = np.random.rand(self.number_of_transmitters, self.dimension)
        self.transmitter_power = np.ones(self.number_of_transmitters, dtype=float)
        if self.configuration['transmitter_power']['is_static']:
            self.transmitter_power *= self.configuration['transmitter_power']['static_value']
        else:
            # build random vector
            range_min = self.configuration['transmitter_power']['dynamic']['range_min']
            range_max = self.configuration['transmitter_power']['dynamic']['range_max']
            dist = range_max - range_min
            self.transmitter_power = range_min + (dist * np.random.rand(self.number_of_transmitters))

    def __init__(self, problem_configuration: dict):
        self.configuration = problem_configuration
        self.number_of_transmitters = problem_configuration['transmitters']['static_value']
        self.noise = problem_configuration['noise']
        self.dimension = problem_configuration['dimension']
        self.transmitters = None
        self.transmitter_power = None
        self.generate_new_problem()

    @property
    def get_beta(self):
        return self.beta

    @property
    def get_noise(self):
        return self.noise

    @property
    def get_transmitters(self):
        return self.transmitters

    @property
    def get_dimension(self):
        return self.dimension

    def get_batch(self, batch_size: int):
        points = np.random.rand(batch_size, self.dimension)
        one_hot = []
        for point in points:
            curr_one_hot = np.zeros(len(self.transmitters), dtype=int)
            distances = np.linalg.norm(self.transmitters - point, axis=1)
            reception_powers = self.transmitter_power / (distances**2)
            arg_max = reception_powers.argmax()
            max_reception = reception_powers[arg_max]
            if max_reception / (self.noise + (reception_powers.sum() - max_reception)) > self.beta:
                curr_one_hot[arg_max] = 1

            one_hot.append(curr_one_hot)

        return points, np.array(one_hot, dtype=float)

    def draw_problem(self, image_size, colors=None):
        """
        Return an image that describes the current problem configuration
        :return: an image
        """

        if colors is None:
            colors = np.random.randint(0, 255, (self.number_of_transmitters, 3))
        out_image = np.zeros((image_size, image_size, 3), dtype='uint8')
        for i in range(image_size):
            for j in range(image_size):
                x = (i + 0.5) / image_size
                y = (j + 0.5) / image_size
                point = np.array([x, y])
                distances = np.linalg.norm(self.transmitters - point, axis=1)
                reception_powers = self.transmitter_power / (distances**2)
                arg_max = reception_powers.argmax()
                max_reception = reception_powers[arg_max]
                if max_reception / (self.noise + (reception_powers.sum() - max_reception)) > self.beta:
                    out_image[i,j, :] = colors[arg_max]

        return out_image, colors






