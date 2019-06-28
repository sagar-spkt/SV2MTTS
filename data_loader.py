import numpy as np
from tensorflow.python.keras.utils import Sequence


class RandomTrainGenerator(Sequence):
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def __len__(self):
        return 50

    def __getitem__(self, index):
        return [np.random.randint(0, 32, (self.batch_size, 23)),
                np.random.rand(self.batch_size, 3, 160, 40),
                np.random.rand(self.batch_size, 33, 400)], \
               [np.random.rand(self.batch_size, 33, 400),
                np.random.rand(self.batch_size, 33*5, 1025)]

