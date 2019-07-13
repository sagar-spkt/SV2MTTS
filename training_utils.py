import glob
import os
import re
import numpy as np
from tensorflow.python.keras.callbacks import LearningRateScheduler

import hparams


def get_init_epoch(check_point_path):
    check_point_list = glob.glob(os.path.join(check_point_path, 'model*.hdf5'))
    base_names = [os.path.basename(check_point) for check_point in check_point_list]
    epochs = [int(re.search(r'\d+', string).group()) for string in base_names]
    return np.max(epochs) if epochs else 0


def lr_schedule_func(global_step):
    step = float(global_step + 1)
    return hparams.LEARNING_RATE * hparams.WARMUP_STEPS ** 0.5 * np.min((
        step * hparams.WARMUP_STEPS ** -1.5,
        step ** -0.5))


class LearningRateSchedulerPerGlobalStep(LearningRateScheduler):
    def __init__(self, schedule, batches_per_epoch=0, initial_epoch=0, verbose=0):
        super(LearningRateSchedulerPerGlobalStep, self).__init__(schedule, verbose)
        self.count = initial_epoch*batches_per_epoch

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        super(LearningRateSchedulerPerGlobalStep, self).on_epoch_begin(self.count, logs)

    def on_batch_end(self, batch, logs=None):
        super(LearningRateSchedulerPerGlobalStep, self).on_epoch_end(self.count, logs)
        self.count += 1
