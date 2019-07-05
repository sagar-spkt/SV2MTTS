import numpy as np
from tensorflow.python.keras.callbacks import LearningRateScheduler

import hparams


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
