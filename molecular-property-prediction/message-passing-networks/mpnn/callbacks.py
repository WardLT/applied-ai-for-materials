"""Custom callbacks for Keras model training"""
from time import perf_counter

from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K


class LRLogger(Callback):
    """Add the LR to the logs
    Must be before any log writers in the callback list"""

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            logs['lr'] = float(K.get_value(self.model.optimizer.lr))


class EpochTimeLogger(Callback):
    """Adds the epoch time to the logs
    Must be before any log writers in the callback list"""

    def __init__(self):
        super().__init__()
        self.time = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.time = perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            logs['epoch_time'] = perf_counter() - self.time
