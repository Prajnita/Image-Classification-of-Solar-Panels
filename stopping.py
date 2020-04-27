import math

class EarlyStoppingCallback:

    def __init__(self, patience):
        # initialize all members you need
        self._patience = patience
        self._previous_values = []

    def step(self, current_loss):
        # check whether the current loss is lower than the previous best value.
        if len(self._previous_values) == 0 or self._previous_values[-1]>current_loss:
            self._previous_values.append(current_loss)
            return False
        else:
            self.counter = 0
            for v in reversed(self._previous_values):
                if v < current_loss:
                    self._previous_values.append(current_loss)
                    return self.should_stop()
                else:
                    self.counter += 1
        # if not count up for how long there was no progress

    def should_stop(self):
        # check whether the duration of where there was no progress is larger or equal to the patience
        return self.counter > self._patience

