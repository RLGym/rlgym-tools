from abc import ABC


class RunningNormalizer(ABC):
    def update(self, values):
        raise NotImplementedError

    def normalize(self, values):
        raise NotImplementedError
