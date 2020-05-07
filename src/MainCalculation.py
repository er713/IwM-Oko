from abc import ABC, abstractmethod
import numpy as np


class MainCalculation(ABC):

    @abstractmethod
    def calculate(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        pass
