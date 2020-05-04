from abc import ABC, abstractmethod


class MainCalculation(ABC):

    @abstractmethod
    def calculate(self):
        pass
