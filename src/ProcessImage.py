from src.MainCalculation import MainCalculation
from src.AlgorithmType import AlgorithmType, constructor
from src.SimpleMethod import SimpleMethod


class ProcessImage:

    _mainCalculation: MainCalculation

    def __init__(self, algorithm: AlgorithmType):
        self._mainCalculation = constructor(algorithm)

    def process(self):
        self._mainCalculation.calculate()
