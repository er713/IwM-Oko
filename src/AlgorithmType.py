from enum import Enum

from src.SimpleMethod import SimpleMethod


class AlgorithmType(Enum):
    SIMPLE = 1
    KNN = 2


__switch = {
    AlgorithmType.SIMPLE: lambda: SimpleMethod(),
    AlgorithmType.KNN: lambda: print("brak")
}


def constructor(algorithm: AlgorithmType):
    return __switch.get(algorithm)()
