from enum import Enum

from src.SimpleMethod import SimpleMethod
from src.SimpleMethod2 import SimpleMethod2
from src.Tree import Tree
from src.KNN import KNN


class AlgorithmType(Enum):
    NONE = 0
    SIMPLE = 1
    KNN = 2
    SIMPLE2 = 3
    TREE = 4


def get_none():
    return None


__switch = {
    AlgorithmType.NONE: lambda: get_none(),
    AlgorithmType.SIMPLE: lambda: SimpleMethod(),
    AlgorithmType.KNN: lambda: KNN(),
    AlgorithmType.SIMPLE2: lambda: SimpleMethod2(),
    AlgorithmType.TREE: lambda: Tree()
}


def constructor(algorithm: AlgorithmType):
    return __switch.get(algorithm)()
