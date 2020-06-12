from enum import Enum

from SimpleMethod import SimpleMethod
from SimpleMethod2 import SimpleMethod2
from Tree import Tree
from KNN import KNN
from Tree2 import Tree2


class AlgorithmType(Enum):
    NONE = 0
    SIMPLE = 1
    KNN = 2
    SIMPLE2 = 3
    TREE = 4
    TREE2 = 5


def get_none():
    return None


__switch = {
    AlgorithmType.NONE: lambda: get_none(),
    AlgorithmType.SIMPLE: lambda: SimpleMethod(),
    AlgorithmType.KNN: lambda: KNN(),
    AlgorithmType.SIMPLE2: lambda: SimpleMethod2(),
    AlgorithmType.TREE: lambda: Tree(),
    AlgorithmType.TREE2: lambda: Tree2()
}


def constructor(algorithm: AlgorithmType):
    return __switch.get(algorithm)()
