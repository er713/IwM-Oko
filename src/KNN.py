import numpy as np
from src.ProcessImage import ProcessImage
from src.MainCalculation import MainCalculation
import pickle


class KNN(MainCalculation):

    def __init__(self):
        with open("../learn/KNN1.pickle", "br") as f:
            self.knn = pickle.load(f)
        print(type(self.knn))

    def calculate(self, image: np.ndarray, mask: np.ndarray, **kwargs) -> np.ndarray:
        result = np.zeros(image.shape)
