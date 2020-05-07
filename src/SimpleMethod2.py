import numpy as np
from src.MainCalculation import MainCalculation
from skimage.filters import laplace, meijering, sato
from skimage.io import imshow
from skimage.filters.rank import maximum, minimum
from skimage.morphology import disk
import matplotlib.pyplot as plt


class SimpleMethod2(MainCalculation):

    def calculate(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        imshow(image)
        plt.show()
        lap = maximum(minimum(image, disk(8)), disk(5))
        imshow(lap)
        plt.show()
        res = meijering(lap)
        imshow(res)
        plt.show()
        for i, l in enumerate(res):
            for j, v in enumerate(l):
                if v >= 0.15:
                    res[i, j] = 1.0
                else:
                    res[i, j] = 0.0
        imshow(res)
        plt.show()
        return res
