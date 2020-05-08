import numpy as np
from src.MainCalculation import MainCalculation
from sklearn.neighbors import KNeighborsClassifier
from src.stats import *
import pickle
from skimage.io import imshow
import matplotlib.pyplot as plt
from tqdm import tqdm


class KNN(MainCalculation):

    def __init__(self):
        with open("../learn/KNN9.pickle", "br") as f:
            self.knn: KNeighborsClassifier = pickle.load(f)
        # print("Typ:", type(self.knn))
        # print(self.knn)
        # print(self.knn.predict(np.zeros((1, 35))))

    def calculate(self, image: np.ndarray, mask: np.ndarray, **kwargs) -> np.ndarray:
        result = np.zeros(image.shape, dtype=np.float)
        org = kwargs["origin"]
        le = len(mask[0]) - 5
        for i in tqdm(range(5, len(mask) - 5)):
            for j in range(5, le):
                if not mask[i, j]:
                    result[i, j] = self.knn.predict([[*get_color_var(org, (i, j)), *get_moments(image, (i, j))]])
        imshow(result, cmap='gray')
        plt.show()

        return result
