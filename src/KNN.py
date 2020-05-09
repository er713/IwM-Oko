import numpy as np
from MainCalculation import MainCalculation
from skimage.transform import rescale
from sklearn.neighbors import KNeighborsClassifier
from stats import *
import pickle
from skimage.io import imshow
import matplotlib.pyplot as plt
from tqdm import tqdm


class KNN(MainCalculation):

    def __init__(self):
        with open("../learn/v2KNN1.pickle", "br") as f:
            self.knn: KNeighborsClassifier = pickle.load(f)
        # print("Typ:", type(self.knn))
        # print(self.knn)
        # print(self.knn.predict(np.zeros((1, 35))))

    def calculate(self, image: np.ndarray, mask: np.ndarray, **kwargs) -> np.ndarray:
        result = np.zeros(image.shape, dtype=np.float)
        stream = kwargs["stream"]
        progress = kwargs["progress"]
        # imshow(image)
        # plt.show()
        org = kwargs["origin"]
        or_shape = image.shape
        part = 500 / or_shape[1]
        im_gr = rescale(image, part, anti_aliasing=False)
        r, g, b = rescale(org[:, :, 0], part, anti_aliasing=False), rescale(org[:, :, 1], part, anti_aliasing=False), \
                  rescale(org[:, :, 2], part, anti_aliasing=False)
        im_col = np.zeros((r.shape[0], r.shape[1], 3))
        progress.progress(2)
        for i in range(len(r)):
            for j in range(len(r[i])):
                im_col[i, j, 0] = r[i, j]
                im_col[i, j, 1] = g[i, j]
                im_col[i, j, 2] = b[i, j]
        # im_col = rescale(org, part, anti_aliasing=False)
        ma_sc = rescale(mask, part, anti_aliasing=False)
        ma_sc = ma_sc > 0.5
        progress.progress(5)
        result = np.zeros(im_gr.shape)
        le = len(ma_sc[0]) - 5
        for i in tqdm(range(5, len(ma_sc) - 5)):
            for j in range(5, le):
                if not ma_sc[i, j]:
                    p = [*[float(i) for i in get_color_var(im_col, (i, j))],
                         *[float(i) for i in get_moments(im_gr, (i, j))]]
                    result[i, j] = self.knn.predict([p])
            progress.progress(5 + int(80 * ((i + 1) / (len(ma_sc) - 10))))
        # imshow(result, cmap='gray')
        # plt.show()

        return result
