import numpy as np
from skimage.exposure import exposure
from skimage.morphology import disk
from skimage.transform import rescale, resize
from skimage.filters import gaussian
from skimage.filters.rank import minimum, maximum
from src.MainCalculation import MainCalculation
import pickle
from src.stats import get_color_var, get_moments
from sklearn.tree import DecisionTreeClassifier
from skimage.io import imshow
import matplotlib.pyplot as plt
from numba import jit
from tqdm import tqdm


class Tree(MainCalculation):

    def __init__(self):
        with open("../learn/DTNone.pickle", "br") as f:
            self.__tree: DecisionTreeClassifier = pickle.load(f)

    def calculate(self, image: np.ndarray, mask: np.ndarray, **kwargs) -> np.ndarray:

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
                    result[i, j] = self.__tree.predict([p])
            progress.progress(5 + int(80 * ((i + 1) / (len(ma_sc) - 10))))
        # imshow(result)
        # plt.show()
        stream[1].append(result)
        stream[0].image(stream[1], width=300)
        result = gaussian(result, 2)
        # result = minimum(maximum(result, disk(6)), disk(7))
        # imshow(result)
        # plt.show()
        stream[1].append(result)
        stream[0].image(stream[1], width=300)
        # result = minimum(maximum(result, disk(5)), disk(5))
        p2, p98 = np.percentile(result, (10, 95))
        result = exposure.rescale_intensity(result, in_range=(p2, p98))
        # result = gaussian(result, 2)
        # result = self.mult(result)
        stream[1].append(result)
        stream[0].image(stream[1], width=300)
        # imshow(result)
        # plt.show()
        # result = np.where(result > 0.25, 1.0, 0.0)
        # imshow(result)
        # plt.show()
        # stream[1].append(result)
        # stream[0].image(stream[1], width=300)

        result = resize(result, or_shape, anti_aliasing=False)
        result = np.where(result > 0.4, 1.0, 0.0)
        # imshow(result)
        # plt.show()
        return result

    @staticmethod
    # @jit(nopython=True)
    def mult(image: np.ndarray):
        for i, l in enumerate(image):
            for j, v in enumerate(l):
                if v >= 0.5:
                    image[i, j] = 1.0
                else:
                    image[i, j] = 2 * v
        return image
