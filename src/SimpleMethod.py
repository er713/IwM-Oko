from MainCalculation import MainCalculation
import numpy as np
from skimage.filters.rank import otsu, minimum, maximum
from skimage.morphology import disk
from skimage.io import imshow
from skimage.filters import gaussian, threshold_multiotsu, threshold_sauvola
from skimage.filters.thresholding import _cross_entropy
import matplotlib.pyplot as plt
from typing import Union, Iterable
from numba import jit


class SimpleMethod(MainCalculation):
    """
    dużo zakłóceń
    """

    def calculate(self, image: np.ndarray, mask: np.ndarray, **kwargs) -> np.ndarray:
        # print(image*255 + 20)
        # imshow(image)
        # plt.show()
        stream = kwargs["stream"]
        progress = kwargs["progress"]
        # return self.__otsu_method(image)
        return self.__classify(self.__sauvola_method(image, stream, progress), mask)

    @staticmethod
    @jit(nopython=True)
    def __classify(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        th: float = 60.0 / 255.0
        for i, l in enumerate(image):
            for j, v in enumerate(l):
                if mask[i, j]:
                    image[i, j] = 0.0
                else:
                    if v > th:
                        image[i, j] = 1.0
                    else:
                        image[i, j] = 0.0
        return image

    @staticmethod
    def __sauvola_method(image: Union[np.ndarray, Iterable, np.uint8], stream, progres) -> np.ndarray:
        th = threshold_sauvola(image, window_size=45)
        im = image <= th
        # imshow(im, cmap='gray')
        # plt.show()
        progres.progress(33)
        # stream[1].append(im)
        # stream[0].image(stream[1], width=300)
        # if im.shape[0] > 1000:
        #     gaus = 7
        # elif im.shape[0] > 650:
        #     gaus = 3
        # else:
        #     gaus = 1
        if im.shape[0] > 400:
            gaus = int(0.00330957*im.shape[0] - 0.13687352)
        else:
            gaus = 1
        result = gaussian(im, gaus)
        # imshow(result)
        # plt.show()
        progres.progress(66)
        stream[1].append(result)
        stream[0].image(stream[1], width=300)
        # result = minimum(maximum(result, disk(5)), disk(12))
        # imshow(result)
        # plt.show()
        return result

    @staticmethod
    def __otsu_method(image: Union[np.ndarray, Iterable, np.uint8]) -> np.ndarray:
        selem = disk(20)
        t_otsu = otsu(image, selem=selem)
        print(t_otsu)
        imshow(t_otsu)
        # th_motsu = threshold_multiotsu(image, classes=2)
        # im = np.digitize(image, bins=th_motsu)
        # imshow(im)
        plt.show()
        test = (image * 255 + 15) <= t_otsu
        result = np.zeros(image.shape, dtype=np.uint8)
        for i, l in enumerate(test):
            for j, v in enumerate(l):
                if v:
                    result[i, j] = 255
        imshow(result)
        plt.show()
        # result = gaussian(gaussian(result, 7), 7)
        result = gaussian(result, 7)
        imshow(result)
        plt.show()
        result = minimum(maximum(result, disk(5)), disk(12))
        imshow(result)
        plt.show()
        result = gaussian(result, 3)
        imshow(result)
        plt.show()

        # return self.__ending(gaussian(self.__ending(result), 7))
        # return self.__ending(result)
        return result

    @staticmethod
    @jit(nopython=True)
    def __ending(image: Union[np.ndarray, Iterable, np.uint8]) -> np.ndarray:
        # print(image)
        thr: float = 70.0 / 255.0
        for i, l in enumerate(image):
            for j, v in enumerate(l):
                if v > thr:
                    image[i, j] = 1.0
                else:
                    image[i, j] = 0.0
        return image
