from src.MainCalculation import MainCalculation
from src.AlgorithmType import AlgorithmType, constructor
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
from numba import jit
import numpy as np
from skimage import exposure
from skimage.morphology import disk
from skimage.filters import rank, unsharp_mask, threshold_otsu
from skimage.filters.rank import otsu
from skimage.color import rgb2gray
from sklearn.metrics import accuracy_score, confusion_matrix
from skimage.measure import moments_hu, moments_central
from typing import Tuple


class ProcessImage:
    _mainCalculation: MainCalculation

    def __init__(self, algorithm: AlgorithmType):
        self._mainCalculation = constructor(algorithm)

    def process(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        # print(image)
        return self._mainCalculation.calculate(image, mask)

    @staticmethod
    # @jit(nopython=True)
    def contrast_change(image: np.ndarray) -> np.ndarray:
        """
        Zwiększenie kontrastu o stałą wartość 0.25
        :param image: obraz, w którym zwiększany jest kontrast
        :return: obraz, ze zwiększonym kontrastem
        """

        # im = Image.fromarray(np.uint8(image))
        # enhance = ImageEnhance.Contrast(im)
        # result = enhance.enhance(2)
        # return np.array(result.getdata(), dtype=np.uint8).reshape((result.size[1], result.size[0], 3))

        # a = 1.25
        # b = int(0.06*255)
        # for i, l in enumerate(image):
        #     for j, v in enumerate(l):
        #         for k, c in enumerate(v):
        #             t = a*c+b
        #             if t > 255:
        #                 t = 255
        #             image[i, j, k] = np.uint8(t)
        # return image
        # selem = disk(30)
        im = exposure.equalize_adapthist(image, clip_limit=0.016)
        p2, p98 = np.percentile(im, (15, 98))
        return exposure.rescale_intensity(im, in_range=(p2, p98))

    @staticmethod
    @jit(nopython=True)
    def color_change(image: np.ndarray) -> np.ndarray:
        """
        Zmiana temperatury obrazu na około 3000K (przynajmniej według programu 'darktable',
        który przy zmniejszaniu temperatury sprawia, że obraz jest bardziej niebieski, co zdaje się, że nie
        jest do końca prawidłowe). W ostateczności sprowadza się to do zmniejszenia wartości odpowiadającemu
        kolorowi czerwonemu o połowę i zwiększeniu wartości dla niebieskiego prawie 3 razy.
        :param image: przetwarzany obraz
        :return: obraz ze zmienioną temperaturą
        """

        # pil_image = Image.fromarray(np.uint8(image))
        # # 155 188 255
        # color_matrix = (155.0 / 255.0, 0.0, 0.0, 0.0, 0.0, 188.0 / 255.0, 0.0, 0.0, 0.0, 0.0, 255.0 / 255.0, 0.0)
        # pil_image = pil_image.convert('RGB', color_matrix)
        # pil_image.save("test.jpeg", "JPEG")

        res = np.zeros(image.shape, dtype=np.uint8)
        mr, mb = 0.5, 2.991
        for i, l in enumerate(image):
            for j, v in enumerate(l):
                r, g, b = v
                tr = mr * r
                tb = b * mb
                if tb > 255:
                    tb = 255
                res[i, j] = (np.uint8(tr), g, np.uint8(tb))
        return res

    @staticmethod
    def to_grayscale(image: np.ndarray) -> np.ndarray:
        return rgb2gray(image)

    @staticmethod
    def statistics(image: np.ndarray, origin: np.ndarray) -> (float, float, float):
        cm = confusion_matrix(origin.reshape((origin.shape[0] * origin.shape[1], 1)),
                              image.reshape((image.shape[0] * image.shape[1], 1)))
        accuracy = float(cm[0, 0] + cm[1, 1]) / sum(sum(cm))
        sensitivity = float(cm[0, 0]) / (cm[0, 0] + cm[0, 1])
        specificity = float(cm[1, 1]) / (cm[1, 0] + cm[1, 1])
        return accuracy, sensitivity, specificity

    @staticmethod
    def get_mask(image: np.ndarray) -> np.ndarray:
        # print(image)
        if type(image[0, 0, 0]) == np.uint8:
            # print("do 255")
            return image[:, :, 0] < 30
        return image[:, :, 0] < 0.12

    @staticmethod
    def preprocesing(image: np.ndarray) -> np.ndarray:
        return ProcessImage.contrast_change(ProcessImage.to_grayscale(ProcessImage.color_change(image)))

    @staticmethod
    def get_moments(image: np.ndarray, position: Tuple[int, int]):
        x, y = position[0], position[1]
        d = 5
        part = image[x - d:x + d + 1, y - d:y + d + 1]
        h = moments_hu(part)
        # print(h)
        c = moments_central(part, order=4)
        # print(c)
        return *(c.reshape((25, 1))), *h

    @staticmethod
    def get_color_var(image: np.ndarray, position: Tuple[int, int]) -> (float, float, float):
        x, y = position[0], position[1]
        d = 5
        part = image[x - d:x + d + 1, y - d:y + d + 1, :]
        r, g, b = part[:, :, 0], part[:, :, 1], part[:, :, 2]
        vr, vg, vb = np.var(r), np.var(g), np.var(b)
        return vr, vg, vb
