from typing import Tuple, List

import numpy as np
import streamlit as st
from AlgorithmType import AlgorithmType, constructor
from MainCalculation import MainCalculation
from skimage import exposure
from skimage.color import rgb2gray


class ProcessImage:
    _mainCalculation: MainCalculation

    def __init__(self, algorithm: AlgorithmType):
        self._mainCalculation = constructor(algorithm)

    def process(self, image: np.ndarray, mask: np.ndarray, **kwargs) -> np.ndarray:
        # print(image)
        return self._mainCalculation.calculate(image, mask, **kwargs)

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
    # @jit(nopython=True)
    def color_change(image: np.ndarray, stream: Tuple[st.empty, List[np.ndarray]], mask: np.ndarray,
                     progress: st.progress) -> np.ndarray:
        """
        Zmiana temperatury obrazu na około 3000K (przynajmniej według programu 'darktable',
        który przy zmniejszaniu temperatury sprawia, że obraz jest bardziej niebieski, co zdaje się, że nie
        jest do końca prawidłowe). W ostateczności sprowadza się to do zmniejszenia wartości odpowiadającemu
        kolorowi czerwonemu o połowę i zwiększeniu wartości dla niebieskiego prawie 3 razy.
        :param progress:
        :param mask:
        :param stream:
        :param image: przetwarzany obraz
        :return: obraz ze zmienioną temperaturą
        """

        # pil_image = Image.fromarray(np.uint8(image))
        # # 155 188 255
        # color_matrix = (155.0 / 255.0, 0.0, 0.0, 0.0, 0.0, 188.0 / 255.0, 0.0, 0.0, 0.0, 0.0, 255.0 / 255.0, 0.0)
        # pil_image = pil_image.convert('RGB', color_matrix)
        # pil_image.save("test.jpeg", "JPEG")

        res = np.zeros(image.shape, dtype=np.uint8)
        mr, mb = 0.5, 2.991  # 0.5
        for i, l in enumerate(image):
            for j, v in enumerate(l):
                r, g, b = v
                if not mask[i, j]:
                    tr = mr * r
                    tb = b * mb
                    if tb > 255:
                        tb = 255
                    res[i, j] = (np.uint8(tr), g, np.uint8(tb))
            progress.progress((i + 1) / len(image))
        stream[1].append(res)
        stream[0].image(stream[1], width=300)  # , use_column_width=True)
        return res

    @staticmethod
    def to_grayscale(image: np.ndarray, stream: Tuple[st.empty, List]) -> np.ndarray:
        res = rgb2gray(image)
        stream[1].append(res)
        stream[0].image(stream[1], width=300)  # , use_column_width=True)
        return res

    @staticmethod
    def get_mask(image: np.ndarray) -> np.ndarray:
        """

        :param image: kolorowy obraz
        :return: tablica booli, True - piksel należy do "ramki"
        """

        # print(image)
        if type(image[0, 0, 0]) == np.uint8:
            # print("do 255")
            return image[:, :, 0] < 30
        return image[:, :, 0] < 0.12

    @staticmethod
    def preprocesing(image: np.ndarray, stream: Tuple[st.empty, List], mask: np.ndarray,
                     progress: st.progress) -> np.ndarray:
        return ProcessImage.contrast_change(
            ProcessImage.to_grayscale(ProcessImage.color_change(image, stream, mask, progress), stream))
