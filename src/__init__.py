from src.ProcessImage import ProcessImage
from src.AlgorithmType import AlgorithmType
from skimage.io import imread, imsave
import numpy as np
import os
import re


def get_image(path: str, name: str) -> str:
    for s in os.listdir(path):
        if re.search("^" + name + ".[a-zA-Z]+$", s) is not None:
            return path + s


if __name__ == "__main__":
    path, file, pathb = "../../picture/", "62", "../../masters/"
    process = ProcessImage(AlgorithmType.SIMPLE)
    image = imread(get_image(path, "color" + file))
    mask = process.get_mask(image)
    im = process.preprocesing(image)
    im = process.process(im, mask)
    # print(im.shape, im)

    # imsave("test.jpg", im)
    # process.get_moments(im, (100, 100))

    # tru = imread(get_image(pathb, "mst"+file))
    # if np.max(tru) > 1.0:
    #     tru = tru/255.0
    # print(process.statistics(im, tru))
