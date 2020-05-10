import numpy as np
from sklearn.metrics import confusion_matrix
from skimage.measure import moments_central, moments_hu
from typing import Tuple
import marshal


def statistics(image: np.ndarray, origin: np.ndarray) -> (float, float, float):
    cm = confusion_matrix(origin.reshape((origin.shape[0] * origin.shape[1], 1)),
                          image.reshape((image.shape[0] * image.shape[1], 1)))
    accuracy = float(cm[0, 0] + cm[1, 1]) / sum(sum(cm))
    sensitivity = float(cm[0, 0]) / (cm[0, 0] + cm[0, 1])  # odwrotnie
    specificity = float(cm[1, 1]) / (cm[1, 0] + cm[1, 1])  # odwrotnie
    return accuracy, sensitivity, specificity, (sensitivity + specificity) / 2


def get_moments(image: np.ndarray, position: Tuple[int, int]):
    x, y = position[0], position[1]
    d = 3
    part = image[x - d:x + d + 1, y - d:y + d + 1]
    h = moments_hu(part)
    # print(h)
    c = moments_central(part, order=4)
    # print(c)
    return *(c.reshape((25, 1))), *h


def get_color_var(image: np.ndarray, position: Tuple[int, int]) -> (float, float, float):
    x, y = position[0], position[1]
    d = 3
    part = image[x - d:x + d + 1, y - d:y + d + 1, :]
    r, g, b = part[:, :, 0], part[:, :, 1], part[:, :, 2]
    vr, vg, vb = np.var(r), np.var(g), np.var(b)
    return vr, vg, vb


def get_data():
    print("wczytywanie...")
    with open("c.pickle", "br") as f:
        c = marshal.load(f)
    with open("v.pickle", "br") as f:
        v = marshal.load(f)
    return c, v
