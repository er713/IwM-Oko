from src.ProcessImage import ProcessImage
import os
import re
import numpy as np
from src.AlgorithmType import AlgorithmType
from skimage.io import imread, imsave
from tqdm import tqdm
from numba import jit


# @jit()
def read_data():
    cls = []
    vals = []
    with open("data.csv", "r") as f:
        while f.readable():
            s = f.readline()
            if len(s) == 0 or s == "":
                break
            if re.search("nan", s) is not None:
                continue
            v = s.split(", ")
            if v[0] != '' and len(v[0]) != 0:
                cl = int(v[0])
                val = [float(k) for k in v[1:]]
                cls.append(cl)
                vals.append(val)
    return cls, vals


def printProgressBar(iteration, total, prefix='Progress: ', suffix='Complete', decimals=1, length=100, fill='█',
                     printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end=printEnd)
    # Print New Line on Complete
    # if iteration == total:
    print()


def get_image(path: str, name: str) -> str:
    for s in os.listdir(path):
        if re.search("^" + name + ".[a-zA-Z]+$", s) is not None:
            return path + s


def prepare_files(path: str, real: str):
    f = open("data.csv", "w")
    # all = len(os.listdir(path))
    # printProgressBar(0, all)
    for file in tqdm(os.listdir(path), ncols=75):
        image = imread(path + file)

        num = re.search("[0-9]+(?=\\.)", file)
        if num is None:
            print("PROBLEM")
        print("mst" + num.string[num.start():num.end()])
        reality = imread(get_image(real, "mst" + num.string[num.start():num.end()]))
        if not type(reality[0, 0]) == float:
            reality = np.uint8(reality / 255)

        mask = ProcessImage.get_mask(image)

        gray = ProcessImage.preprocesing(image)
        imsave("../../gray/" + re.sub("color", "gray", file), gray)

        i = 0
        # pix = image.shape[0] * image.shape[1] * 0.2 // 46.5
        pix = 10870
        # print(pix)
        while i < pix:
            x, y = int(np.random.random() * (image.shape[0] - 10) + 5), int(
                np.random.random() * (image.shape[1] - 10) + 5)
            # print(x, y)
            # print(mask[x, y])
            if not mask[x][y]:
                vr, vg, vb = ProcessImage.get_color_var(image, (x, y))
                moments = ProcessImage.get_moments(gray, (x, y))
                rel = reality[x, y]
                f.write(str(rel) + ", " + str(vr) + ", " + str(vg) + ", " + str(vb))
                for m in moments:
                    f.write(", " + str(float(m)))
                f.write("\n")
                i += 1
                # printProgressBar(i, pix)

        # print(path + re.sub("color", "gray", file))

        # printProgressBar(i + 1, all)
    f.close()
