import numpy as np
from learn.prepareFiles import read_data  # , prepare_files
import marshal
import marshal

import numpy as np

from learn.prepareFiles import read_data  # , prepare_files

if __name__ == "__main__":
    # prepare_files("../../picture/", "../../masters/")
    # c, v = read_data()
    # print(len(c), len(v))
    # with open("c.pickle", "bw") as f:
    #     marshal.dump(c, f)
    # with open("v.pickle", "bw") as f:
    #     marshal.dump(v, f)
    # proc = ProcessImage(AlgorithmType.KNN)

    with open("c.pickle", "br") as f:
        c = marshal.load(f)
    with open("v.pickle", "br") as f:
        v = marshal.load(f)

    s = np.sum(c)
    print(s, s/len(c)*100)

    # image = imread("../src/test.jpg")
    # imshow(image)
    # plt.show()
    # image = gaussian(image, 2)
    # imshow(image)
    # plt.show()
    # image = np.where(image < 0.08, 0.0, 1.0)
    # # image = minimum(maximum(image, disk(5)), disk(5))
    # imshow(image)
    # plt.show()
    # tru = imread("../../masters/mst65.ppm")
    # if np.max(tru) > 1.0:
    #     tru = tru / 255.0
    # print(statistics(image, tru))
