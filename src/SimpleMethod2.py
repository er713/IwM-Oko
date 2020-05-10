import numpy as np
from MainCalculation import MainCalculation
from skimage.filters import meijering
from skimage.filters.rank import maximum, minimum
from skimage.morphology import disk


class SimpleMethod2(MainCalculation):

    def calculate(self, image: np.ndarray, mask: np.ndarray, **kwargs) -> np.ndarray:
        # imshow(image)
        # plt.show()
        stream = kwargs["stream"]
        progress = kwargs["progress"]
        lap = maximum(minimum(image, disk(8)), disk(5))
        progress.progress(30)
        # imshow(lap)
        # plt.show()
        stream[1].append(lap)
        stream[0].image(stream[1], width=300)
        res = meijering(lap)
        progress.progress(60)
        # imshow(res)
        # plt.show()
        stream[1].append(res)
        stream[0].image(stream[1], width=300)
        for i, l in enumerate(res):
            for j, v in enumerate(l):
                if v >= 0.15:
                    res[i, j] = 1.0
                else:
                    res[i, j] = 0.0
            progress.progress(60 + int(40 * ((i + 1) / len(res))))
        # imshow(res)
        # plt.show()
        # stream[1].append(res)
        # stream[0].image(stream[1], width=300)
        return res
