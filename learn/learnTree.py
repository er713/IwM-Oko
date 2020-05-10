from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import marshal
import pickle
import numpy as np
from numba import jit
from typing import Iterable, Union
from src.stats import get_data
from tqdm import tqdm


def make_decision_tree(n, X_train, X_test, Y_train, Y_test):
    dt = tree.DecisionTreeClassifier(max_depth=n, criterion='entropy')

    print("uczenie...")
    dt.fit(X_train, Y_train)

    print("przewidywanie...")
    Y_pred = dt.predict(X_test)

    print("wyniki...")
    cm = confusion_matrix(Y_test, Y_pred)
    accuracy = float(cm[0, 0] + cm[1, 1]) / sum(sum(cm))
    sensitivity = float(cm[0, 0]) / (cm[0, 0] + cm[0, 1])
    specificity = float(cm[1, 1]) / (cm[1, 0] + cm[1, 1])
    print(n, dt.get_depth(), accuracy, sensitivity, specificity)

    with open("v4DT" + str(n) + ".pickle", "bw") as f:
        pickle.dump(dt, f)


# @jit(nopython=True)
def add_one(c, v, c0, v0):
    for i, a in tqdm(enumerate(c)):
        if a == 1:
            c0.append(c[i])
            v0.append(v[i])
    return c0, v0


if __name__ == "__main__":
    c, v = get_data()
    print("dzielenie...")
    # i1 = np.sum(c)
    # v0 = []
    # c0 = []
    # for i in range(len(c)):
    #     if c[i] == 0:
    #         v0.append(v[i])
    #         c0.append(c[i])
    #         if i1 * 18 <= len(v0):
    #             break
    # c0, v0 = add_one(c, v, c0, v0)
    X_train, X_test, Y_train, Y_test = train_test_split(v, c, test_size=0.3)

    make_decision_tree(None, X_train, X_test, Y_train, Y_test)
    # make_decision_tree(100, X_train, X_test, Y_train, Y_test)
    # make_decision_tree(70, X_train, X_test, Y_train, Y_test)
    make_decision_tree(55, X_train, X_test, Y_train, Y_test)
    make_decision_tree(40, X_train, X_test, Y_train, Y_test)
    make_decision_tree(35, X_train, X_test, Y_train, Y_test)
    make_decision_tree(20, X_train, X_test, Y_train, Y_test)
    make_decision_tree(15, X_train, X_test, Y_train, Y_test)
    make_decision_tree(5, X_train, X_test, Y_train, Y_test)
