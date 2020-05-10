import pickle as pk

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from learn.prepareFiles import *


def learnKNN(X_train, X_test, Y_train, Y_test, n):
    knn = KNeighborsClassifier(n_neighbors=n)

    print("uczenie...")
    knn.fit(X_train, Y_train)

    print("przewidywanie...")
    Y_pred = knn.predict(X_test)

    print("wyniki...")
    cm = confusion_matrix(Y_test, Y_pred)
    print(cm)
    accuracy = float(cm[0, 0] + cm[1, 1]) / sum(sum(cm))
    sensitivity = float(cm[0, 0]) / (cm[0, 0] + cm[0, 1])
    specificity = float(cm[1, 1]) / (cm[1, 0] + cm[1, 1])
    print(n, accuracy, sensitivity, specificity)

    with open("v2KNN" + str(n) + ".pickle", "bw") as f:
        pk.dump(knn, f)


if __name__ == "__main__":
    print("wczytywanie...")
    c, v = read_data()

    print("dzielenie...")
    # v = [v[:, i] for i in (0, 1, 2, 3, 4, 5, 6, 7, 8, -7, -6, -5, -4, -3, -2, -1)]
    # print(v)
    X_train, X_test, Y_train, Y_test = train_test_split(v, c, test_size=0.25)
    X_train, X_test, Y_train, Y_test = train_test_split(X_test, Y_test, test_size=0.3)

    learnKNN(X_train, X_test, Y_train, Y_test, 1)
    learnKNN(X_train, X_test, Y_train, Y_test, 2)
    learnKNN(X_train, X_test, Y_train, Y_test, 3)
    learnKNN(X_train, X_test, Y_train, Y_test, 4)
    learnKNN(X_train, X_test, Y_train, Y_test, 5)
    learnKNN(X_train, X_test, Y_train, Y_test, 6)
    learnKNN(X_train, X_test, Y_train, Y_test, 7)
    learnKNN(X_train, X_test, Y_train, Y_test, 8)
    learnKNN(X_train, X_test, Y_train, Y_test, 9)
    learnKNN(X_train, X_test, Y_train, Y_test, 10)
    learnKNN(X_train, X_test, Y_train, Y_test, 12)
    learnKNN(X_train, X_test, Y_train, Y_test, 15)
    learnKNN(X_train, X_test, Y_train, Y_test, 20)
