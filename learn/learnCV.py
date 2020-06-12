from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
import pickle
from src.stats import get_data
from tqdm import tqdm
import numpy as np

if __name__ == "__main__":
    c, v = get_data()
    print("dzielenie...")
    X_train, X_test, Y_train, Y_test = train_test_split(v, c, test_size=0.3)

    # GridSearchCV(estimator=sgd, param_grid={},
    #              scoring='f1', n_jobs=1,
    #              return_train_score=True)
    # sgd = SVC()
    cTj = np.sum(Y_train)
    cTz = len(Y_train) - cTj
    param = dict(criterion=['gini', 'entropy'], splitter=['best', 'random'], max_depth=[10, 20, 40, None],
                 min_impurity_decrease=[0.0, 0.2, 0.5], max_leaf_nodes=[2, 5, None],
                 class_weight=['balanced', {0: (cTj / len(Y_train)), 1: (cTz / len(Y_train))}, {0: 1, 1: 10}])
    grid = RandomizedSearchCV(estimator=DecisionTreeClassifier(), param_distributions=param,
                              scoring='balanced_accuracy', refit=True, verbose=100)

    print("uczenie...")
    grid.fit(X_train, Y_train)
    with open("GSCV1.pickle", 'wb') as file:
        pickle.dump(grid, file)
    print(grid.cv_results_, grid.scoring)
    print(grid.best_params_, grid.best_estimator_)

    print("testowanie...")
    res = grid.predict(X_test)

    print("wyniki...")
    cm = confusion_matrix(Y_test, res)
    accuracy = float(cm[0, 0] + cm[1, 1]) / sum(sum(cm))
    sensitivity = float(cm[0, 0]) / (cm[0, 0] + cm[0, 1])
    specificity = float(cm[1, 1]) / (cm[1, 0] + cm[1, 1])
    print(accuracy, sensitivity, specificity, (sensitivity + specificity) / 2)
