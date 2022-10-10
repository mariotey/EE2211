import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

# Please replace "MatricNumber" with your actual matric number here and in the filename
def A2_A0220997U(N):
    """
    Input type
    :N type: int

    Return type
    :X_train type: numpy.ndarray of size (number_of_training_samples, 4)
    :y_train type: numpy.ndarray of size (number_of_training_samples,)
    :X_test type: numpy.ndarray of size (number_of_test_samples, 4)
    :y_test type: numpy.ndarray of size (number_of_test_samples,)
    :Ytr type: numpy.ndarray of size (number_of_training_samples, 3)
    :Yts type: numpy.ndarray of size (number_of_test_samples, 3)
    :Ptrain_list type: List[numpy.ndarray]
    :Ptest_list type: List[numpy.ndarray]
    :w_list type: List[numpy.ndarray]
    :error_train_array type: numpy.ndarray
    :error_test_array type: numpy.ndarray
    """
    # your code goes here

    iris_dataset = load_iris()

    # Create dateframe from data in X_train
    X_train, X_test, Y_train, Y_test = train_test_split(iris_dataset["data"], iris_dataset["target"], test_size=0.8, random_state=N)


    # return in this order
    return X_train, y_train, X_test, y_test, Ytr, Yts, Ptrain_list, Ptest_list, w_list, error_train_array, error_test_array
