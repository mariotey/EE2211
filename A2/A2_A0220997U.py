import numpy as np
from numpy.linalg import inv
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

    print("X_train: (", len(X_train),"),", type(X_train), "\n", X_train,"\n")
    print("X_test: (", len(X_test),"),", type(X_test), "\n", X_test,"\n")
    print("Y_train: (", len(Y_train),"),", type(Y_train), "\n", Y_train,"\n")
    print("Y_test: (", len(Y_test),"),", type(Y_test), "\n",Y_test,"\n")

    #################################################################################################################################################

    Ytr, Yts = [], []

    for elem in Y_train:
        new_list = np.array([0,0,0])
        new_list[elem]  = 1
        Ytr.append(new_list)

    for elem in Y_test:
        new_list = np.array([0,0,0])
        new_list[elem]  = 1
        Yts.append(new_list)

    Ytr, Yts = np.array(Ytr), np.array(Yts)

    print("Ytr: (", len(Ytr),"),", type(Ytr), "\n", Ytr, "\n")
    print("Yts: (", len(Yts),"),", type(Yts), "\n", Yts, "\n")
    
    #################################################################################################################################################
    
    w_list, Ptrain_list, Ptest_list = [], [], []

    for order in range(1, 11):
        poly = PolynomialFeatures(order)
        # print(poly, "\n")

        P_test = poly.fit_transform(X_train)
        Ptrain_list.append(P_test)
        # print("Matrix P:" , P.shape, "\n", P, "\n")  

        if (P_test.shape[1] > P_test.shape[0]): # Use Dual Solution
            w = P_test.T @ inv(P_test @ P_test.T) @ Y_train
        else: # Use Primal Solution
            w = (inv(P_test.T @ P_test) @ P_test.T) @ Y_train

        w_list.append(w)

    w_list, Ptrain_list, Ptest_list = np.array(w_list), np.array(Ptrain_list), np.array(Ptest_list)

    print("w_list: (", len(w_list),"),", type(w_list), "\n", w_list, "\n")       
    print("Ptrain_list: (", len(Ptrain_list),"),", type(Ptrain_list), "\n", Ptrain_list, "\n")       
    print("Ptest_list: (", len(Ptest_list),"),", type(Ptest_list), "\n", Ptest_list, "\n")       
    
    # return in this order
    # return X_train, y_train, X_test, y_test, Ytr, Yts, Ptrain_list, Ptest_list, w_list, error_train_array, error_test_array
