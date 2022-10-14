import numpy as np
from numpy.linalg import inv
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

REG = 0.0001

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
    X_train, X_test, y_train, y_test = train_test_split(iris_dataset["data"], iris_dataset["target"], test_size=0.8, random_state=N)

    # print("X_train: (", len(X_train),"),", type(X_train), "\n", X_train,"\n")
    # print("X_test: (", len(X_test),"),", type(X_test), "\n", X_test,"\n")
    # print("Y_train: (", len(y_train),"),", type(y_train), "\n", y_train,"\n")
    # print("Y_test: (", len(y_test),"),", type(y_test), "\n",y_test,"\n")

    #################################################################################################################################################

    Ytr, Yts = [], []

    for elem in y_train:
        new_list = np.array([0,0,0])
        new_list[elem]  = 1
        Ytr.append(new_list)

    for elem in y_test:
        new_list = np.array([0,0,0])
        new_list[elem]  = 1
        Yts.append(new_list)

    Ytr, Yts = np.array(Ytr), np.array(Yts)

    # print("Ytr: (", len(Ytr),"),", type(Ytr), "\n", Ytr, "\n")
    # print("Yts: (", len(Yts),"),", type(Yts), "\n", Yts, "\n")
    
    #################################################################################################################################################

    w_list, Ptrain_list, Ptest_list = [], [], []

    for test_order in range(1, 11):
        poly = PolynomialFeatures(test_order)

        P_train, P_test = poly.fit_transform(X_train), poly.fit_transform(X_test)
        Ptrain_list.append(P_train)
        Ptest_list.append(P_test)

        if P_train.shape[1] >= P_train.shape[0]: # Use Dual Solution
            ppt = P_train @ P_train.T
            lamda_i = REG * np.identity(ppt.shape[0])
            
            w = P_train.T @ inv(ppt + lamda_i) @ Ytr

        else: # Use Primal Solution
            ptp = P_train.T @ P_train
            lamda_i = REG * np.identity(ptp.shape[0])
            
            w = inv(ptp + lamda_i) @ P_train.T @ Ytr

        w_list.append(w)

    # print("w_list: (", len(w_list),"),", type(w_list), "\n", w_list, "\n")       
    # print("Ptrain_list: (", len(Ptrain_list),"),", type(Ptrain_list), "\n", Ptrain_list, "\n")       
    # print("Ptest_list: (", len(Ptest_list),"),", type(Ptest_list), "\n", Ptest_list, "\n")       
    
    #################################################################################################################################################

    y_trainPred_list, y_testPred_list = [], []

    for order, p_elem in enumerate(Ptrain_list):
        y_trainPred_list.append(p_elem @ w_list[order])

    for order, p_elem in enumerate(Ptest_list):
        y_testPred_list.append(p_elem @ w_list[order])

    error_train_array, error_test_array = [], []
  
    for order in y_trainPred_list:
        errCount = 0
        for index, pred_test in enumerate(order):
            # print(list(pred_test).index(max(pred_test)), list(Ytr[index]).index(max(Ytr[index])))

            if list(pred_test).index(max(pred_test)) != list(Ytr[index]).index(max(Ytr[index])):
                errCount += 1

        error_train_array.append(errCount)

    for order in y_testPred_list:
        errCount = 0
        for index, pred_test in enumerate(order):
            # print(list(pred_test).index(max(pred_test)), list(Yts[index]).index(max(Yts[index])))

            if list(pred_test).index(max(pred_test)) != list(Yts[index]).index(max(Yts[index])):
                errCount += 1

        error_test_array.append(errCount)

    error_train_array = np.array(error_train_array).reshape((len(error_train_array),1))
    error_test_array = np.array(error_test_array).reshape((len(error_test_array),1))

    # print(type(X_train), type(y_train), type(X_test), type(y_test), type(Ytr), type(Yts), type(Ptrain_list), type(Ptest_list), type(w_list), type(error_train_array), type(error_test_array))

    # return in this order
    return X_train, y_train, X_test, y_test, Ytr, Yts, Ptrain_list, Ptest_list, w_list, error_train_array, error_test_array
