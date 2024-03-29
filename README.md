# EE2211: Introduction to Machine Learning #

## Description ##
This module introduces students to various machine learning concepts and applications, and the tools needed to understand them. Topics include supervised and unsupervised machine learning techniques, optimization, overfitting, regularization, cross-validation and evaluation metrics. The mathematical tools include basic topics in probability and statistics, linear algebra, and optimization. These concepts will be illustrated through various machine-learning techniques and examples. (Description from https://nusmods.com/modules/EE2211/introduction-to-machine-learning)

This repository contains the assignments that the author has done previously for this module.
- A1: Matrix Calculation
- A2: Training and Testing of Model with Regularization
- A3: Gradient Descent to Minimize Cost Function

## Getting Started ##
### Dependencies ###
- numpy
- sklearn

### Installing ###
- Clone Project from git repository

### Executing Program ###
- After all files have been cloned, simply run "test_file.py"
- To run AX, kindly make the following amendments to "test_file.py"
  - Modify line 1: 
    ```python
    import AX.AX_A0220997U as grading
    ```
  - For A1, modify line 6. Kindly input matrix input for X and y: 
    ```python
    InvXTX, w = grading.A1_A0220997U(X,y)
    ```
  - For A2, modify line 6: 
    ```python
    X_train, y_train, X_test, y_test, Ytr, Yts, Ptrain_list, Ptest_list, w_list, error_train_array, error_test_array = grading.A2_A0220997U(5)
    ```
  - For A3, modify line 6: 
    ```python
    a_out, f1_out, b_out, f2_out, c_out, d_out, f3_out = grading.A3_A0220997U(learning_rate, num_iters)
    ```

## Author ##
Tey Ming Chuan
