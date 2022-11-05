import numpy as np

# Please replace "StudentMatriculationNumber" with your actual matric number here and in the filename
def A3_A0220997U(learning_rate, num_iters):
    """
    Input type
    :learning_rate type: float
    :num_iters type: int

    Return type
    :a_out type: numpy array of length num_iters
    :f1_out type: numpy array of length num_iters
    :b_out type: numpy array of length num_iters
    :f2_out type: numpy array of length num_iters
    :c_out type: numpy array of length num_iters
    :d_out type: numpy array of length num_iters
    :f3_out type: numpy array of length num_iters
    """
    # your code goes here
    a_0, b_0, c_0, d_0 = 1.5, 0.3, 1, 2

    a_out,f1_out = np.zeros(num_iters), np.zeros(num_iters)
    b_out,f2_out = np.zeros(num_iters), np.zeros(num_iters)
    c_out, d_out,f3_out = np.zeros(num_iters), np.zeros(num_iters), np.zeros(num_iters)

    for i in range(0,num_iters):
        if i == 0:
            a_out[i] = a_0-learning_rate*(4*np.power(a_0,3))
            b_out[i] = b_0-learning_rate*(np.sin(2*b_0))
            c_out[i] = c_0-learning_rate*(2*c_0)
            d_out[i] = d_0-learning_rate*(2*d_0*np.sin(d_0) + np.power(d_0,2)*np.cos(d_0))

        else:
            a_out[i] = a_out[i-1]-learning_rate*(4*np.power(a_out[i-1],3))
            b_out[i] = b_out[i-1]-learning_rate*(np.sin(2*b_out[i-1]))
            c_out[i] = c_out[i-1]-learning_rate*(2*c_out[i-1])
            d_out[i] = d_out[i-1]-learning_rate*(2*d_out[i-1]*np.sin(d_out[i-1]) + np.power(d_out[i-1],2)*np.cos(d_out[i-1]))
            
        f1_out[i] = np.power(a_out[i], 4)
        f2_out[i] = np.power(np.sin(b_out[i]), 2)
        f3_out[i] = np.power(c_out[i], 2) + np.power(d_out[i], 2)*np.sin(d_out[i])

    print(type(a_out), type(f1_out), type(b_out), type(f2_out), type(c_out), type(d_out), type(f3_out), "\n")
    print(len(a_out), len(f1_out), len(b_out), len(f2_out), len(c_out), len(d_out), len(f3_out), "\n")

    # return in this order
    return a_out, f1_out, b_out, f2_out, c_out, d_out, f3_out 
