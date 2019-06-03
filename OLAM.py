import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from numpy.linalg import pinv,inv,norm
from sklearn.preprocessing import normalize
from SplitData import split_train_test

def olam(data):

    X_train,X_test,Y_train,Y_test = split_train_test(data)

    # train and compute the OLAM map matrix
    M = np.dot(Y_train,pinv(X_train))

    # test and compute the output class Y_out 
    Y_out = np.dot(M,X_test)

    # normalize the Y_out so that it has the same format with Y_test
    Y_out = normalize(Y_out, axis=0, norm='max')
    Y_out[Y_out!=1]=0

    TF = (Y_out.T==Y_test.T)
    acc = np.sum(np.all(TF , axis = 1).astype(int))/Y_test.shape[1]
    return acc

if __name__ == "__main__":
       
    # load the iris.data 
    data = pd.read_csv("iris.data",header=None)

    # store the classifier's accuracy rate
    olam_acc_lst = np.zeros(20)
    #fisher_acc_lst = np.zeros(20)

    # loop 20 times for training and testing
    for i in range(20):
        olam_acc_lst[i] = olam(data)
        #fisher_acc_lst[i] = fisher(data)

    # evaluate the mean and variance of accuracy rates for both methods
    olam_acc_mean = np.mean(olam_acc_lst)
    olam_acc_var = np.var(olam_acc_lst)

    print('olam_acc_mean=%.3f' %(olam_acc_mean))
    print('olam_acc_var= %.6f' %(olam_acc_var))
