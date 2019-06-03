import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from numpy.linalg import pinv,inv,norm
from sklearn.preprocessing import normalize

def split_train_test(data,rs=None):
    """
    This is the func to achieve the training and testing dataset in dividing
    the data of each class into half.
    data: iris dataset
    rs: random state, we can use it to check the result.
    return the training data and test data with shape:
    4x75 , 3x75 , 4x75 , 3x75    
    """
    # setosa_data, X1 is the feature matrix, Y1 is the label matrix
    X1 = data[:50].values[:,:4].astype('float')
    Y1 = np.tile(np.array([1,0,0]),(50,1))
    # versicolor_data, X2 is the feature matrix, Y2 is the label matrix
    X2 = data[50:100].values[:,:4].astype('float')
    Y2 = np.tile(np.array([0,1,0]),(50,1))
    # virginica_data, X3 is the feature matrix, Y3 is the label matrix
    X3 = data[100:].values[:,:4].astype('float')
    Y3 = np.tile(np.array([0,0,1]),(50,1))

    # divide data of each class into half, one half for training, the other for testing
    
    X1_train, X1_test, Y1_train, Y1_test = train_test_split(
            X1, Y1, test_size=0.5, random_state=rs)
    X2_train, X2_test, Y2_train, Y2_test = train_test_split(
            X2, Y2, test_size=0.5, random_state=rs)
    X3_train, X3_test, Y3_train, Y3_test = train_test_split(
            X3, Y3, test_size=0.5, random_state=rs)

    # construct the training and testing dataset of iris
    X_train = np.concatenate((X1_train,X2_train,X3_train),axis=0) 
    X_test = np.concatenate((X1_test,X2_test,X3_test),axis=0) 
    Y_train = np.concatenate((Y1_train,Y2_train,Y3_train),axis=0)
    Y_test = np.concatenate((Y1_test,Y2_test,Y3_test),axis=0)

    X_train = X_train.T
    X_test = X_test.T
    Y_train = Y_train.T
    Y_test = Y_test.T
    
    return X_train,X_test,Y_train,Y_test


if __name__ == "__main__":
    df = pd.read_csv('iris.data',header=None)
    X_train,X_test,Y_train,Y_test = split_train_test(df)
    
    print('X_train:')
    print(X_train)
    print('X_test:')
    print(X_test)
    print('Y_train:')
    print(Y_train)
    print('Y_test:')
    print(Y_test)
