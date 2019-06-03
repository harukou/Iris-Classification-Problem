import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from numpy.linalg import pinv,inv,norm
from sklearn.preprocessing import normalize
from SplitData import split_train_test

def fld_w(C1_train,C2_train):
    '''
    A two class classifier using fisher linear
    c1_train: class 1 training set 
    c2_train: class 2 training set 
    '''
    # compute the mean vectors of each class
    m1 = np.mean(C1_train,axis=0).reshape(4,1)
    m2 = np.mean(C2_train,axis=0).reshape(4,1)
    # compute scatter matrices and within scatter matrix
    s1 = (C1_train.shape[0])*np.cov(C1_train,rowvar=False)
    s2 = (C2_train.shape[0])*np.cov(C2_train,rowvar=False)
    sw = s1 + s2
    # compute the projection direction w
    w = inv(sw).dot(m1-m2)
    # normalize
    w=w/norm(w)
    return w

def fld(data):
    '''
    Fisher Linear Discriminant for 3 class problem
    split the data and compute the projection w
    use the w to get the low dimension y and classify 3 class
    data: iris data
    '''
    #split the data into training and test dataset 
    X_train,X_test,Y_train,Y_test = split_train_test(data)
    
    X1_train = X_train.T[:25]
    X2_train = X_train.T[25:50]
    X3_train = X_train.T[50:]

    X1_test = X_test.T[:25]
    X2_test = X_test.T[25:50]
    X3_test = X_test.T[50:]
    
    # w1, the optimal projection, is used to classify x1 class
    w1 = fld_w(X1_train,np.concatenate((X2_train,X3_train),axis=0))
    # w2, the optimal projection, is used to classify x2 class
    w2 = fld_w(X2_train,np.concatenate((X1_train,X3_train),axis=0))
    # w3, the optimal projection, is used to classify x3 class
    w3 = fld_w(X3_train,np.concatenate((X2_train,X1_train),axis=0))

    # compute the vector y to determine the classification
    # from the experiment on train datasets. 
    y1 = X_test.T.dot(w1).reshape(25,3,order='F') 
    y2 = X_test.T.dot(w2).reshape(25,3,order='F')
    y3 = X_test.T.dot(w3).reshape(25,3,order='F')
    
    # sort the y and determine the boundary among the classes.
    y1 = np.sort(y1,axis=0)
    y2 = np.sort(y2,axis=0)
    y3 = np.sort(y3,axis=0)
    
    # define the boundary
    '''
    let the class which need to be classified become the boundary, so 
    we have 2 boundaries in every classifiers, one is the min value, the 
    other is the max value.
    We can test if the other 2 class data belong to the class so that we
    can estimate the classification accuracy. 
    '''
    boundary_c1_min = y1[:,0][0]
    boundary_c1_max = y1[:,0][-1]
    boundary_c2_min = y2[:,1][0]
    boundary_c2_max = y2[:,1][-1]
    boundary_c3_min = y3[:,2][0]
    boundary_c3_max = y3[:,2][-1]

    # compute the accuracy, accx = how many data points are belong to wrong class
    y12 = y1[:,1]
    y13 = y1[:,2]
    acc1 = len(y12[(y12>boundary_c1_min) & (y12<boundary_c1_max)]) + len(y13[(y13>boundary_c1_min) & (y13<boundary_c1_max)])

    y21 = y2[:,0]
    y23 = y2[:,2]
    acc2 = len(y21[(y21>boundary_c2_min) & (y21<boundary_c2_max)]) + len(y23[(y23>boundary_c2_min) & (y23<boundary_c2_max)])

    y31 = y3[:,0]
    y32 = y3[:,1]
    acc3 = len(y31[(y31>boundary_c3_min) & (y31<boundary_c3_max)]) + len(y32[(y32>boundary_c3_min) & (y32<boundary_c3_max)])

    acc = (len(y12)*2*3-acc1-acc2-acc3)/(len(y12)*2*3) 
    
    return acc
    
if __name__ == "__main__":
       
    # load the iris.data 
    data = pd.read_csv("iris.data",header=None)

    # store the classifier's accuracy rate
    fld_acc_lst = np.zeros(20)
    

    # loop 20 times for training and testing
    for i in range(20):
        fld_acc_lst[i] = fld(data)
                

    # evaluate the mean and variance of accuracy rates for both methods
    fld_acc_mean = np.mean(fld_acc_lst)
    fld_acc_var = np.var(fld_acc_lst)

    print('fld_acc_mean=%.3f' %(fld_acc_mean))
    print('fld_acc_var= %.6f' %(fld_acc_var))
