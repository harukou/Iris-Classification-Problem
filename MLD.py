import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from numpy.linalg import pinv,inv,norm
from scipy.linalg import eig
from sklearn.preprocessing import normalize
from SplitData import split_train_test

def mld_w(C1_train,C2_train,C3_train):
    '''
    A three class classifier using fisher linear
    c1_train: class 1 training set 
    c2_train: class 2 training set
    c3_train: class 3 training set 
    '''
    # compute the mean vectors of each class
    m1 = np.mean(C1_train,axis=0).reshape(4,1)
    m2 = np.mean(C2_train,axis=0).reshape(4,1)
    m3 = np.mean(C3_train,axis=0).reshape(4,1)
    m  = np.mean(np.concatenate((C1_train,C2_train,C3_train),axis=0),axis=0).reshape(4,1)    
    sb = 25*((m1-m).dot((m1-m).T)+(m2-m).dot((m2-m).T)+(m3-m).dot((m3-m).T))
    # compute scatter matrices and within scatter matrix
    s1 = (C1_train.shape[0])*np.cov(C1_train,rowvar=False)
    s2 = (C2_train.shape[0])*np.cov(C2_train,rowvar=False)
    s3 = (C3_train.shape[0])*np.cov(C3_train,rowvar=False)
    sw = s1 + s2 + s3
    # compute the projection direction w
    v,w = eig(sb,sw) # v= eigenvalue,w=eigenvector
    
    return v,w



def mld(data):
    '''
    Multiple Linear Discriminant for 3 class problem
    split the data and solve the generalized eigenvalue and eigenvector
    data: iris data
    '''
    #split the data into training and test dataset 
    X_train,X_test,Y_train,Y_test = split_train_test(data)
    # X_train size: 25*4
    X1_train = X_train.T[:25] 
    X2_train = X_train.T[25:50]
    X3_train = X_train.T[50:]

    X1_test = X_test.T[:25]
    X2_test = X_test.T[25:50]
    X3_test = X_test.T[50:]
    
    # w, the optimal projection matrix 
    v,w = mld_w(X1_train,X2_train,X3_train)
    # retrieve the eigenvector associated with the greatest eigenvalue
    idx = np.argmax(v)
    pw = w[:,idx].reshape(4,1)
    # compute the output matrix Y 25*3
    y = X_test.T.dot(pw).reshape((25,3),order='F')
    y=np.sort(y,axis=0)
    y=np.sort(y,axis=1)
    # boundary1 is to seperate class1 and class2
    boundary1 = np.min(y[:,1])
    # boundary2 is to seperate class2 from class1, class3
    boundary21 = np.max(y[:,0])
    boundary22 = np.min(y[:,2])
    # boundary3 is to seperate class3 and class2
    boundary3 = np.max(y[:,1])
    
    y1 = y[:,0]
    # class1 classification accuracy num, it should less than class2
    acc1 = len(y1[y1<=boundary1])
    
    y2 = y[:,1]
    # class2 classification accuracy num
    acc2 = len(y2[(y2>=boundary21) & (y2<=boundary22)])
    
    y3 = y[:,2]
    # class3 classification accuracy
    acc3 = len(y3[y3>=boundary3])
    
    acc = (acc1+acc2+acc3)/X_test.shape[1]    
    return acc,y

if __name__ == "__main__":
       
    # load the iris.data 
    data = pd.read_csv("iris.data",header=None)
    
    # store the classifier's accuracy rate
    mld_acc_lst = np.zeros(20)
    

    # loop 20 times for training and testing
    for i in range(20):
        mld_acc_lst[i],y = mld(data)
        print('Projection Direction of X_Test:')
        print(y)
        

    # evaluate the mean and variance of accuracy rates for both methods
    mld_acc_mean = np.mean(mld_acc_lst)
    mld_acc_var = np.var(mld_acc_lst)

    print('mld_acc_mean=%.3f' %(mld_acc_mean))
    print('mld_acc_var= %.6f' %(mld_acc_var))
   