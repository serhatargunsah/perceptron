import numpy as np
from matplotlib import pyplot as plt
from perceptron import SimplePerceptron
import random

def fake_data(): # m - no. of examples    
    D = np.genfromtxt("wdbc.data", delimiter=",")
    y = D[:, 1].astype(np.int8)
    X = D[:, 2:]
    return X, y

def train_test_split(X, y, train_fraction=0.75, seed=0):
    np.random.seed(seed)
    m = X.shape[0] # m - no. of examples
    indexes = np.random.permutation(m)
    X = X[indexes]
    y = y[indexes]
    i = int(np.round(train_fraction * m))
    X_train = X[:i]
    y_train = y[:i]
    X_test = X[i:]
    y_test = y[i:]
    return X_train, y_train, X_test, y_test

if __name__ == '__main__':
    for kmax in [1000, 2000, 10000]:
        ACC_SUM = 0
        for i in range(1, 11):
            #BINS = 5
            seed_value = random.randint(0, 10000) # (default 2)
            X, y = fake_data()
            X_train, y_train, X_test, y_test = train_test_split(X, y, train_fraction=0.75, seed=seed_value)
            #X_train_d, mins_ref, maxes_ref = discretize(X_train, bins=BINS)    
            #X_test_d, _, _ = discretize(X_test, bins=BINS, mins_ref=mins_ref, maxes_ref=maxes_ref)
            clf = SimplePerceptron(learning_rate=1.0, k_max=kmax)
            clf.fit(X_train, y_train)
            print("------------------------------------EXPERIMENT NUMBER " + str(i) + "------------------------------------")
            print(f"WEIGHTS: {clf.w_}")
            print(f"STEPS: {clf.k_}")
            print(f"ACC: {clf.score(X_test, y_test)}")
            ACC_SUM += clf.score(X_test, y_test)
        ACC_AVG = ACC_SUM / 10
        print(f"AVG.  test ACC FOR KMAX {kmax}: {ACC_AVG}")
    
    #x1 = np.array([10, 40])
    #x2 = -(clf.w_[0] + clf.w_[1] * x1) / clf.w_[2] 
    #x3 = np.array([x2[1],x2[0]]) # x3 = np.array([x2[1]/2,x2[1]/2])
    #plt.scatter(X[:, 1], X[:, 2], c=y, cmap="coolwarm", s=5)
    #plt.plot(x1, x3, c="black")
    #plt.show()