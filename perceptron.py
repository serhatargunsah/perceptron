from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class SimplePerceptron(BaseEstimator, ClassifierMixin):
    def __init__(self, learning_rate=1.0, k_max=1000):
        self.w_ = None # weights
        self.k_ = None # counter of update steps
        self.class_labels_ = None
        self.k_max_ = k_max
        self.learning_rate_ = learning_rate
    
    def fit(self, X, y):
        m, n = X.shape
        self.class_labels_ = np.unique(y)
        #print(self.class_labels_)
        yy = np.zeros(m, dtype=np.int8) # class labels mapped to: -1, 1
        yy[y == self.class_labels_[0]] = -1
        yy[y == self.class_labels_[1]] = 1        
        self.w_ = np.zeros(n + 1)
        X = np.c_[np.ones(m), X] # adding a column of ones in front of actual features
        self.k_ = 0
        while self.k_ < self.k_max_:
            E = [] # list for indexes of misclassified points
            for i in range(m): # for loop checks which data points are misclassified
                s = self.w_.dot(X[i])
                f = 1 if s > 0 else -1
                if f != yy[i]:
                    E.append(i)
            if len(E) == 0:
                break # now whole data set correctly classified
            i = np.random.choice(E)
            self.w_ = self.w_ + self.learning_rate_ * yy[i] * X[i] # UPDATE FORMULA (!)
            self.k_ += 1  
                
    def predict(self, X):
        m = X.shape[0]
        X = np.c_[np.ones(m), X] # adding a column of ones in front of actual features
        sums = self.w_.dot(X.T)
        predictions = np.ones(m, dtype=np.int8)
        predictions[sums <= 0.0] = 0 # mathematically, corresponds to -1
        return self.class_labels_[predictions]