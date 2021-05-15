# -*- coding: utf-8 -*-

import numpy as np
import math
from scipy.stats import multivariate_normal
from sklearn.utils import shuffle

def generate_semi_definite(dim):
    '''
    Generate a semi definite matrix to be sure that the matrix is inversible

    params:
    ----------
    dim: Number of dimensions (int)


    returns:
    ----------
    gen_mat: The generated matrix of shape (dim, dim)
    '''
    tmp = np.random.rand(dim, dim)
    return np.dot(tmp.T, tmp)

def compute_cov(X, mean):
    '''
    Compute the covariance of the data

    params:
    ----------
    X: Input data of shape (n_samples, n_features)

    mean: Mean of the current class (n_features,)


    returns:
    ----------
    cov_mat: covariance matrix of shape (n_features, n_features)
    '''
    n, dim = X.shape
    mean = np.reshape(mean, (dim, 1))
    res = np.zeros((dim, dim))
    for i in range(n):
        x = X[i]
        x = np.reshape(x, (dim, 1))
        res = np.add(res, np.dot((x - mean), (x - mean).T))
    tmp = res / n
    return np.dot(tmp.T, tmp)

class GaussianNaiveBayesClassifier:
    def __init__(self, nb_gaussian, n_features, cov_type='identity'):
        '''
        Instantiation of the gausian classifier

        params:
        ----------
        nb_gaussian: Number of gaussian to create (== number of classes)

        cov_type: Type of covariance matrix, can be: ['identity', 'same', 'unique']
            'identity': All the gaussian have the identity matrix as covariance
            'same': All the gaussian have random same covariance matrix
            'unique': All the gaussian have unique covariance matrix computed of each class

        n_features: Dimension of the data (number of features)


        returns:
        ----------
        The new GaussianNaiveBayes object classifier
        '''
        self.nb_gaussian = nb_gaussian
        self.cov_type = cov_type
        self.means = np.zeros((nb_gaussian, n_features))
        self.covs = np.zeros((nb_gaussian, n_features, n_features))

    def fit(self, X, y):
        '''
        Fit the GaussianNaiveBayes Classifier
        Compute the mean and the covariance (if necessary) for each gaussian
        The model is going to learn the spatial representation of each class (using a gaussian)

        params:
        ----------
        X: Data of shape (n_samples, n_features)
        y: Labels of shape (n_samples,)
        '''
        n, dim = X.shape

        if (self.cov_type == 'same'): ## same covariance for all gaussian
            self.covs = generate_semi_definite(dim)

        for i in range(self.nb_gaussian):
            print("Fit Gaussian nb. {}".format(i))

            ## compute the mean for each gaussian
            datas = X[y == i] ## get the data that have the label i
            self.means[i] = np.mean(datas, axis=0)
            if (self.cov_type == 'identity'): ## identity covariance for all gaussian
                self.covs[i] = np.identity(dim)
            elif (self.cov_type == 'unique'):
                self.covs[i] = compute_cov(datas, self.means[i])

    def predict(self, X):
        '''
        Make predictions

        params:
        ----------
        X: Data of shape (n_samples, n_features)


        returns:
        ----------
        res: Result of prediction of shape (n_samples,)
        '''
        res = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            pred = np.zeros(self.nb_gaussian) ## predictions for all gaussian
            for j in range(self.nb_gaussian):
                pred[j] = multivariate_normal.logpdf(X[i], self.means[j], self.covs[j])
            res[i] = np.argmax(pred)
        return res

    def measure_accuracy(self, preds, y, verbose=0):
        '''
        Measure the accuracy of a model

        params:
        ----------
        preds: Predictions array of shape (n_samples,)
        y: Reference array of shape (n_samples,)


        returns:
        ----------
        acc: Accuracy of the predictions (float)
        '''
        n = preds.shape[0]
        nb_good = 0
        for i in range(n):
            nb_good += (preds[i] == y[i])
            if verbose:
                print("prediction: {}   expected: {}".format(int(preds[i]), y[i]))
        return nb_good / n
