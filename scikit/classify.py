#!/usr/bin/python3.4

import sys
import h5py
import numpy as np

from sklearn                import preprocessing, metrics, cross_validation, linear_model
from sklear.decomposition   import PCA


def load_input(inpath):
    """
    Reads features and labels from an hdf5 input file.
    """

    inputs = []
    labels = None

    with h5py.File(inpath, 'r') as fin:
        inputs = fin['data'][:]

        if 'label' in fin:
            labels = fin['label'][:]

    return inputs, labels


def classification_loss(Y_pred, Y_true):
    """
    """

    return np.sum(Y_pred != Y_true) / Y_true.size


def output_predictions(classifier, inpath, outpath):
    """
    Generates a file containing predicted labels.
    """

    X, Y = load_input(inpath)
    Ypred = classifier.predict(X)

    np.savetxt(Ypred, fmt="%i") # the last parameter converts the floats to ints


def training(X, Y):
    """
    Describes the training-steps of our classifier.
    """

    pca = PCA(n_components=600)
    X = pca.fit_transform(X)

    classifier = linear_model.LogisticRegression()
    classifier.fit(X, Y)

    return classifier


def main():
    # Read labelled training data.
    X, Y = load_input('../data/train.h5')

    if '--output' in sys.argv:
        classifier = training(X, Y)

        print("Outputting predictions...")
        output_predictions(classifier, '../data/validate.h5', 'out/labels.txt')

    else:
        Xtrain, Xtest, Ytrain, Ytest = cross_validation.train_test_split(X, Y, train_size=0.75)
        classifier = training(Xtrain, Ytrain)

        scorefun = metrics.make_scorer(classification_loss)
        scores = cross_validation.cross_val_score(classifier, X, Y, scoring=scorefun, cv=5)

        print('Mean: %s +/- %s' % (np.mean(scores), np.std(scores)))


if __name__ == "__main__":
    main()
