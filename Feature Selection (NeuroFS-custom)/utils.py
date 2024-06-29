from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import errno
import os
import sys
import pickle
sys.path.append(os.getcwd())


"""*************************************************************************"""
"""                              Functions                                  """


def check_path(filename):
    import os
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: 
            if exc.errno != errno.EEXIST:
                raise 


def save_obj(name, obj):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


"""*************************************************************************"""
"""                             Load data                                   """


def load_data(args):
    data = pd.read_csv(args.data, sep=",", header=0, index_col=None)
    args.column_names = list(data.columns)
    data = data.values

    X = data[:, :-2]  # get all columns except for data_type and attack_type
    y = data[:, -1:]  # target is data_type
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    y_test = y_test.ravel()
    y_train = y_train.ravel()

    # apply minmaxscaler to data
    mms = MinMaxScaler()
    mms.fit(X_train)
    X_train = mms.transform(X_train)
    X_test = mms.transform(X_test)
        
    print("train labels: ", np.unique(y_train))
    print("test labels: ", np.unique(y_test))
    
    args.num_classes = len(np.unique(y_train))
    args.dim = X_train.shape[1]
    args.dim_to_remove = int(args.dim - args.K)
    print("dim_to_remove = ", args.dim_to_remove)
    
    args.epoch_remove = int(args.epochs*args.frac_epoch_remove)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    y_train = y_train.astype('int')
    y_test = y_test.astype('int')

    y_train = np.asarray(pd.get_dummies(y_train))
    y_test = np.asarray(pd.get_dummies(y_test))
    print("X_train: ", X_train.shape)
    print("y_train: ", y_train.shape)
    print("X_test: ", X_test.shape)
    print("y_test: ", y_test.shape)  

    return args, [X_train, X_test, y_train, y_test]


"""*************************************************************************"""
"""                             Evaluation                                  """


def eval_subset_supervised(train, test):
    print("X_train shape = " + str(train[0].shape))
    print("X_test shape = " + str(test[0].shape))

    clf = KNeighborsClassifier(n_neighbors=1, algorithm='kd_tree', n_jobs=-1)
    clf.fit(train[0], train[2])
    KNNacc = float(clf.score(test[0], test[2]))
    print('KNN done')

    clf = ExtraTreesClassifier(n_estimators=50, n_jobs=-1)
    clf.fit(train[0], train[2])
    ETacc = float(clf.score(test[0], test[2]))
    print('ET done')

    clf = svm.LinearSVC()
    clf.fit(train[0], train[2])
    SVCacc = float(clf.score(test[0], test[2]))
    print('SVC done')

    print('KNNacc = {:.3f}, ETacc = {:.3f}, SVCacc = {:.3f}'.format(KNNacc, ETacc, SVCacc))

    return KNNacc, ETacc, SVCacc
