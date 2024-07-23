from statistics import mean

import numpy as np
import pandas as pd
import csv
import time
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from matplotlib import pyplot as plt
from ngboost import NGBClassifier, NGBRegressor
from sklearn import svm
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, GradientBoostingRegressor, \
    HistGradientBoostingClassifier, HistGradientBoostingRegressor, AdaBoostClassifier, AdaBoostRegressor, \
    RandomForestClassifier, RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Lasso, ElasticNet, LassoLars, \
    OrthogonalMatchingPursuit, SGDClassifier, SGDRegressor
from sklearn.metrics import accuracy_score, classification_report, roc_curve, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB, ComplementNB, CategoricalNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestCentroid
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from perpetual import PerpetualBooster
from starboost import BoostingRegressor, BoostingClassifier
from xgboost import XGBClassifier, XGBRegressor

df = pd.read_csv("modbusDataset.csv", sep=",", header=0, index_col=None)

x = df[['flow', 'modbus_pdu', 'src_port', 'dst_port', 'ip_len', 'ip_chksum', 'modbus_len',
        'modbus_start_addr_NA_IND', 'modbus_output_addr_NA_IND', 'modbus_output_value_NA_IND',
        'modbus_byte_count_NA_IND', 'modbus_coil_status_0_NA_IND', 'modbus_register_val_NA_IND',
        'modbus_register_addr_NA_IND']]
y = df['attack_type']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

mms = MinMaxScaler()
mms.fit(x_train)
x_train = mms.transform(x_train)
x_test = mms.transform(x_test)

with open("models_results_attack_type_OVR.csv", "a", newline="") as file:
    def get_confusion_matrix_elements(cm):
        FP = cm.sum(axis=0) - np.diag(cm)
        FN = cm.sum(axis=1) - np.diag(cm)
        TP = np.diag(cm)
        TN = cm.sum() - (FP + FN + TP)
        return TN, FP, FN, TP

    writer = csv.writer(file)
    writer.writerow(['Name', 'Test Accuracy', 'Precision', 'Recall', 'F1-score'])

    unique_classes = np.unique(y_train)
    classifiers = {}

    for cls in unique_classes:
        y_train_binary = (y_train == cls).astype(int)
        clf = PerpetualBooster(objective="LogLoss")
        clf.fit(x_train, y_train_binary, budget=0.1)
        classifiers[int(cls)] = clf

    predictions = np.zeros((x_test.shape[0], len(unique_classes)))
    for cls, clf in classifiers.items():
        pred = clf.predict(x_test)

        fpr, tpr, tresholds = roc_curve((y_test == cls).astype(int), pred)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_treshold = tresholds[optimal_idx]
        y_pred = (pred >= optimal_treshold).astype(int)

        predictions[:, cls] = y_pred

    y_pred = np.argmax(predictions, axis=1)

    PPBacc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = get_confusion_matrix_elements(cm)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * ((precision * recall) / (precision + recall))
    print("------PerpetualBooster------")
    print(f'Accuracy on the test set: {PPBacc:.4f}')
    print(classification_report(y_test, y_pred))
    writer.writerow([
        "PerpetualBooster", round(PPBacc * 100, 2),
        round(mean(precision), 2), round(mean(recall), 2), round(mean(f1_score), 2),
    ])
