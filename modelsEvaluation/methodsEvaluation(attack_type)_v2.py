from statistics import mean

import numpy as np
import pandas as pd
import csv
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
from sklearn.multiclass import OneVsOneClassifier
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
x_train = x_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
x_test = x_test.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

mms = MinMaxScaler()
mms.fit(x_train)
x_train = mms.transform(x_train)
x_test = mms.transform(x_test)

with open("models_results(attack_type).csv", "a", newline="") as file:
    def get_confusion_matrix_elements(cm):
        FP = cm.sum(axis=0) - np.diag(cm)
        FN = cm.sum(axis=1) - np.diag(cm)
        TP = np.diag(cm)
        TN = cm.sum() - (FP + FN + TP)
        return TN, FP, FN, TP

    writer = csv.writer(file)
    writer.writerow(['Name', 'Test Accuracy', 'Precision', 'Recall', 'F1-score'])

    # conversione in indici di tipo int per poterli usare per le posizioni dell'array
    unique_classes = np.unique(y_train)
    class_to_index = {cls: idx for idx, cls in enumerate(unique_classes)}
    index_to_class = {idx: cls for idx, cls in enumerate(unique_classes)}

    class_pairs = [(i, j) for i in unique_classes for j in unique_classes if i < j] # tutte le coppie di classi
    classifiers = {}

    for (i, j) in class_pairs:
        # seleziono il subset contenente gli elementi della coppia
        idx = np.where((y_train == i) | (y_train == j))[0]
        x_pair = x_train[idx]
        y_pair = y_train[idx]

        # conversione in 0-1, altrimenti non viene accettato da perpetual
        y_pair_binary = np.where(y_pair == i, 0, 1)

        clf = PerpetualBooster(objective="LogLoss")
        clf.fit(x_pair, y_pair_binary, budget=0.1)
        classifiers[(i, j)] = clf

    def ovo_predict(X):
        votes = np.zeros((X.shape[0], len(np.unique(y_train)))) # creo matrice di voti

        count = 0
        for (i, j), clf in classifiers.items():
            # seleziono subset come prima
            idx = np.where((y_test == i) | (y_test == j))[0]
            x_pair_test = x_test[idx]
            y_pair_test = y_test[idx]

            pred = clf.predict(x_pair_test)

            if not np.isnan(pred).any():  # il modello genera errori su alcune coppie, è un problema nel metodo fit, interno al modello
                # per convertire l'output della log function a binario
                fpr, tpr, tresholds = roc_curve(y_pair_test, pred, pos_label=j)
                j_scores = tpr - fpr
                optimal_idx = np.argmax(j_scores)
                optimal_treshold = tresholds[optimal_idx]
                binary_pred = (pred >= optimal_treshold).astype(int)

                # aggiungo i voti del classificatore alla matrice
                votes[idx, class_to_index[i]] += (binary_pred == 0)
                votes[idx, class_to_index[j]] += (binary_pred == 1)

            print('{} pair done'.format(count))
            count+=1

        return np.array([index_to_class[np.argmax(vote)] for vote in votes])
    y_pred = ovo_predict(x_test)

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
