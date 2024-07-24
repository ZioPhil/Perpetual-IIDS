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

# conversione in indici di tipo int per poterli usare per le posizioni dell'array
unique_classes = np.unique(y_train)
class_to_index = {cls: idx for idx, cls in enumerate(unique_classes)}
index_to_class = {idx: cls for idx, cls in enumerate(unique_classes)}
class_pairs = [(i, j) for i in unique_classes for j in unique_classes if i < j]  # tutte le coppie di classi


def get_confusion_matrix_elements(cm):
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    return TN, FP, FN, TP


def ovo_predict_perpetual(X):
    votes = np.zeros((X.shape[0], len(np.unique(y_train))))  # creo matrice di voti

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

    return np.array([index_to_class[np.argmax(vote)] for vote in votes])


def ovo_predict(X):
    votes = np.zeros((X.shape[0], len(np.unique(y_train))))  # creo matrice di voti

    for (i, j), clf in classifiers.items():
        # seleziono subset come prima
        idx = np.where((y_test == i) | (y_test == j))[0]
        x_pair_test = x_test[idx]

        pred = clf.predict(x_pair_test)

        if not np.isnan(pred).any():
            # aggiungo i voti del classificatore alla matrice
            votes[idx, class_to_index[i]] += (pred == 0)
            votes[idx, class_to_index[j]] += (pred == 1)

    return np.array([index_to_class[np.argmax(vote)] for vote in votes])


with open("models_results_attack_type_OVO.csv", "a", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(['Name', 'Test Accuracy', 'Precision', 'Recall', 'F1-score'])

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
    y_pred = ovo_predict_perpetual(x_test)
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

    max_iter = 50
    classifiers = {}
    for (i, j) in class_pairs:
        # seleziono il subset contenente gli elementi della coppia
        idx = np.where((y_train == i) | (y_train == j))[0]
        x_pair = x_train[idx]
        y_pair = y_train[idx]

        # conversione in 0-1, altrimenti non viene accettato da perpetual
        y_pair_binary = np.where(y_pair == i, 0, 1)

        clf = SGDClassifier(max_iter=max_iter, random_state=42)
        clf.fit(x_pair, y_pair_binary)
        classifiers[(i, j)] = clf
    y_pred = ovo_predict(x_test)
    SGCacc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = get_confusion_matrix_elements(cm)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * ((precision * recall) / (precision + recall))
    print("------StochasticGradientDescent------")
    print(f'Accuracy on the test set: {SGCacc:.4f}')
    print(classification_report(y_test, y_pred))
    writer.writerow([
        "StochasticGradientDescent", round(SGCacc * 100, 2),
        round(mean(precision), 2), round(mean(recall), 2), round(mean(f1_score), 2),
    ])

    classifiers = {}
    for (i, j) in class_pairs:
        # seleziono il subset contenente gli elementi della coppia
        idx = np.where((y_train == i) | (y_train == j))[0]
        x_pair = x_train[idx]
        y_pair = y_train[idx]

        # conversione in 0-1, altrimenti non viene accettato da perpetual
        y_pair_binary = np.where(y_pair == i, 0, 1)

        clf = CategoricalNB()
        clf.fit(x_pair, y_pair_binary)
        classifiers[(i, j)] = clf
    y_pred = ovo_predict(x_test)
    CANBacc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = get_confusion_matrix_elements(cm)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * ((precision * recall) / (precision + recall))
    print("------CategoricalNaiveBayes------")
    print(f'Accuracy on the test set: {CANBacc:.4f}')
    print(classification_report(y_test, y_pred))
    writer.writerow([
        "CategoricalNaiveBayes", round(CANBacc * 100, 2),
        round(mean(precision), 2), round(mean(recall), 2), round(mean(f1_score), 2),
    ])

    classifiers = {}
    for (i, j) in class_pairs:
        # seleziono il subset contenente gli elementi della coppia
        idx = np.where((y_train == i) | (y_train == j))[0]
        x_pair = x_train[idx]
        y_pair = y_train[idx]

        # conversione in 0-1, altrimenti non viene accettato da perpetual
        y_pair_binary = np.where(y_pair == i, 0, 1)

        clf = KNeighborsClassifier(algorithm='kd_tree')
        clf.fit(x_pair, y_pair_binary)
        classifiers[(i, j)] = clf
    y_pred = ovo_predict(x_test)
    KNNacc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = get_confusion_matrix_elements(cm)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * ((precision * recall) / (precision + recall))
    print("------KNeighbors------")
    print(f'Accuracy on the test set: {KNNacc:.4f}')
    print(classification_report(y_test, y_pred))
    writer.writerow([
        "KNeighbors", round(KNNacc * 100, 2),
        round(mean(precision), 2), round(mean(recall), 2), round(mean(f1_score), 2),
    ])

    classifiers = {}
    for (i, j) in class_pairs:
        # seleziono il subset contenente gli elementi della coppia
        idx = np.where((y_train == i) | (y_train == j))[0]
        x_pair = x_train[idx]
        y_pair = y_train[idx]

        # conversione in 0-1, altrimenti non viene accettato da perpetual
        y_pair_binary = np.where(y_pair == i, 0, 1)

        clf = NearestCentroid()
        clf.fit(x_pair, y_pair_binary)
        classifiers[(i, j)] = clf
    y_pred = ovo_predict(x_test)
    NECacc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = get_confusion_matrix_elements(cm)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * ((precision * recall) / (precision + recall))
    print("------NearestCentroid------")
    print(f'Accuracy on the test set: {NECacc:.4f}')
    print(classification_report(y_test, y_pred))
    writer.writerow([
        "NearestCentroid", round(NECacc * 100, 2),
        round(mean(precision), 2), round(mean(recall), 2), round(mean(f1_score), 2),
    ])

    max_iter = 50
    classifiers = {}
    for (i, j) in class_pairs:
        # seleziono il subset contenente gli elementi della coppia
        idx = np.where((y_train == i) | (y_train == j))[0]
        x_pair = x_train[idx]
        y_pair = y_train[idx]

        # conversione in 0-1, altrimenti non viene accettato da perpetual
        y_pair_binary = np.where(y_pair == i, 0, 1)

        clf = svm.LinearSVC(max_iter=max_iter, random_state=42)
        clf.fit(x_pair, y_pair_binary)
        classifiers[(i, j)] = clf
    y_pred = ovo_predict(x_test)
    SVCacc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = get_confusion_matrix_elements(cm)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * ((precision * recall) / (precision + recall))
    print("------LinearSVC------")
    print(f'Accuracy on the test set: {SVCacc:.4f}')
    print(classification_report(y_test, y_pred))
    writer.writerow([
        "LinearSVC", round(SVCacc * 100, 2),
        round(mean(precision), 2), round(mean(recall), 2), round(mean(f1_score), 2),
    ])

    classifiers = {}
    for (i, j) in class_pairs:
        # seleziono il subset contenente gli elementi della coppia
        idx = np.where((y_train == i) | (y_train == j))[0]
        x_pair = x_train[idx]
        y_pair = y_train[idx]

        # conversione in 0-1, altrimenti non viene accettato da perpetual
        y_pair_binary = np.where(y_pair == i, 0, 1)

        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(x_pair, y_pair_binary)
        classifiers[(i, j)] = clf
    y_pred = ovo_predict(x_test)
    DTCacc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = get_confusion_matrix_elements(cm)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * ((precision * recall) / (precision + recall))
    print("------DecisionTree------")
    print(f'Accuracy on the test set: {DTCacc:.4f}')
    print(classification_report(y_test, y_pred))
    writer.writerow([
        "DecisionTree", round(DTCacc * 100, 2),
        round(mean(precision), 2), round(mean(recall), 2), round(mean(f1_score), 2),
    ])

    n_estimators = 100
    classifiers = {}
    for (i, j) in class_pairs:
        # seleziono il subset contenente gli elementi della coppia
        idx = np.where((y_train == i) | (y_train == j))[0]
        x_pair = x_train[idx]
        y_pair = y_train[idx]

        # conversione in 0-1, altrimenti non viene accettato da perpetual
        y_pair_binary = np.where(y_pair == i, 0, 1)

        clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        clf.fit(x_pair, y_pair_binary)
        classifiers[(i, j)] = clf
    y_pred = ovo_predict(x_test)
    RFCacc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = get_confusion_matrix_elements(cm)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * ((precision * recall) / (precision + recall))
    print("------RandomForest------")
    print(f'Accuracy on the test set: {RFCacc:.4f}')
    print(classification_report(y_test, y_pred))
    writer.writerow([
        "RandomForest", round(RFCacc * 100, 2),
        round(mean(precision), 2), round(mean(recall), 2), round(mean(f1_score), 2),
    ])

    n_estimators = 50
    classifiers = {}
    for (i, j) in class_pairs:
        # seleziono il subset contenente gli elementi della coppia
        idx = np.where((y_train == i) | (y_train == j))[0]
        x_pair = x_train[idx]
        y_pair = y_train[idx]

        # conversione in 0-1, altrimenti non viene accettato da perpetual
        y_pair_binary = np.where(y_pair == i, 0, 1)

        clf = ExtraTreesClassifier(n_estimators=n_estimators, random_state=42)
        clf.fit(x_pair, y_pair_binary)
        classifiers[(i, j)] = clf
    y_pred = ovo_predict(x_test)
    ETacc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = get_confusion_matrix_elements(cm)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * ((precision * recall) / (precision + recall))
    print("------ExtraTrees------")
    print(f'Accuracy on the test set: {ETacc:.4f}')
    print(classification_report(y_test, y_pred))
    writer.writerow([
        "ExtraTrees", round(ETacc * 100, 2),
        round(mean(precision), 2), round(mean(recall), 2), round(mean(f1_score), 2),
    ])

    max_iter = 125
    classifiers = {}
    for (i, j) in class_pairs:
        # seleziono il subset contenente gli elementi della coppia
        idx = np.where((y_train == i) | (y_train == j))[0]
        x_pair = x_train[idx]
        y_pair = y_train[idx]

        # conversione in 0-1, altrimenti non viene accettato da perpetual
        y_pair_binary = np.where(y_pair == i, 0, 1)

        clf = MLPClassifier(max_iter=max_iter, random_state=42)
        clf.fit(x_pair, y_pair_binary)
        classifiers[(i, j)] = clf
    y_pred = ovo_predict(x_test)
    MLCacc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = get_confusion_matrix_elements(cm)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * ((precision * recall) / (precision + recall))
    print("------MultiLayerPerceptron------")
    print(f'Accuracy on the test set: {MLCacc:.4f}')
    print(classification_report(y_test, y_pred))
    writer.writerow([
        "MultiLayerPerceptron", round(MLCacc * 100, 2),
        round(mean(precision), 2), round(mean(recall), 2), round(mean(f1_score), 2),
    ])

    n_estimators = 750
    classifiers = {}
    for (i, j) in class_pairs:
        # seleziono il subset contenente gli elementi della coppia
        idx = np.where((y_train == i) | (y_train == j))[0]
        x_pair = x_train[idx]
        y_pair = y_train[idx]

        # conversione in 0-1, altrimenti non viene accettato da perpetual
        y_pair_binary = np.where(y_pair == i, 0, 1)

        clf = GradientBoostingClassifier(n_estimators=n_estimators, random_state=42)
        clf.fit(x_pair, y_pair_binary)
        classifiers[(i, j)] = clf
    y_pred = ovo_predict(x_test)
    GBCacc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = get_confusion_matrix_elements(cm)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * ((precision * recall) / (precision + recall))
    print("------GradientBoosting------")
    print(f'Accuracy on the test set: {GBCacc:.4f}')
    print(classification_report(y_test, y_pred))
    writer.writerow([
        "GradientBoosting", round(GBCacc * 100, 2),
        round(mean(precision), 2), round(mean(recall), 2), round(mean(f1_score), 2),
    ])

    max_iter = 400
    classifiers = {}
    for (i, j) in class_pairs:
        # seleziono il subset contenente gli elementi della coppia
        idx = np.where((y_train == i) | (y_train == j))[0]
        x_pair = x_train[idx]
        y_pair = y_train[idx]

        # conversione in 0-1, altrimenti non viene accettato da perpetual
        y_pair_binary = np.where(y_pair == i, 0, 1)

        clf = HistGradientBoostingClassifier(max_iter=max_iter, random_state=42)
        clf.fit(x_pair, y_pair_binary)
        classifiers[(i, j)] = clf
    y_pred = ovo_predict(x_test)
    HGBCacc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = get_confusion_matrix_elements(cm)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * ((precision * recall) / (precision + recall))
    print("------HistGradientBoosting------")
    print(f'Accuracy on the test set: {HGBCacc:.4f}')
    print(classification_report(y_test, y_pred))
    writer.writerow([
        "HistGradientBoosting", round(HGBCacc * 100, 2),
        round(mean(precision), 2), round(mean(recall), 2), round(mean(f1_score), 2),
    ])

    n_estimators = 500
    classifiers = {}
    for (i, j) in class_pairs:
        # seleziono il subset contenente gli elementi della coppia
        idx = np.where((y_train == i) | (y_train == j))[0]
        x_pair = x_train[idx]
        y_pair = y_train[idx]

        # conversione in 0-1, altrimenti non viene accettato da perpetual
        y_pair_binary = np.where(y_pair == i, 0, 1)

        clf = AdaBoostClassifier(n_estimators=n_estimators, random_state=42)
        clf.fit(x_pair, y_pair_binary)
        classifiers[(i, j)] = clf
    y_pred = ovo_predict(x_test)
    ADACacc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = get_confusion_matrix_elements(cm)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * ((precision * recall) / (precision + recall))
    print("------AdaBoost------")
    print(f'Accuracy on the test set: {ADACacc:.4f}')
    print(classification_report(y_test, y_pred))
    writer.writerow([
        "AdaBoost", round(ADACacc * 100, 2),
        round(mean(precision), 2), round(mean(recall), 2), round(mean(f1_score), 2),
    ])

    n_estimators = 350
    classifiers = {}
    for (i, j) in class_pairs:
        # seleziono il subset contenente gli elementi della coppia
        idx = np.where((y_train == i) | (y_train == j))[0]
        x_pair = x_train[idx]
        y_pair = y_train[idx]

        # conversione in 0-1, altrimenti non viene accettato da perpetual
        y_pair_binary = np.where(y_pair == i, 0, 1)

        clf = XGBClassifier(n_estimators=n_estimators, random_state=42)
        clf.fit(x_pair, y_pair_binary)
        classifiers[(i, j)] = clf
    y_pred = ovo_predict(x_test)
    XGBacc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = get_confusion_matrix_elements(cm)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * ((precision * recall) / (precision + recall))
    print("------XGBoost------")
    print(f'Accuracy on the test set: {XGBacc:.4f}')
    print(classification_report(y_test, y_pred))
    writer.writerow([
        "XGBoost", round(XGBacc * 100, 2),
        round(mean(precision), 2), round(mean(recall), 2), round(mean(f1_score), 2),
    ])

    n_estimators = 700
    classifiers = {}
    for (i, j) in class_pairs:
        # seleziono il subset contenente gli elementi della coppia
        idx = np.where((y_train == i) | (y_train == j))[0]
        x_pair = x_train[idx]
        y_pair = y_train[idx]

        # conversione in 0-1, altrimenti non viene accettato da perpetual
        y_pair_binary = np.where(y_pair == i, 0, 1)

        clf = LGBMClassifier(n_estimators=n_estimators, random_state=42)
        clf.fit(x_pair, y_pair_binary)
        classifiers[(i, j)] = clf
    y_pred = ovo_predict(x_test)
    LGCacc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = get_confusion_matrix_elements(cm)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * ((precision * recall) / (precision + recall))
    print("------LightGBM------")
    print(f'Accuracy on the test set: {LGCacc:.4f}')
    print(classification_report(y_test, y_pred))
    writer.writerow([
        "LightGBM", round(LGCacc * 100, 2),
        round(mean(precision), 2), round(mean(recall), 2), round(mean(f1_score), 2),
    ])

    n_estimators = 250
    classifiers = {}
    for (i, j) in class_pairs:
        # seleziono il subset contenente gli elementi della coppia
        idx = np.where((y_train == i) | (y_train == j))[0]
        x_pair = x_train[idx]
        y_pair = y_train[idx]

        # conversione in 0-1, altrimenti non viene accettato da perpetual
        y_pair_binary = np.where(y_pair == i, 0, 1)

        clf = CatBoostClassifier(n_estimators=n_estimators, random_state=42)
        clf.fit(x_pair, y_pair_binary)
        classifiers[(i, j)] = clf
    y_pred = ovo_predict(x_test)
    CBCacc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = get_confusion_matrix_elements(cm)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * ((precision * recall) / (precision + recall))
    print("------CatBoost------")
    print(f'Accuracy on the test set: {CBCacc:.4f}')
    print(classification_report(y_test, y_pred))
    writer.writerow([
        "CatBoost", round(CBCacc * 100, 2),
        round(mean(precision), 2), round(mean(recall), 2), round(mean(f1_score), 2),
    ])

    n_estimators = 300
    classifiers = {}
    for (i, j) in class_pairs:
        # seleziono il subset contenente gli elementi della coppia
        idx = np.where((y_train == i) | (y_train == j))[0]
        x_pair = x_train[idx]
        y_pair = y_train[idx]

        # conversione in 0-1, altrimenti non viene accettato da perpetual
        y_pair_binary = np.where(y_pair == i, 0, 1)

        clf = NGBClassifier(n_estimators=n_estimators, random_state=42)
        clf.fit(x_pair, y_pair_binary)
        classifiers[(i, j)] = clf
    y_pred = ovo_predict(x_test)
    NGCacc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = get_confusion_matrix_elements(cm)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * ((precision * recall) / (precision + recall))
    print("------NGBoost------")
    print(f'Accuracy on the test set: {NGCacc:.4f}')
    print(classification_report(y_test, y_pred))
    writer.writerow([
        "NGBoost", round(NGCacc * 100, 2),
        round(mean(precision), 2), round(mean(recall), 2), round(mean(f1_score), 2),
    ])

    n_estimators = 900
    classifiers = {}
    for (i, j) in class_pairs:
        # seleziono il subset contenente gli elementi della coppia
        idx = np.where((y_train == i) | (y_train == j))[0]
        x_pair = x_train[idx]
        y_pair = y_train[idx]

        # conversione in 0-1, altrimenti non viene accettato da perpetual
        y_pair_binary = np.where(y_pair == i, 0, 1)

        clf = BoostingClassifier(
            n_estimators=n_estimators,
            base_estimator=DecisionTreeRegressor(max_depth=3),
            learning_rate=0.1
        )
        clf.fit(x_pair, y_pair_binary)
        classifiers[(i, j)] = clf
    y_pred = ovo_predict(x_test)
    STRacc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = get_confusion_matrix_elements(cm)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * ((precision * recall) / (precision + recall))
    print("------StarBoost------")
    print(f'Accuracy on the test set: {STRacc:.4f}')
    print(classification_report(y_test, y_pred))
    writer.writerow([
        "StarBoost", round(STRacc * 100, 2),
        round(mean(precision), 2), round(mean(recall), 2), round(mean(f1_score), 2),
    ])
