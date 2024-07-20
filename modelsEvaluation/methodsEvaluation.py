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
y = df['data_type']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

mms = MinMaxScaler()
mms.fit(x_train)
x_train = mms.transform(x_train)
x_test = mms.transform(x_test)

# TN = Benigno classificato come Benigno
# TP = Maligno classificato come Maligno
# FN = Maligno classificato come Benigno
# FP = Benigno classificato come Maligno

with open("models_results.csv", "a", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(['Name', 'Test Accuracy', 'Precision', 'Recall', 'F1-score', 'TN', 'FP', 'TP', 'FN', 'N Rounds',
                     'Execution time NoCV  (s)', 'Execution time  (s)', 'Round/s'])

    clf = PerpetualBooster(objective="LogLoss")
    start_time = time.time()
    clf.fit(x_train, y_train, budget=0.1)
    end_time = time.time()
    best_exec_time = end_time - start_time
    PPBpred = clf.predict(x_test)
    fpr, tpr, tresholds = roc_curve(y_test, PPBpred)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_treshold = tresholds[optimal_idx]
    y_pred = (PPBpred >= optimal_treshold).astype(int)
    PPBacc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * ((precision * recall) / (precision + recall))
    print("------PerpetualBooster------")
    print(f'Accuracy on the test set: {PPBacc:.4f}')
    print(classification_report(y_test, y_pred))
    writer.writerow([
        "PerpetualBooster", round(PPBacc * 100, 2),
        round(precision, 2), round(recall, 2), round(f1_score, 2),
        tn, fp, tp, fn,
        "",
        round(best_exec_time, 2),
        round(best_exec_time, 2),
        ""
    ])

    max_iter = 50
    clf = SGDClassifier()
    param_grid = {
        'max_iter': [max_iter],
        'random_state': [42],
        'alpha': [0.00001, 0.0001, 0.001],
        'l1_ratio': [0.0, 0.15, 0.40, 0.60, 0.85, 1.0]
    }
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='accuracy')
    start_time = time.time()
    grid_search.fit(x_train, y_train)
    end_time = time.time()
    exec_time = end_time - start_time
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(x_test)
    start_time = time.time()
    best_model.fit(x_train, y_train)
    end_time = time.time()
    best_exec_time = end_time - start_time
    SGCacc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * ((precision * recall) / (precision + recall))
    print("------StochasticGradientDescent------")
    print(f'Accuracy on the test set: {SGCacc:.4f}')
    print(classification_report(y_test, y_pred))
    writer.writerow([
        "StochasticGradientDescent", round(SGCacc * 100, 2),
        round(precision, 2), round(recall, 2), round(f1_score, 2),
        tn, fp, tp, fn,
        "",
        round(best_exec_time, 2),
        round(exec_time, 2),
        ""
    ])

    clf = CategoricalNB()
    start_time = time.time()
    clf.fit(x_train, y_train)
    end_time = time.time()
    best_exec_time = end_time - start_time
    y_pred = clf.predict(x_test)
    CANBacc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * ((precision * recall) / (precision + recall))
    print("------CategoricalNaiveBayes------")
    print(f'Accuracy on the test set: {CANBacc:.4f}')
    print(classification_report(y_test, y_pred))
    writer.writerow([
        "CategoricalNaiveBayes", round(CANBacc * 100, 2),
        round(precision, 2), round(recall, 2), round(f1_score, 2),
        tn, fp, tp, fn,
        "",
        round(best_exec_time, 2),
        round(best_exec_time, 2),
        ""
    ])

    clf = KNeighborsClassifier()
    param_grid = {
        'n_neighbors': [3, 5, 7],
        'algorithm': ['kd_tree'],
        'leaf_size': [20, 30, 40]
    }
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='accuracy')
    start_time = time.time()
    grid_search.fit(x_train, y_train)
    end_time = time.time()
    exec_time = end_time - start_time
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(x_test)
    start_time = time.time()
    best_model.fit(x_train, y_train)
    end_time = time.time()
    best_exec_time = end_time - start_time
    KNNacc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * ((precision * recall) / (precision + recall))
    print("------KNeighbors------")
    print(f'Accuracy on the test set: {KNNacc:.4f}')
    print(classification_report(y_test, y_pred))
    writer.writerow([
        "KNeighbors", round(KNNacc * 100, 2),
        round(precision, 2), round(recall, 2), round(f1_score, 2),
        tn, fp, tp, fn,
        "",
        round(best_exec_time, 2),
        round(exec_time, 2),
        ""
    ])

    clf = NearestCentroid()
    start_time = time.time()
    clf.fit(x_train, y_train)
    end_time = time.time()
    best_exec_time = end_time - start_time
    y_pred = clf.predict(x_test)
    NECacc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * ((precision * recall) / (precision + recall))
    print("------NearestCentroid------")
    print(f'Accuracy on the test set: {NECacc:.4f}')
    print(classification_report(y_test, y_pred))
    writer.writerow([
        "NearestCentroid", round(NECacc * 100, 2),
        round(precision, 2), round(recall, 2), round(f1_score, 2),
        tn, fp, tp, fn,
        "",
        round(best_exec_time, 2),
        round(best_exec_time, 2),
        ""
    ])

    max_iter = 50
    clf = svm.LinearSVC()
    param_grid = {
        'max_iter': [max_iter],
        'random_state': [42],
        'C': [0.5, 1.0, 2.0],
    }
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='accuracy')
    start_time = time.time()
    grid_search.fit(x_train, y_train)
    end_time = time.time()
    exec_time = end_time - start_time
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(x_test)
    start_time = time.time()
    best_model.fit(x_train, y_train)
    end_time = time.time()
    best_exec_time = end_time - start_time
    SVCacc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * ((precision * recall) / (precision + recall))
    print("------LinearSVC------")
    print(f'Accuracy on the test set: {SVCacc:.4f}')
    print(classification_report(y_test, y_pred))
    writer.writerow([
        "LinearSVC", round(SVCacc * 100, 2),
        round(precision, 2), round(recall, 2), round(f1_score, 2),
        tn, fp, tp, fn,
        max_iter,
        round(best_exec_time, 2),
        round(exec_time, 2),
        round(max_iter/best_exec_time, 4)
    ])

    clf = DecisionTreeClassifier()
    param_grid = {
        'random_state': [42],
        'min_samples_split': [2, 3],
        'min_samples_leaf': [1, 2, 3],
        'min_impurity_decrease': [0.0, 0.1, 0.2],
        'ccp_alpha': [0.0, 0.1, 0.2]
    }
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='accuracy')
    start_time = time.time()
    grid_search.fit(x_train, y_train)
    end_time = time.time()
    exec_time = end_time - start_time
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(x_test)
    start_time = time.time()
    best_model.fit(x_train, y_train)
    end_time = time.time()
    best_exec_time = end_time - start_time
    DTCacc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * ((precision * recall) / (precision + recall))
    print("------DecisionTree------")
    print(f'Accuracy on the test set: {DTCacc:.4f}')
    print(classification_report(y_test, y_pred))
    writer.writerow([
        "DecisionTree", round(DTCacc * 100, 2),
        round(precision, 2), round(recall, 2), round(f1_score, 2),
        tn, fp, tp, fn,
        "",
        round(best_exec_time, 2),
        round(exec_time, 2),
        ""
    ])

    n_estimators = 100
    clf = RandomForestClassifier()
    param_grid = {
        'n_estimators': [n_estimators],
        'random_state': [42],
        'min_samples_split': [2, 3],
        'min_samples_leaf': [1, 2],
        'min_impurity_decrease': [0.0, 0.1],
        'ccp_alpha': [0.0, 0.1]
    }
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='accuracy')
    start_time = time.time()
    grid_search.fit(x_train, y_train)
    end_time = time.time()
    exec_time = end_time - start_time
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(x_test)
    start_time = time.time()
    best_model.fit(x_train, y_train)
    end_time = time.time()
    best_exec_time = end_time - start_time
    RFCacc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * ((precision * recall) / (precision + recall))
    print("------RandomForest------")
    print(f'Accuracy on the test set: {RFCacc:.4f}')
    print(classification_report(y_test, y_pred))
    writer.writerow([
        "RandomForest", round(RFCacc * 100, 2),
        round(precision, 2), round(recall, 2), round(f1_score, 2),
        tn, fp, tp, fn,
        n_estimators,
        round(best_exec_time, 2),
        round(exec_time, 2),
        round(n_estimators/best_exec_time, 4)
    ])

    n_estimators = 50
    clf = ExtraTreesClassifier()
    param_grid = {
        'n_estimators': [n_estimators],
        'random_state': [42],
        'min_samples_split': [2, 3],
        'min_samples_leaf': [1, 2],
        'min_impurity_decrease': [0.0, 0.1],
        'ccp_alpha': [0.0, 0.1]
    }
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='accuracy')
    start_time = time.time()
    grid_search.fit(x_train, y_train)
    end_time = time.time()
    exec_time = end_time - start_time
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(x_test)
    start_time = time.time()
    best_model.fit(x_train, y_train)
    end_time = time.time()
    best_exec_time = end_time - start_time
    ETacc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * ((precision * recall) / (precision + recall))
    print("------ExtraTrees------")
    print(f'Accuracy on the test set: {ETacc:.4f}')
    print(classification_report(y_test, y_pred))
    writer.writerow([
        "ExtraTrees", round(ETacc * 100, 2),
        round(precision, 2), round(recall, 2), round(f1_score, 2),
        tn, fp, tp, fn,
        n_estimators,
        round(best_exec_time, 2),
        round(exec_time, 2),
        round(n_estimators/best_exec_time, 4)
    ])

    max_iter = 125
    clf = MLPClassifier()
    param_grid = {
        'max_iter': [max_iter],
        'random_state': [42],
        'hidden_layer_sizes': [50, 100, 200],
        'alpha': [0.00001, 0.0001, 0.001],
        'learning_rate_init': [0.0001, 0.001]
    }
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='accuracy')
    start_time = time.time()
    grid_search.fit(x_train, y_train)
    end_time = time.time()
    exec_time = end_time - start_time
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(x_test)
    start_time = time.time()
    best_model.fit(x_train, y_train)
    end_time = time.time()
    best_exec_time = end_time - start_time
    MLCacc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * ((precision * recall) / (precision + recall))
    print("------MultiLayerPerceptron------")
    print(f'Accuracy on the test set: {MLCacc:.4f}')
    print(classification_report(y_test, y_pred))
    writer.writerow([
        "MultiLayerPerceptron", round(MLCacc * 100, 2),
        round(precision, 2), round(recall, 2), round(f1_score, 2),
        tn, fp, tp, fn,
        max_iter,
        round(best_exec_time, 2),
        round(exec_time, 2),
        round(max_iter/best_exec_time, 4)
    ])

    n_estimators = 750
    clf = GradientBoostingClassifier()
    param_grid = {
        'n_estimators': [n_estimators],
        'random_state': [42],
        'learning_rate': [0.01, 0.1],
        'min_samples_split': [2, 3],
        'min_samples_leaf': [1, 2],
        'min_impurity_decrease': [0.0, 0.1],
    }
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='accuracy')
    start_time = time.time()
    grid_search.fit(x_train, y_train)
    end_time = time.time()
    exec_time = end_time - start_time
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(x_test)
    start_time = time.time()
    best_model.fit(x_train, y_train)
    end_time = time.time()
    best_exec_time = end_time - start_time
    GBCacc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * ((precision * recall) / (precision + recall))
    print("------GradientBoosting------")
    print(f'Accuracy on the test set: {GBCacc:.4f}')
    print(classification_report(y_test, y_pred))
    writer.writerow([
        "GradientBoosting", round(GBCacc * 100, 2),
        round(precision, 2), round(recall, 2), round(f1_score, 2),
        tn, fp, tp, fn,
        n_estimators,
        round(best_exec_time, 2),
        round(exec_time, 2),
        round(n_estimators/best_exec_time, 4)
    ])

    max_iter = 400
    clf = HistGradientBoostingClassifier()
    param_grid = {
        'max_iter': [max_iter],
        'random_state': [42],
        'learning_rate': [0.01, 0.1],
        'max_leaf_nodes': [21, 31, 41],
        'min_samples_leaf': [10, 20, 30],
        'max_bins': [50, 150, 255],
    }
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='accuracy')
    start_time = time.time()
    grid_search.fit(x_train, y_train)
    end_time = time.time()
    exec_time = end_time - start_time
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(x_test)
    start_time = time.time()
    best_model.fit(x_train, y_train)
    end_time = time.time()
    best_exec_time = end_time - start_time
    HGBCacc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * ((precision * recall) / (precision + recall))
    print("------HistGradientBoosting------")
    print(f'Accuracy on the test set: {HGBCacc:.4f}')
    print(classification_report(y_test, y_pred))
    writer.writerow([
        "HistGradientBoosting", round(HGBCacc * 100, 2),
        round(precision, 2), round(recall, 2), round(f1_score, 2),
        tn, fp, tp, fn,
        max_iter,
        round(best_exec_time, 2),
        round(exec_time, 2),
        round(max_iter/best_exec_time, 4)
    ])

    n_estimators = 500
    clf = AdaBoostClassifier()
    param_grid = {
        'n_estimators': [n_estimators],
        'random_state': [42],
        'learning_rate': [0.1, 1.0, 10.0],
    }
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='accuracy')
    start_time = time.time()
    grid_search.fit(x_train, y_train)
    end_time = time.time()
    exec_time = end_time - start_time
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(x_test)
    start_time = time.time()
    best_model.fit(x_train, y_train)
    end_time = time.time()
    best_exec_time = end_time - start_time
    ADACacc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * ((precision * recall) / (precision + recall))
    print("------AdaBoost------")
    print(f'Accuracy on the test set: {ADACacc:.4f}')
    print(classification_report(y_test, y_pred))
    writer.writerow([
        "AdaBoost", round(ADACacc * 100, 2),
        round(precision, 2), round(recall, 2), round(f1_score, 2),
        tn, fp, tp, fn,
        n_estimators,
        round(best_exec_time, 2),
        round(exec_time, 2),
        round(n_estimators/best_exec_time, 4)
    ])

    n_estimators = 350
    clf = XGBClassifier()
    param_grid = {
        'n_estimators': [n_estimators],
        'random_state': [42],
        'max_depth': [3, 6, 9],
        'max_bin': [128, 256, 512],
        'learning_rate': [0.1, 0.3, 0.5]
    }
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='accuracy')
    start_time = time.time()
    grid_search.fit(x_train, y_train)
    end_time = time.time()
    exec_time = end_time - start_time
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(x_test)
    start_time = time.time()
    best_model.fit(x_train, y_train)
    end_time = time.time()
    best_exec_time = end_time - start_time
    XGBacc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * ((precision * recall) / (precision + recall))
    print("------XGBoost------")
    print(f'Accuracy on the test set: {XGBacc:.4f}')
    print(classification_report(y_test, y_pred))
    writer.writerow([
        "XGBoost", round(XGBacc * 100, 2),
        round(precision, 2), round(recall, 2), round(f1_score, 2),
        tn, fp, tp, fn,
        n_estimators,
        round(best_exec_time, 2),
        round(exec_time, 2),
        round(n_estimators/best_exec_time, 4)
    ])

    n_estimators = 700
    clf = LGBMClassifier()
    param_grid = {
        'n_estimators': [n_estimators],
        'random_state': [42],
        'num_leaves': [21, 31, 41],
        'learning_rate': [0.01, 0.1, 1.0],
        'min_child_samples': [10, 20, 30]
    }
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='accuracy')
    start_time = time.time()
    grid_search.fit(x_train, y_train)
    end_time = time.time()
    exec_time = end_time - start_time
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(x_test)
    start_time = time.time()
    best_model.fit(x_train, y_train)
    end_time = time.time()
    best_exec_time = end_time - start_time
    LGCacc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * ((precision * recall) / (precision + recall))
    print("------LightGBM------")
    print(f'Accuracy on the test set: {LGCacc:.4f}')
    print(classification_report(y_test, y_pred))
    writer.writerow([
        "LightGBM", round(LGCacc * 100, 2),
        round(precision, 2), round(recall, 2), round(f1_score, 2),
        tn, fp, tp, fn,
        n_estimators,
        round(best_exec_time, 2),
        round(exec_time, 2),
        round(n_estimators/best_exec_time, 4)
    ])

    n_estimators = 250
    clf = CatBoostClassifier()
    param_grid = {
        'n_estimators': [n_estimators],
        'random_state': [42],
        'learning_rate': [0.01, 0.03],
        'depth': [4, 6, 8],
        'l2_leaf_reg': [1.0, 3.0, 5.0],
        'bagging_temperature': [0.33, 0.66, 1.0]
    }
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='accuracy')
    start_time = time.time()
    grid_search.fit(x_train, y_train)
    end_time = time.time()
    exec_time = end_time - start_time
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(x_test)
    start_time = time.time()
    best_model.fit(x_train, y_train)
    end_time = time.time()
    best_exec_time = end_time - start_time
    CBCacc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * ((precision * recall) / (precision + recall))
    print("------CatBoost------")
    print(f'Accuracy on the test set: {CBCacc:.4f}')
    print(classification_report(y_test, y_pred))
    writer.writerow([
        "CatBoost", round(CBCacc * 100, 2),
        round(precision, 2), round(recall, 2), round(f1_score, 2),
        tn, fp, tp, fn,
        n_estimators,
        round(best_exec_time, 2),
        round(exec_time, 2),
        round(n_estimators/best_exec_time, 4)
    ])

    n_estimators = 300
    clf = NGBClassifier()
    param_grid = {
        'n_estimators': [n_estimators],
        'random_state': [42],
        'learning_rate': [0.001, 0.01],
        'minibatch_frac': [0.5, 1.0],
        'col_sample': [0.5, 1.0]
    }
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='accuracy')
    start_time = time.time()
    grid_search.fit(x_train, y_train)
    end_time = time.time()
    exec_time = end_time - start_time
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(x_test)
    start_time = time.time()
    best_model.fit(x_train, y_train)
    end_time = time.time()
    best_exec_time = end_time - start_time
    NGCacc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * ((precision * recall) / (precision + recall))
    print("------NGBoost------")
    print(f'Accuracy on the test set: {NGCacc:.4f}')
    print(classification_report(y_test, y_pred))
    writer.writerow([
        "NGBoost", round(NGCacc * 100, 2),
        round(precision, 2), round(recall, 2), round(f1_score, 2),
        tn, fp, tp, fn,
        n_estimators,
        round(best_exec_time, 2),
        round(exec_time, 2),
        round(n_estimators/best_exec_time, 4)
    ])

    n_estimators = 900
    clf = BoostingClassifier(
        n_estimators=n_estimators,
        base_estimator=DecisionTreeRegressor(max_depth=3),
        learning_rate=0.1
    )
    start_time = time.time()
    clf.fit(x_train, y_train)
    end_time = time.time()
    best_exec_time = end_time - start_time
    y_pred = clf.predict(x_test)
    STRacc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * ((precision * recall) / (precision + recall))
    print("------StarBoost------")
    print(f'Accuracy on the test set: {STRacc:.4f}')
    print(classification_report(y_test, y_pred))
    writer.writerow([
        "StarBoost", round(STRacc * 100, 2),
        round(precision, 2), round(recall, 2), round(f1_score, 2),
        tn, fp, tp, fn,
        n_estimators,
        round(best_exec_time, 2),
        round(best_exec_time, 2),
        round(n_estimators/best_exec_time, 4)
    ])
