import numpy as np
import pandas as pd
import csv
import time
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from ngboost import NGBClassifier, NGBRegressor
from sklearn import svm
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, GradientBoostingRegressor, \
    HistGradientBoostingClassifier, HistGradientBoostingRegressor, AdaBoostClassifier, AdaBoostRegressor, \
    RandomForestClassifier, RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Lasso, ElasticNet, LassoLars, \
    OrthogonalMatchingPursuit, SGDClassifier, SGDRegressor
from sklearn.metrics import accuracy_score, classification_report, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, ComplementNB, CategoricalNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestCentroid
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from perpetual import PerpetualBooster
from starboost import BoostingRegressor
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

with open("models_results.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(['Name', 'Train Accuracy', 'Test Accuracy', 'Precision', 'Recall', 'F1-score', 'N Rounds',
                     'Execution time'])

    clf = PerpetualBooster(objective="LogLoss")
    start_time = time.time()
    clf.fit(x_train, y_train, budget=0.1)
    end_time = time.time()
    exec_time = end_time - start_time
    PPB_tpred = clf.predict(x_train)
    fpr, tpr, tresholds = roc_curve(y_train, PPB_tpred)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_treshold = tresholds[optimal_idx]
    y_pred = (PPB_tpred >= optimal_treshold).astype(int)
    PPB_tacc = accuracy_score(y_train, y_pred)
    PPBpred = clf.predict(x_test)
    fpr, tpr, tresholds = roc_curve(y_test, PPBpred)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_treshold = tresholds[optimal_idx]
    y_pred = (PPBpred >= optimal_treshold).astype(int)
    PPBacc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    print("------PerpetualBoosterClassifier------")
    print(f'Accuracy on the train set: {PPB_tacc:.4f}')
    print(f'Accuracy on the test set: {PPBacc:.4f}')
    print(classification_report(y_test, y_pred))
    writer.writerow([
        "PerpetualBoosterClassifier", round(PPB_tacc * 100, 2), round(PPBacc * 100, 2),
        round(report['weighted avg']['precision'] * 100, 2),
        round(report['weighted avg']['recall'] * 100, 2),
        round(report['weighted avg']['f1-score'] * 100, 2),
        "",
        round(exec_time, 2)
    ])

    clf = PerpetualBooster(objective="SquaredLoss")
    start_time = time.time()
    clf.fit(x_train, y_train, budget=0.1)
    end_time = time.time()
    exec_time = end_time - start_time
    PPB_tpred = clf.predict(x_train)
    fpr, tpr, tresholds = roc_curve(y_train, PPB_tpred)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_treshold = tresholds[optimal_idx]
    y_pred = (PPB_tpred >= optimal_treshold).astype(int)
    PPB_tacc = accuracy_score(y_train, y_pred)
    PPBpred = clf.predict(x_test)
    fpr, tpr, tresholds = roc_curve(y_test, PPBpred)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_treshold = tresholds[optimal_idx]
    y_pred = (PPBpred >= optimal_treshold).astype(int)
    PPBacc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    print("------PerpetualBoosterRegressor------")
    print(f'Accuracy on the train set: {PPB_tacc:.4f}')
    print(f'Accuracy on the test set: {PPBacc:.4f}')
    print(classification_report(y_test, y_pred))
    writer.writerow([
        "PerpetualBoosterRegressor", round(PPB_tacc * 100, 2), round(PPBacc * 100, 2),
        round(report['weighted avg']['precision'] * 100, 2),
        round(report['weighted avg']['recall'] * 100, 2),
        round(report['weighted avg']['f1-score'] * 100, 2),
        "",
        round(exec_time, 2)
    ])

    max_iter = 50
    clf = RidgeClassifier(max_iter=max_iter)
    start_time = time.time()
    clf.fit(x_train, y_train)
    end_time = time.time()
    exec_time = end_time - start_time
    RIC_tpred = clf.predict(x_train)
    RIC_tacc = accuracy_score(y_train, RIC_tpred)
    y_pred = clf.predict(x_test)
    RICacc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    print("------RidgeClassifier------")
    print(f'Accuracy on the train set: {RIC_tacc:.4f}')
    print(f'Accuracy on the test set: {RICacc:.4f}')
    print(classification_report(y_test, y_pred, zero_division=0))
    writer.writerow([
        "Ridge", round(RIC_tacc * 100, 2), round(RICacc * 100, 2),
        round(report['weighted avg']['precision'] * 100, 2),
        round(report['weighted avg']['recall'] * 100, 2),
        round(report['weighted avg']['f1-score'] * 100, 2),
        max_iter,
        round(exec_time, 2)
    ])

    max_iter = 50
    clf = Lasso(max_iter=max_iter)
    start_time = time.time()
    clf.fit(x_train, y_train)
    end_time = time.time()
    exec_time = end_time - start_time
    LAS_tpred = clf.predict(x_train)
    fpr, tpr, tresholds = roc_curve(y_train, LAS_tpred)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_treshold = tresholds[optimal_idx]
    y_pred = (LAS_tpred >= optimal_treshold).astype(int)
    LAS_tacc = accuracy_score(y_train, y_pred)
    LASpred = clf.predict(x_test)
    fpr, tpr, tresholds = roc_curve(y_test, LASpred)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_treshold = tresholds[optimal_idx]
    y_pred = (LASpred >= optimal_treshold).astype(int)
    LASacc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    print("------Lasso------")
    print(f'Accuracy on the train set: {LAS_tacc:.4f}')
    print(f'Accuracy on the test set: {LASacc:.4f}')
    print(classification_report(y_test, y_pred, zero_division=0))
    writer.writerow([
        "Lasso", round(LAS_tacc * 100, 2), round(LASacc * 100, 2),
        round(report['weighted avg']['precision'] * 100, 2),
        round(report['weighted avg']['recall'] * 100, 2),
        round(report['weighted avg']['f1-score'] * 100, 2),
        max_iter,
        round(exec_time, 2)
    ])

    max_iter = 50
    clf = ElasticNet(max_iter=max_iter)
    start_time = time.time()
    clf.fit(x_train, y_train)
    end_time = time.time()
    exec_time = end_time - start_time
    ELN_tpred = clf.predict(x_train)
    fpr, tpr, tresholds = roc_curve(y_train, ELN_tpred)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_treshold = tresholds[optimal_idx]
    y_pred = (ELN_tpred >= optimal_treshold).astype(int)
    ELN_tacc = accuracy_score(y_train, y_pred)
    ELNpred = clf.predict(x_test)
    fpr, tpr, tresholds = roc_curve(y_test, ELNpred)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_treshold = tresholds[optimal_idx]
    y_pred = (ELNpred >= optimal_treshold).astype(int)
    ELNacc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    print("------ElasticNet------")
    print(f'Accuracy on the train set: {ELN_tacc:.4f}')
    print(f'Accuracy on the test set: {ELNacc:.4f}')
    print(classification_report(y_test, y_pred, zero_division=0))
    writer.writerow([
        "ElasticNet", round(ELN_tacc * 100, 2), round(ELNacc * 100, 2),
        round(report['weighted avg']['precision'] * 100, 2),
        round(report['weighted avg']['recall'] * 100, 2),
        round(report['weighted avg']['f1-score'] * 100, 2),
        max_iter,
        round(exec_time, 2)
    ])

    clf = OrthogonalMatchingPursuit()
    start_time = time.time()
    clf.fit(x_train, y_train)
    end_time = time.time()
    exec_time = end_time - start_time
    OMP_tpred = clf.predict(x_train)
    fpr, tpr, tresholds = roc_curve(y_train, OMP_tpred)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_treshold = tresholds[optimal_idx]
    y_pred = (OMP_tpred >= optimal_treshold).astype(int)
    OMP_tacc = accuracy_score(y_train, y_pred)
    OMPpred = clf.predict(x_test)
    fpr, tpr, tresholds = roc_curve(y_test, OMPpred)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_treshold = tresholds[optimal_idx]
    y_pred = (OMPpred >= optimal_treshold).astype(int)
    OMPacc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    print("------OrthogonalMatchingPursuit------")
    print(f'Accuracy on the train set: {OMP_tacc:.4f}')
    print(f'Accuracy on the test set: {OMPacc:.4f}')
    print(classification_report(y_test, y_pred, zero_division=0))
    writer.writerow([
        "OrthogonalMatchingPursuit", round(OMP_tacc * 100, 2), round(OMPacc * 100, 2),
        round(report['weighted avg']['precision'] * 100, 2),
        round(report['weighted avg']['recall'] * 100, 2),
        round(report['weighted avg']['f1-score'] * 100, 2),
        "",
        round(exec_time, 2)
    ])

    max_iter = 50
    clf = SGDClassifier(max_iter=max_iter)
    start_time = time.time()
    clf.fit(x_train, y_train)
    end_time = time.time()
    exec_time = end_time - start_time
    SGC_tpred = clf.predict(x_train)
    SGC_tacc = accuracy_score(y_train, SGC_tpred)
    y_pred = clf.predict(x_test)
    SGCacc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    print("------SGDClassifier------")
    print(f'Accuracy on the train set: {SGC_tacc:.4f}')
    print(f'Accuracy on the test set: {SGCacc:.4f}')
    print(classification_report(y_test, y_pred))
    writer.writerow([
        "SGDClassifier", round(SGC_tacc * 100, 2), round(SGCacc * 100, 2),
        round(report['weighted avg']['precision'] * 100, 2),
        round(report['weighted avg']['recall'] * 100, 2),
        round(report['weighted avg']['f1-score'] * 100, 2),
        "",
        round(exec_time, 2)
    ])

    clf = SGDRegressor(max_iter=max_iter)
    start_time = time.time()
    clf.fit(x_train, y_train)
    end_time = time.time()
    exec_time = end_time - start_time
    SGR_tpred = clf.predict(x_train)
    fpr, tpr, tresholds = roc_curve(y_train, SGR_tpred)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_treshold = tresholds[optimal_idx]
    y_pred = (SGR_tpred >= optimal_treshold).astype(int)
    SGR_tacc = accuracy_score(y_train, y_pred)
    SGRpred = clf.predict(x_test)
    fpr, tpr, tresholds = roc_curve(y_test, SGRpred)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_treshold = tresholds[optimal_idx]
    y_pred = (SGRpred >= optimal_treshold).astype(int)
    SGRacc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    print("------SGDRegressor------")
    print(f'Accuracy on the train set: {SGR_tacc:.4f}')
    print(f'Accuracy on the test set: {SGRacc:.4f}')
    print(classification_report(y_test, y_pred))
    writer.writerow([
        "SGDRegressor", round(SGR_tacc * 100, 2), round(SGRacc * 100, 2),
        round(report['weighted avg']['precision'] * 100, 2),
        round(report['weighted avg']['recall'] * 100, 2),
        round(report['weighted avg']['f1-score'] * 100, 2),
        "",
        round(exec_time, 2)
    ])

    clf = GaussianNB()
    start_time = time.time()
    clf.fit(x_train, y_train)
    end_time = time.time()
    exec_time = end_time - start_time
    GNB_tpred = clf.predict(x_train)
    GNB_tacc = accuracy_score(y_train, GNB_tpred)
    y_pred = clf.predict(x_test)
    GNBacc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    print("------GaussianNaiveBayes------")
    print(f'Accuracy on the train set: {GNB_tacc:.4f}')
    print(f'Accuracy on the test set: {GNBacc:.4f}')
    print(classification_report(y_test, y_pred))
    writer.writerow([
        "GaussianNaiveBayes", round(GNB_tacc * 100, 2), round(GNBacc * 100, 2),
        round(report['weighted avg']['precision'] * 100, 2),
        round(report['weighted avg']['recall'] * 100, 2),
        round(report['weighted avg']['f1-score'] * 100, 2),
        "",
        round(exec_time, 2)
    ])

    clf = ComplementNB()
    start_time = time.time()
    clf.fit(x_train, y_train)
    end_time = time.time()
    exec_time = end_time - start_time
    CNB_tpred = clf.predict(x_train)
    CNB_tacc = accuracy_score(y_train, CNB_tpred)
    y_pred = clf.predict(x_test)
    CNBacc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    print("------ComplementNaiveBayes------")
    print(f'Accuracy on the train set: {CNB_tacc:.4f}')
    print(f'Accuracy on the test set: {CNBacc:.4f}')
    print(classification_report(y_test, y_pred))
    writer.writerow([
        "ComplementNaiveBayes", round(CNB_tacc * 100, 2), round(CNBacc * 100, 2),
        round(report['weighted avg']['precision'] * 100, 2),
        round(report['weighted avg']['recall'] * 100, 2),
        round(report['weighted avg']['f1-score'] * 100, 2),
        "",
        round(exec_time, 2)
    ])

    clf = CategoricalNB()
    start_time = time.time()
    clf.fit(x_train, y_train)
    end_time = time.time()
    exec_time = end_time - start_time
    CANB_tpred = clf.predict(x_train)
    CANB_tacc = accuracy_score(y_train, CANB_tpred)
    y_pred = clf.predict(x_test)
    CANBacc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    print("------CategoricalNaiveBayes------")
    print(f'Accuracy on the train set: {CANB_tacc:.4f}')
    print(f'Accuracy on the test set: {CANBacc:.4f}')
    print(classification_report(y_test, y_pred))
    writer.writerow([
        "CategoricalNaiveBayes", round(CANB_tacc * 100, 2), round(CANBacc * 100, 2),
        round(report['weighted avg']['precision'] * 100, 2),
        round(report['weighted avg']['recall'] * 100, 2),
        round(report['weighted avg']['f1-score'] * 100, 2),
        "",
        round(exec_time, 2)
    ])

    clf = KNeighborsClassifier(n_neighbors=1, algorithm='kd_tree', n_jobs=-1)
    start_time = time.time()
    clf.fit(x_train, y_train)
    end_time = time.time()
    exec_time = end_time - start_time
    KNN_tpred = clf.predict(x_train)
    KNN_tacc = accuracy_score(y_train, KNN_tpred)
    y_pred = clf.predict(x_test)
    KNNacc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    print("------KNN------")
    print(f'Accuracy on the train set: {KNN_tacc:.4f}')
    print(f'Accuracy on the test set: {KNNacc:.4f}')
    print(classification_report(y_test, y_pred))
    writer.writerow([
        "KNeighbors", round(KNN_tacc * 100, 2), round(KNNacc * 100, 2),
        round(report['weighted avg']['precision'] * 100, 2),
        round(report['weighted avg']['recall'] * 100, 2),
        round(report['weighted avg']['f1-score'] * 100, 2),
        "",
        round(exec_time, 2)
    ])

    clf = NearestCentroid()
    start_time = time.time()
    clf.fit(x_train, y_train)
    end_time = time.time()
    exec_time = end_time - start_time
    NEC_tpred = clf.predict(x_train)
    NEC_tacc = accuracy_score(y_train, NEC_tpred)
    y_pred = clf.predict(x_test)
    NECacc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    print("------NearestCentroid------")
    print(f'Accuracy on the train set: {NEC_tacc:.4f}')
    print(f'Accuracy on the test set: {NECacc:.4f}')
    print(classification_report(y_test, y_pred))
    writer.writerow([
        "NearestCentroid", round(NEC_tacc * 100, 2), round(NECacc * 100, 2),
        round(report['weighted avg']['precision'] * 100, 2),
        round(report['weighted avg']['recall'] * 100, 2),
        round(report['weighted avg']['f1-score'] * 100, 2),
        "",
        round(exec_time, 2)
    ])

    max_iter = 50
    clf = svm.LinearSVC(max_iter=max_iter)
    start_time = time.time()
    clf.fit(x_train, y_train)
    end_time = time.time()
    exec_time = end_time - start_time
    SVC_tpred = clf.predict(x_train)
    SVC_tacc = accuracy_score(y_train, SVC_tpred)
    y_pred = clf.predict(x_test)
    SVCacc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    print("------SVC------")
    print(f'Accuracy on the train set: {SVC_tacc:.4f}')
    print(f'Accuracy on the test set: {SVCacc:.4f}')
    print(classification_report(y_test, y_pred))
    writer.writerow([
        "LinearSVC", round(SVC_tacc * 100, 2), round(SVCacc * 100, 2),
        round(report['weighted avg']['precision'] * 100, 2),
        round(report['weighted avg']['recall'] * 100, 2),
        round(report['weighted avg']['f1-score'] * 100, 2),
        max_iter,
        round(exec_time, 2)
    ])

    max_iter = 100
    clf = LogisticRegression(max_iter=max_iter, n_jobs=-1)
    start_time = time.time()
    clf.fit(x_train, y_train)
    end_time = time.time()
    exec_time = end_time - start_time
    LGR_tpred = clf.predict(x_train)
    LGR_tacc = accuracy_score(y_train, LGR_tpred)
    y_pred = clf.predict(x_test)
    LGRacc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    print("------LogisticRegression------")
    print(f'Accuracy on the train set: {LGR_tacc:.4f}')
    print(f'Accuracy on the test set: {LGRacc:.4f}')
    print(classification_report(y_test, y_pred))
    writer.writerow([
        "LogisticRegression", round(LGR_tacc * 100, 2), round(LGRacc * 100, 2),
        round(report['weighted avg']['precision'] * 100, 2),
        round(report['weighted avg']['recall'] * 100, 2),
        round(report['weighted avg']['f1-score'] * 100, 2),
        max_iter,
        round(exec_time, 2)
    ])

    clf = DecisionTreeClassifier()
    start_time = time.time()
    clf.fit(x_train, y_train)
    end_time = time.time()
    exec_time = end_time - start_time
    DTC_tpred = clf.predict(x_train)
    DTC_tacc = accuracy_score(y_train, DTC_tpred)
    y_pred = clf.predict(x_test)
    DTCacc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    print("------DecisionTree------")
    print(f'Accuracy on the train set: {DTC_tacc:.4f}')
    print(f'Accuracy on the test set: {DTCacc:.4f}')
    print(classification_report(y_test, y_pred))
    writer.writerow([
        "DecisionTree", round(DTC_tacc * 100, 2), round(DTCacc * 100, 2),
        round(report['weighted avg']['precision'] * 100, 2),
        round(report['weighted avg']['recall'] * 100, 2),
        round(report['weighted avg']['f1-score'] * 100, 2),
        "",
        round(exec_time, 2)
    ])

    n_estimators = 100
    clf = RandomForestClassifier(n_estimators=n_estimators)
    start_time = time.time()
    clf.fit(x_train, y_train)
    end_time = time.time()
    exec_time = end_time - start_time
    RFC_tpred = clf.predict(x_train)
    RFC_tacc = accuracy_score(y_train, RFC_tpred)
    y_pred = clf.predict(x_test)
    RFCacc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    print("------RandomForestClassifier------")
    print(f'Accuracy on the train set: {RFC_tacc:.4f}')
    print(f'Accuracy on the test set: {RFCacc:.4f}')
    print(classification_report(y_test, y_pred))
    writer.writerow([
        "RandomForestClassifier", round(RFC_tacc * 100, 2), round(RFCacc * 100, 2),
        round(report['weighted avg']['precision'] * 100, 2),
        round(report['weighted avg']['recall'] * 100, 2),
        round(report['weighted avg']['f1-score'] * 100, 2),
        n_estimators,
        round(exec_time, 2)
    ])

    clf = RandomForestRegressor(n_estimators=n_estimators)
    start_time = time.time()
    clf.fit(x_train, y_train)
    end_time = time.time()
    exec_time = end_time - start_time
    RFR_tpred = clf.predict(x_train)
    fpr, tpr, tresholds = roc_curve(y_train, RFR_tpred)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_treshold = tresholds[optimal_idx]
    y_pred = (RFR_tpred >= optimal_treshold).astype(int)
    RFR_tacc = accuracy_score(y_train, y_pred)
    RFRpred = clf.predict(x_test)
    fpr, tpr, tresholds = roc_curve(y_test, RFRpred)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_treshold = tresholds[optimal_idx]
    y_pred = (RFRpred >= optimal_treshold).astype(int)
    RFRacc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    print("------RandomForestRegressor------")
    print(f'Accuracy on the train set: {RFR_tacc:.4f}')
    print(f'Accuracy on the test set: {RFRacc:.4f}')
    print(classification_report(y_test, y_pred))
    writer.writerow([
        "RandomForestRegressor", round(RFR_tacc * 100, 2), round(RFRacc * 100, 2),
        round(report['weighted avg']['precision'] * 100, 2),
        round(report['weighted avg']['recall'] * 100, 2),
        round(report['weighted avg']['f1-score'] * 100, 2),
        n_estimators,
        round(exec_time, 2)
    ])

    n_estimators = 50
    clf = ExtraTreesClassifier(n_estimators=n_estimators)
    start_time = time.time()
    clf.fit(x_train, y_train)
    end_time = time.time()
    exec_time = end_time - start_time
    ET_tpred = clf.predict(x_train)
    ET_tacc = accuracy_score(y_train, ET_tpred)
    y_pred = clf.predict(x_test)
    ETacc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    print("------ExtraTrees------")
    print(f'Accuracy on the train set: {ET_tacc:.4f}')
    print(f'Accuracy on the test set: {ETacc:.4f}')
    print(classification_report(y_test, y_pred))
    writer.writerow([
        "ExtraTreesClassifier", round(ET_tacc * 100, 2), round(ETacc * 100, 2),
        round(report['weighted avg']['precision'] * 100, 2),
        round(report['weighted avg']['recall'] * 100, 2),
        round(report['weighted avg']['f1-score'] * 100, 2),
        n_estimators,
        round(exec_time, 2)
    ])

    clf = ExtraTreesRegressor(n_estimators=n_estimators)
    start_time = time.time()
    clf.fit(x_train, y_train)
    end_time = time.time()
    exec_time = end_time - start_time
    ETR_tpred = clf.predict(x_train)
    fpr, tpr, tresholds = roc_curve(y_train, ETR_tpred)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_treshold = tresholds[optimal_idx]
    y_pred = (ETR_tpred >= optimal_treshold).astype(int)
    ETR_tacc = accuracy_score(y_train, y_pred)
    ETRpred = clf.predict(x_test)
    fpr, tpr, tresholds = roc_curve(y_test, ETRpred)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_treshold = tresholds[optimal_idx]
    y_pred = (ETRpred >= optimal_treshold).astype(int)
    ETRacc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    print("------ExtraTreesRegressor------")
    print(f'Accuracy on the train set: {ETR_tacc:.4f}')
    print(f'Accuracy on the test set: {ETRacc:.4f}')
    print(classification_report(y_test, y_pred))
    writer.writerow([
        "ExtraTreesRegressor", round(ETR_tacc * 100, 2), round(ETRacc * 100, 2),
        round(report['weighted avg']['precision'] * 100, 2),
        round(report['weighted avg']['recall'] * 100, 2),
        round(report['weighted avg']['f1-score'] * 100, 2),
        n_estimators,
        round(exec_time, 2)
    ])

    max_iter = 125
    clf = MLPClassifier(max_iter=max_iter)
    start_time = time.time()
    clf.fit(x_train, y_train)
    end_time = time.time()
    exec_time = end_time - start_time
    MLC_tpred = clf.predict(x_train)
    MLC_tacc = accuracy_score(y_train, MLC_tpred)
    y_pred = clf.predict(x_test)
    MLCacc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    print("------MLPClassifier------")
    print(f'Accuracy on the train set: {MLC_tacc:.4f}')
    print(f'Accuracy on the test set: {MLCacc:.4f}')
    print(classification_report(y_test, y_pred))
    writer.writerow([
        "MLPClassifier", round(MLC_tacc * 100, 2), round(MLCacc * 100, 2),
        round(report['weighted avg']['precision'] * 100, 2),
        round(report['weighted avg']['recall'] * 100, 2),
        round(report['weighted avg']['f1-score'] * 100, 2),
        max_iter,
        round(exec_time, 2)
    ])

    clf = MLPRegressor(max_iter=max_iter)
    start_time = time.time()
    clf.fit(x_train, y_train)
    end_time = time.time()
    exec_time = end_time - start_time
    MLR_tpred = clf.predict(x_train)
    fpr, tpr, tresholds = roc_curve(y_train, MLR_tpred)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_treshold = tresholds[optimal_idx]
    y_pred = (MLR_tpred >= optimal_treshold).astype(int)
    MLR_tacc = accuracy_score(y_train, y_pred)
    MLRpred = clf.predict(x_test)
    fpr, tpr, tresholds = roc_curve(y_test, MLRpred)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_treshold = tresholds[optimal_idx]
    y_pred = (MLRpred >= optimal_treshold).astype(int)
    MLRacc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    print("------MLPRegressor------")
    print(f'Accuracy on the train set: {MLR_tacc:.4f}')
    print(f'Accuracy on the test set: {MLRacc:.4f}')
    print(classification_report(y_test, y_pred))
    writer.writerow([
        "MLPRegressor", round(MLR_tacc * 100, 2), round(MLRacc * 100, 2),
        round(report['weighted avg']['precision'] * 100, 2),
        round(report['weighted avg']['recall'] * 100, 2),
        round(report['weighted avg']['f1-score'] * 100, 2),
        max_iter,
        round(exec_time, 2)
    ])

    n_estimators = 750
    clf = GradientBoostingClassifier(n_estimators=n_estimators)
    start_time = time.time()
    clf.fit(x_train, y_train)
    end_time = time.time()
    exec_time = end_time - start_time
    GBC_tpred = clf.predict(x_train)
    GBC_tacc = accuracy_score(y_train, GBC_tpred)
    y_pred = clf.predict(x_test)
    GBCacc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    print("------GradientBoostingClassifier------")
    print(f'Accuracy on the train set: {GBC_tacc:.4f}')
    print(f'Accuracy on the test set: {GBCacc:.4f}')
    print(classification_report(y_test, y_pred))
    writer.writerow([
        "GradientBoostingClassifier", round(GBC_tacc * 100, 2), round(GBCacc * 100, 2),
        round(report['weighted avg']['precision'] * 100, 2),
        round(report['weighted avg']['recall'] * 100, 2),
        round(report['weighted avg']['f1-score'] * 100, 2),
        n_estimators,
        round(exec_time, 2)
    ])

    clf = GradientBoostingRegressor(n_estimators=n_estimators)
    start_time = time.time()
    clf.fit(x_train, y_train)
    end_time = time.time()
    exec_time = end_time - start_time
    GBR_tpred = clf.predict(x_train)
    fpr, tpr, tresholds = roc_curve(y_train, GBR_tpred)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_treshold = tresholds[optimal_idx]
    y_pred = (GBR_tpred >= optimal_treshold).astype(int)
    GBR_tacc = accuracy_score(y_train, y_pred)
    GBRpred = clf.predict(x_test)
    fpr, tpr, tresholds = roc_curve(y_test, GBRpred)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_treshold = tresholds[optimal_idx]
    y_pred = (GBRpred >= optimal_treshold).astype(int)
    GBRacc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    print("------GradientBoostingRegressor------")
    print(f'Accuracy on the train set: {GBR_tacc:.4f}')
    print(f'Accuracy on the test set: {GBRacc:.4f}')
    print(classification_report(y_test, y_pred))
    writer.writerow([
        "GradientBoostingRegressor", round(GBR_tacc * 100, 2), round(GBRacc * 100, 2),
        round(report['weighted avg']['precision'] * 100, 2),
        round(report['weighted avg']['recall'] * 100, 2),
        round(report['weighted avg']['f1-score'] * 100, 2),
        n_estimators,
        round(exec_time, 2)
    ])

    max_iter = 400
    clf = HistGradientBoostingClassifier(max_iter=max_iter)
    start_time = time.time()
    clf.fit(x_train, y_train)
    end_time = time.time()
    exec_time = end_time - start_time
    HGBC_tpred = clf.predict(x_train)
    HGBC_tacc = accuracy_score(y_train, HGBC_tpred)
    y_pred = clf.predict(x_test)
    HGBCacc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    print("------HistGradientBoostingClassifier------")
    print(f'Accuracy on the train set: {HGBC_tacc:.4f}')
    print(f'Accuracy on the test set: {HGBCacc:.4f}')
    print(classification_report(y_test, y_pred))
    writer.writerow([
        "HistGradientBoostingClassifier", round(HGBC_tacc * 100, 2), round(HGBCacc * 100, 2),
        round(report['weighted avg']['precision'] * 100, 2),
        round(report['weighted avg']['recall'] * 100, 2),
        round(report['weighted avg']['f1-score'] * 100, 2),
        max_iter,
        round(exec_time, 2)
    ])

    clf = HistGradientBoostingRegressor(max_iter=max_iter)
    start_time = time.time()
    clf.fit(x_train, y_train)
    end_time = time.time()
    exec_time = end_time - start_time
    HGBR_tpred = clf.predict(x_train)
    fpr, tpr, tresholds = roc_curve(y_train, HGBR_tpred)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_treshold = tresholds[optimal_idx]
    y_pred = (HGBR_tpred >= optimal_treshold).astype(int)
    HGBR_tacc = accuracy_score(y_train, y_pred)
    HGBRpred = clf.predict(x_test)
    fpr, tpr, tresholds = roc_curve(y_test, HGBRpred)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_treshold = tresholds[optimal_idx]
    y_pred = (HGBRpred >= optimal_treshold).astype(int)
    HGBRacc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    print("------HistGradientBoostingRegressor------")
    print(f'Accuracy on the train set: {HGBR_tacc:.4f}')
    print(f'Accuracy on the test set: {HGBRacc:.4f}')
    print(classification_report(y_test, y_pred))
    writer.writerow([
        "HistGradientBoostingRegressor", round(HGBR_tacc * 100, 2), round(HGBRacc * 100, 2),
        round(report['weighted avg']['precision'] * 100, 2),
        round(report['weighted avg']['recall'] * 100, 2),
        round(report['weighted avg']['f1-score'] * 100, 2),
        max_iter,
        round(exec_time, 2)
    ])

    n_estimators = 500
    clf = AdaBoostClassifier(n_estimators=n_estimators)
    start_time = time.time()
    clf.fit(x_train, y_train)
    end_time = time.time()
    exec_time = end_time - start_time
    ADAC_tpred = clf.predict(x_train)
    ADAC_tacc = accuracy_score(y_train, ADAC_tpred)
    y_pred = clf.predict(x_test)
    ADACacc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    print("------AdaBoostClassifier------")
    print(f'Accuracy on the train set: {ADAC_tacc:.4f}')
    print(f'Accuracy on the test set: {ADACacc:.4f}')
    print(classification_report(y_test, y_pred))
    writer.writerow([
        "AdaBoostClassifier", round(ADAC_tacc * 100, 2), round(ADACacc * 100, 2),
        round(report['weighted avg']['precision'] * 100, 2),
        round(report['weighted avg']['recall'] * 100, 2),
        round(report['weighted avg']['f1-score'] * 100, 2),
        n_estimators,
        round(exec_time, 2)
    ])

    clf = AdaBoostRegressor(n_estimators=n_estimators)
    start_time = time.time()
    clf.fit(x_train, y_train)
    end_time = time.time()
    exec_time = end_time - start_time
    ADAR_tpred = clf.predict(x_train)
    fpr, tpr, tresholds = roc_curve(y_train, ADAR_tpred)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_treshold = tresholds[optimal_idx]
    y_pred = (ADAR_tpred >= optimal_treshold).astype(int)
    ADAR_tacc = accuracy_score(y_train, y_pred)
    ADARpred = clf.predict(x_test)
    fpr, tpr, tresholds = roc_curve(y_test, ADARpred)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_treshold = tresholds[optimal_idx]
    y_pred = (ADARpred >= optimal_treshold).astype(int)
    ADARacc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    print("------AdaBoostRegressor------")
    print(f'Accuracy on the train set: {ADAR_tacc:.4f}')
    print(f'Accuracy on the test set: {ADARacc:.4f}')
    print(classification_report(y_test, y_pred))
    writer.writerow([
        "AdaBoostRegressor", round(ADAR_tacc * 100, 2), round(ADARacc * 100, 2),
        round(report['weighted avg']['precision'] * 100, 2),
        round(report['weighted avg']['recall'] * 100, 2),
        round(report['weighted avg']['f1-score'] * 100, 2),
        n_estimators,
        round(exec_time, 2)
    ])

    n_estimators = 350
    clf = XGBClassifier(n_estimators=n_estimators)
    start_time = time.time()
    clf.fit(x_train, y_train)
    end_time = time.time()
    exec_time = end_time - start_time
    XGB_tpred = clf.predict(x_train)
    XGB_tacc = accuracy_score(y_train, XGB_tpred)
    y_pred = clf.predict(x_test)
    XGBacc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    print("------XGBoost------")
    print(f'Accuracy on the train set: {XGB_tacc:.4f}')
    print(f'Accuracy on the test set: {XGBacc:.4f}')
    print(classification_report(y_test, y_pred))
    writer.writerow([
        "XGBoostClassifier", round(XGB_tacc * 100, 2), round(XGBacc * 100, 2),
        round(report['weighted avg']['precision'] * 100, 2),
        round(report['weighted avg']['recall'] * 100, 2),
        round(report['weighted avg']['f1-score'] * 100, 2),
        n_estimators,
        round(exec_time, 2)
    ])

    clf = XGBRegressor(n_estimators=n_estimators)
    start_time = time.time()
    clf.fit(x_train, y_train)
    end_time = time.time()
    exec_time = end_time - start_time
    XBR_tpred = clf.predict(x_train)
    fpr, tpr, tresholds = roc_curve(y_train, XBR_tpred)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_treshold = tresholds[optimal_idx]
    y_pred = (XBR_tpred >= optimal_treshold).astype(int)
    XBR_tacc = accuracy_score(y_train, y_pred)
    XBRpred = clf.predict(x_test)
    fpr, tpr, tresholds = roc_curve(y_test, XBRpred)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_treshold = tresholds[optimal_idx]
    y_pred = (XBRpred >= optimal_treshold).astype(int)
    XBRacc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    print("------XGBoostRegressor------")
    print(f'Accuracy on the train set: {XBR_tacc:.4f}')
    print(f'Accuracy on the test set: {XBRacc:.4f}')
    print(classification_report(y_test, y_pred))
    writer.writerow([
        "XGBoostRegressor", round(XBR_tacc * 100, 2), round(XBRacc * 100, 2),
        round(report['weighted avg']['precision'] * 100, 2),
        round(report['weighted avg']['recall'] * 100, 2),
        round(report['weighted avg']['f1-score'] * 100, 2),
        n_estimators,
        round(exec_time, 2)
    ])

    n_estimators = 700
    clf = LGBMClassifier(n_estimators=n_estimators)
    start_time = time.time()
    clf.fit(x_train, y_train)
    end_time = time.time()
    exec_time = end_time - start_time
    LGC_tpred = clf.predict(x_train)
    LGC_tacc = accuracy_score(y_train, LGC_tpred)
    y_pred = clf.predict(x_test)
    LGCacc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    print("------LightGBMClassifier------")
    print(f'Accuracy on the train set: {LGC_tacc:.4f}')
    print(f'Accuracy on the test set: {LGCacc:.4f}')
    print(classification_report(y_test, y_pred))
    writer.writerow([
        "LightGBMClassifier", round(LGC_tacc * 100, 2), round(LGCacc * 100, 2),
        round(report['weighted avg']['precision'] * 100, 2),
        round(report['weighted avg']['recall'] * 100, 2),
        round(report['weighted avg']['f1-score'] * 100, 2),
        n_estimators,
        round(exec_time, 2)
    ])

    clf = LGBMRegressor(n_estimators=n_estimators)
    start_time = time.time()
    clf.fit(x_train, y_train)
    end_time = time.time()
    exec_time = end_time - start_time
    LGBR_tpred = clf.predict(x_train)
    fpr, tpr, tresholds = roc_curve(y_train, LGBR_tpred)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_treshold = tresholds[optimal_idx]
    y_pred = (LGBR_tpred >= optimal_treshold).astype(int)
    LGBR_tacc = accuracy_score(y_train, y_pred)
    LGBRpred = clf.predict(x_test)
    fpr, tpr, tresholds = roc_curve(y_test, LGBRpred)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_treshold = tresholds[optimal_idx]
    y_pred = (LGBRpred >= optimal_treshold).astype(int)
    LGBRacc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    print("------LightGBMRegressor------")
    print(f'Accuracy on the train set: {LGBR_tacc:.4f}')
    print(f'Accuracy on the test set: {LGBRacc:.4f}')
    print(classification_report(y_test, y_pred))
    writer.writerow([
        "LightGBMRegressor", round(LGBR_tacc * 100, 2), round(LGBRacc * 100, 2),
        round(report['weighted avg']['precision'] * 100, 2),
        round(report['weighted avg']['recall'] * 100, 2),
        round(report['weighted avg']['f1-score'] * 100, 2),
        n_estimators,
        round(exec_time, 2)
    ])

    n_estimators = 250
    clf = CatBoostClassifier(n_estimators=n_estimators)
    start_time = time.time()
    clf.fit(x_train, y_train)
    end_time = time.time()
    exec_time = end_time - start_time
    CBC_tpred = clf.predict(x_train)
    CBC_tacc = accuracy_score(y_train, CBC_tpred)
    y_pred = clf.predict(x_test)
    CBCacc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    print("------CatBoostClassifier------")
    print(f'Accuracy on the train set: {CBC_tacc:.4f}')
    print(f'Accuracy on the test set: {CBCacc:.4f}')
    print(classification_report(y_test, y_pred))
    writer.writerow([
        "CatBoostClassifier", round(CBC_tacc * 100, 2), round(CBCacc * 100, 2),
        round(report['weighted avg']['precision'] * 100, 2),
        round(report['weighted avg']['recall'] * 100, 2),
        round(report['weighted avg']['f1-score'] * 100, 2),
        n_estimators,
        round(exec_time, 2)
    ])

    clf = CatBoostRegressor(n_estimators=n_estimators)
    start_time = time.time()
    clf.fit(x_train, y_train)
    end_time = time.time()
    exec_time = end_time - start_time
    CBR_tpred = clf.predict(x_train)
    fpr, tpr, tresholds = roc_curve(y_train, CBR_tpred)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_treshold = tresholds[optimal_idx]
    y_pred = (CBR_tpred >= optimal_treshold).astype(int)
    CBR_tacc = accuracy_score(y_train, y_pred)
    CBRpred = clf.predict(x_test)
    fpr, tpr, tresholds = roc_curve(y_test, CBRpred)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_treshold = tresholds[optimal_idx]
    y_pred = (CBRpred >= optimal_treshold).astype(int)
    CBRacc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    print("------CatBoostRegressor------")
    print(f'Accuracy on the train set: {CBR_tacc:.4f}')
    print(f'Accuracy on the test set: {CBRacc:.4f}')
    print(classification_report(y_test, y_pred))
    writer.writerow([
        "CatBoostRegressor", round(CBR_tacc * 100, 2), round(CBRacc * 100, 2),
        round(report['weighted avg']['precision'] * 100, 2),
        round(report['weighted avg']['recall'] * 100, 2),
        round(report['weighted avg']['f1-score'] * 100, 2),
        n_estimators,
        round(exec_time, 2)
    ])

    n_estimators = 300
    clf = NGBClassifier(n_estimators=n_estimators)
    start_time = time.time()
    clf.fit(x_train, y_train)
    end_time = time.time()
    exec_time = end_time - start_time
    NGC_tpred = clf.predict(x_train)
    NGC_tacc = accuracy_score(y_train, NGC_tpred)
    y_pred = clf.predict(x_test)
    NGCacc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    print("------NGBoostClassifier------")
    print(f'Accuracy on the train set: {NGC_tacc:.4f}')
    print(f'Accuracy on the test set: {NGCacc:.4f}')
    print(classification_report(y_test, y_pred))
    writer.writerow([
        "NGBoostClassifier", round(NGC_tacc * 100, 2), round(NGCacc * 100, 2),
        round(report['weighted avg']['precision'] * 100, 2),
        round(report['weighted avg']['recall'] * 100, 2),
        round(report['weighted avg']['f1-score'] * 100, 2),
        n_estimators,
        round(exec_time, 2)
    ])

    clf = NGBRegressor(n_estimators=n_estimators)
    start_time = time.time()
    clf.fit(x_train, y_train)
    end_time = time.time()
    exec_time = end_time - start_time
    NGR_tpred = clf.predict(x_train)
    fpr, tpr, tresholds = roc_curve(y_train, NGR_tpred)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_treshold = tresholds[optimal_idx]
    y_pred = (NGR_tpred >= optimal_treshold).astype(int)
    NGR_tacc = accuracy_score(y_train, y_pred)
    NGRpred = clf.predict(x_test)
    fpr, tpr, tresholds = roc_curve(y_test, NGRpred)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_treshold = tresholds[optimal_idx]
    y_pred = (NGRpred >= optimal_treshold).astype(int)
    NGRacc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    print("------NGBoostRegressor------")
    print(f'Accuracy on the train set: {NGR_tacc:.4f}')
    print(f'Accuracy on the test set: {NGRacc:.4f}')
    print(classification_report(y_test, y_pred))
    writer.writerow([
        "NGBoostRegressor", round(NGR_tacc * 100, 2), round(NGRacc * 100, 2),
        round(report['weighted avg']['precision'] * 100, 2),
        round(report['weighted avg']['recall'] * 100, 2),
        round(report['weighted avg']['f1-score'] * 100, 2),
        n_estimators,
        round(exec_time, 2)
    ])

    from pgbm.sklearn import HistGradientBoostingRegressor
    max_iter = 650
    clf = HistGradientBoostingRegressor(max_iter=max_iter)
    start_time = time.time()
    clf.fit(x_train, y_train)
    end_time = time.time()
    exec_time = end_time - start_time
    HGBR_tpred = clf.predict(x_train)
    fpr, tpr, tresholds = roc_curve(y_train, HGBR_tpred)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_treshold = tresholds[optimal_idx]
    y_pred = (HGBR_tpred >= optimal_treshold).astype(int)
    HGBR_tacc = accuracy_score(y_train, y_pred)
    HGBRpred = clf.predict(x_test)
    fpr, tpr, tresholds = roc_curve(y_test, HGBRpred)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_treshold = tresholds[optimal_idx]
    y_pred = (HGBRpred >= optimal_treshold).astype(int)
    HGBRacc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    print("------PGBM------")
    print(f'Accuracy on the train set: {HGBR_tacc:.4f}')
    print(f'Accuracy on the test set: {HGBRacc:.4f}')
    print(classification_report(y_test, y_pred))
    writer.writerow([
        "PGBM", round(HGBR_tacc * 100, 2), round(HGBRacc * 100, 2),
        round(report['weighted avg']['precision'] * 100, 2),
        round(report['weighted avg']['recall'] * 100, 2),
        round(report['weighted avg']['f1-score'] * 100, 2),
        max_iter,
        round(exec_time, 2)
    ])

    n_estimators = 900
    clf = BoostingRegressor(
        base_estimator=DecisionTreeRegressor(max_depth=3),
        n_estimators=n_estimators,
        learning_rate=0.1
    )
    start_time = time.time()
    clf.fit(x_train, y_train)
    end_time = time.time()
    exec_time = end_time - start_time
    STR_tpred = clf.predict(x_train)
    fpr, tpr, tresholds = roc_curve(y_train, STR_tpred)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_treshold = tresholds[optimal_idx]
    y_pred = (STR_tpred >= optimal_treshold).astype(int)
    STR_tacc = accuracy_score(y_train, y_pred)
    STRpred = clf.predict(x_test)
    fpr, tpr, tresholds = roc_curve(y_test, STRpred)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_treshold = tresholds[optimal_idx]
    y_pred = (STRpred >= optimal_treshold).astype(int)
    STRacc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    print("------StarBoostRegressor------")
    print(f'Accuracy on the train set: {STR_tacc:.4f}')
    print(f'Accuracy on the test set: {STRacc:.4f}')
    print(classification_report(y_test, y_pred))
    writer.writerow([
        "StarBoost", round(STR_tacc * 100, 2), round(STRacc * 100, 2),
        round(report['weighted avg']['precision'] * 100, 2),
        round(report['weighted avg']['recall'] * 100, 2),
        round(report['weighted avg']['f1-score'] * 100, 2),
        n_estimators,
        round(exec_time, 2)
    ])
