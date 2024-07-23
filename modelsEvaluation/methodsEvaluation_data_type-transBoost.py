import pandas as pd
import csv
import time
from sklearn.metrics import accuracy_score, classification_report, roc_curve, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from xgboost import TransBoostClassifier

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

with open("models_results.csv", "a", newline="") as file:
    writer = csv.writer(file)

    n_estimators = 850
    clf = TransBoostClassifier(max_depth=4,
                               learning_rate=0.1,
                               n_estimators=n_estimators,
                               min_child_weight=0,
                               reg_alpha=0.,
                               reg_lambda=1.,
                               objective='binary:logistic',
                               seed=1440,
                               transfer_decay_ratio=2.,
                               transfer_velocity=1.,
                               transfer_rebalance=False,
                               transfer_min_leaf_size=10,
                               transfer_prior_margin='mirror',
                               transfer_margin_estimation='firstorder',
                               verbosity=0,
                               nthread=64)
    start_time = time.time()
    clf.fit(x_train, y_train, x_test, y_test)
    end_time = time.time()
    best_exec_time = end_time - start_time
    y_pred = clf.predict(x_test)
    TBCacc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * ((precision * recall) / (precision + recall))
    print("------TransBoost------")
    print(f'Accuracy on the test set: {TBCacc:.4f}')
    print(classification_report(y_test, y_pred))
    writer.writerow([
        "TransBoost", round(TBCacc * 100, 2),
        round(precision, 2), round(recall, 2), round(f1_score, 2),
        tn, fp, tp, fn,
        n_estimators,
        round(best_exec_time, 2),
        round(best_exec_time, 2),
        round(n_estimators/best_exec_time, 4)
    ])
