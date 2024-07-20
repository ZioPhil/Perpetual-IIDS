import csv

import numpy as np
import pandas as pd
import time
from sklearn.metrics import classification_report, roc_curve, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

df = pd.read_csv("modbusDataset.csv", sep=",", header=0, index_col=None)

x = df[['flow', 'modbus_pdu', 'src_port', 'dst_port', 'ip_len', 'ip_chksum', 'modbus_len',
        'modbus_start_addr_NA_IND', 'modbus_output_addr_NA_IND', 'modbus_output_value_NA_IND',
        'modbus_byte_count_NA_IND', 'modbus_coil_status_0_NA_IND', 'modbus_register_val_NA_IND',
        'modbus_register_addr_NA_IND']]
y = df['data_type']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

mms = MinMaxScaler()
mms.fit(X_train)
x_train = mms.transform(X_train)
x_test = mms.transform(X_test)

with open("models_results.csv", "a", newline="") as file:
    writer = csv.writer(file)

    n_epochs = 22
    model = Sequential([
        Dense(256, activation='relu', input_shape=(14,)),  # Input layer with 14 features
        Dropout(0.2),
        Dense(128, activation='relu'),                      # Hidden layer 1
        Dense(64, activation='relu'),                      # Hidden layer 2
        Dense(32, activation='relu'),                      # Hidden layer 3
        Dense(16, activation='relu'),                      # Hidden layer 4
        Dense(1, activation='sigmoid')                     # Output layer with 1 neuron for binary classification
    ])

    model.compile(optimizer="rmsprop", loss='binary_crossentropy', metrics=['accuracy'])

    start_time = time.time()
    history = model.fit(X_train, y_train, epochs=n_epochs, batch_size=64, validation_split=0.2)
    end_time = time.time()
    best_exec_time = end_time - start_time
    y_pred = model.predict(X_test)
    fpr, tpr, tresholds = roc_curve(y_test, y_pred)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_treshold = tresholds[optimal_idx]
    y_pred = (y_pred >= optimal_treshold).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * ((precision * recall) / (precision + recall))
    print("------NeuralNetwork------")
    print(f'Accuracy on the test set: {accuracy:.4f}')
    print(classification_report(y_test, y_pred))
    writer.writerow([
        "NeuralNetwork", round(accuracy * 100, 2),
        round(precision, 2), round(recall, 2), round(f1_score, 2),
        tn, fp, tp, fn,
        n_epochs,
        round(best_exec_time, 2),
        round(best_exec_time, 2),
        round(n_epochs/best_exec_time, 4)
    ])
