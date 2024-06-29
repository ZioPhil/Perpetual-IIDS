# imports
import numpy as np
import matplotlib.pyplot as plt


def plt_loss(args):

    loss_train = np.asarray(args.results["Loss_train_model"])
    loss_test = np.asarray(args.results["Loss_test_model"]) 
    epochs = np.arange(loss_train.shape[0])
    plt.figure(figsize=(10, 5))

    dots, = plt.plot(epochs, loss_train, linestyle='--', color='blue', label="Train Loss")
    dots, = plt.plot(epochs, loss_test, linestyle='--', color='green', label="Validation Loss")
    # plt.legend([dot0, dot1],  ["Train", "Test"])

    plt.xlabel('# of epoch')
    plt.ylabel('MSE')
    plt.title("Loss") 
    plt.legend(loc="best")
    plt.grid()
    plt.savefig(args.save_path+"Loss.pdf")
    plt.close()


def zero_to_nan(values):
    """Replace every 0 with 'nan' and return a copy."""
    return [np.nan if x == 0 else x for x in values]


def plt_accuracy_supervised_feature_selection(args):
    acc_train = np.asarray(args.results["ACC_train_model"]) * 100
    acc_test = np.asarray(args.results["ACC_test_model"]) * 100
    SVCacc = np.asarray(args.results["SVC"]) * 100
    ETacc = np.asarray(args.results["EXT"]) * 100
    KNNacc = np.asarray(args.results["KNN"]) * 100
    plt.figure()

    len_arr = np.arange(SVCacc.shape[0])
    # print(SVCacc)
    SVCacc = np.asarray(zero_to_nan(SVCacc))
    SVCacc = np.array(SVCacc).astype(np.double)
    mask = np.isfinite(SVCacc)
    KNNacc = np.array(KNNacc).astype(np.double)
    ETacc = np.array(ETacc).astype(np.double)

    # print(SVCacc)
    ETacc = np.asarray(zero_to_nan(ETacc))
    KNNacc = np.asarray(zero_to_nan(KNNacc))
    
    dots, = plt.plot(len_arr[mask], SVCacc[mask], color='red', linestyle='-', marker='o', label="SVC")
    dots, = plt.plot(len_arr[mask], ETacc[mask],  color='m', linestyle='-', marker='o', label="ExtraTrees")
    dots, = plt.plot(len_arr[mask], KNNacc[mask], color='sienna', linestyle='-', marker='o', label="KNN")
 
    plt.xlabel('# of epoch')
    plt.ylabel('Accuracy (%)')
    plt.title("Feature Selection Supervised Evaluation" + "( " + str(args.K) + " features (#)"+")")
    plt.legend(loc="best")
    plt.grid()
    plt.savefig(args.save_path+"Feature_Selection_Accuracy.pdf")
    plt.close()
