# imports
from utils import *
from NeuroFS import NeuroFS
from argparse import Namespace

import os
import sys
import random
import numpy as np
import tensorflow as tf
sys.path.append(os.getcwd())


if __name__ == '__main__':
    #########################################################################
    # parameter initialization and load data
    args = Namespace(
        data="modbusDataset.csv",  # data to filter
        K=20,  # Number of features to select
        # network sparsity, if i see that the sparsity level in the first 3 layers is 0, this is too high
        epsilon=7,
        zeta_in=0.2,
        zeta_hid=0.3,
        frac_epoch_remove=0.65,
        # empirical rule for number of hidden neurons: fewer than twice the size of the input layer, 2/3 of the size
        # of the input layer + the size of the output layer, between the size of input and output layer
        num_hidden=18,
        update_interval=1,  # weights update interval
        gradient_addition=True,
        epochs=70,  # number of epochs
        seed=42,  # seed for randomness
        batch_size=32,  # size of the batches the input data gets divided into during training
        lr=0.01,  # learning rate, set according to earlier results
        wd=0.001,  # weight decay, set according to earlier results
        momentum=0.9
        )

    args, data = load_data(args)

    os.environ['PYTHONHASHSEED'] = str(args.seed)

    # 2. Set the `python` built-in pseudo-random generator at a fixed value
    random.seed(args.seed)

    # 3. Set the `numpy` pseudo-random generator at a fixed value
    np.random.seed(args.seed)

    # 4. Set the `tensorflow` pseudo-random generator at a fixed value
    tf.random.set_seed(args.seed)

    #########################################################################
    # create directory to save the results

    path = "./results/" + args.data + "/K=" + str(args.K) + "/" + "seed=" + str(args.seed) + "/"
    check_path(path)
    save_path = path  # + + "_"
    # save parameters in save_path
    f = open(save_path+"result.txt", 'w')
    args.save_path = save_path
    args.result_path = args.save_path + "result.txt"
    args.columns_path = args.save_path + "columns.txt"
    f.close()
    print("Feature Selection\n")
    print(args)
    # log_file = open(save_path +"_log.txt", "w")
    results = {"ACC_train_model": [], "Loss_train_model": [],
               "ACC_test_model": [], "Loss_test_model": [],
               "SVC": [], "KNN": [], "EXT": []}
    args.results = results
    args.data = data

    #########################################################################
    # Create Model
    model = NeuroFS(args)

    #########################################################################
    # start training
    model.train(args)
