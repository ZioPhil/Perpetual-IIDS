from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
from tensorflow.keras import activations
from tensorflow.keras.constraints import Constraint
from keras.callbacks import EarlyStopping
import datetime
from utils import *
from plt_utils import *
import random


##########################################################################################################
# Functions
class MaskWeights(Constraint):
    def __init__(self, mask):
        self.mask = mask
        self.mask = K.cast(self.mask, K.floatx())

    def __call__(self, w):
        w.assign(w * self.mask)
        return w

    def get_config(self):
        return {'mask': self.mask}


class DataGenerator(tf.keras.utils.Sequence):
    """Generates data for Keras"""
    def __init__(self, X, labels, batch_size=32, dim=(32, 32, 32), n_channels=1,
                 n_classes=10, shuffle=True):
        """Initialization"""
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.X = X
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        # list_IDs_temp = [self.X[k] for k in indexes]

        # Generate data
        y = tf.keras.utils.to_categorical(self.labels[indexes], num_classes=self.n_classes)
        X = self.X[indexes]
        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.X))
        if self.shuffle:
            np.random.shuffle(self.indexes)


class CustomCallback(tf.keras.callbacks.Callback):
    def on_train_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Training: start of batch {}; got log keys: {}".format(batch, keys))

    def on_train_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Training: end of batch {}; got log keys: {}".format(batch, keys))

    def on_test_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Evaluating: start of batch {}; got log keys: {}".format(batch, keys))

    def on_test_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Evaluating: end of batch {}; got log keys: {}".format(batch, keys))

    def on_predict_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Predicting: start of batch {}; got log keys: {}".format(batch, keys))

    def on_predict_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Predicting: end of batch {}; got log keys: {}".format(batch, keys))


def find_first_pos(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx


def find_last_pos(array, value):
    idx = (np.abs(array - value))[::-1].argmin()
    return array.shape[0] - idx


def createWeightsMask(epsilon, noRows, noCols):
    # generate an Erdos Renyi sparse weights mask
    mask_weights = np.random.uniform(low=0, high=1, size=(noRows, noCols))  # np.random.rand(noRows, noCols)
    prob = 1 - (epsilon * (noRows + noCols)) / (noRows * noCols)  # normal to have 8x connections
    mask_weights[mask_weights < prob] = 0
    mask_weights[mask_weights >= prob] = 1
    noParameters = np.sum(mask_weights)
    print("Create Sparse Matrix: No parameters, NoRows, NoCols ", noParameters, noRows, noCols)
    return [noParameters, mask_weights]


##########################################################################################################
# Model
class NeuroFS:
    def __init__(self, args):
        # set model parameters
        self.args = args

        print("Model parameters initialized \n")
        # generate an Erdos Renyi sparse weights mask for each layer
        print("\nLayer 1 ")
        [self.noPar1, self.wm1] = createWeightsMask(self.args.epsilon, self.args.dim, self.args.num_hidden)
        print("\nLayer 2 ")
        [self.noPar2, self.wm2] = createWeightsMask(self.args.epsilon, self.args.num_hidden, self.args.num_hidden)
        print("\nLayer 3 ")
        [self.noPar3, self.wm3] = createWeightsMask(self.args.epsilon, self.args.num_hidden, self.args.num_hidden)
        print("\nLayer 4 ")
        [self.noPar4, self.wm4] = createWeightsMask(self.args.epsilon, self.args.num_hidden, self.args.num_classes)
        self.gradients = []

        # initialize layers weights
        self.w1 = None
        self.w2 = None
        self.w3 = None
        self.w4 = None

        self.protection = np.zeros(self.args.dim)
        self.dim_removed = 0
        self.create_model(True)

    def create_model(self, begin):
        self.model = Sequential()
        act = activations.tanh

        self.model.add(Dense(self.args.num_hidden, name="sparse_1",
                             kernel_regularizer=l2(self.args.wd), bias_regularizer=l2(self.args.wd),
                             kernel_constraint=MaskWeights(self.wm1),
                             input_shape=(self.args.dim,), activation=act))
        if begin:
            self.w1 = [np.random.rand(*w.shape) for w in self.model.layers[0].get_weights()]
        self.model.layers[0].set_weights(self.w1)

        self.model.add(Dense(self.args.num_hidden, name="sparse_2",
                             kernel_regularizer=l2(self.args.wd), bias_regularizer=l2(self.args.wd),
                             kernel_constraint=MaskWeights(self.wm2),
                             activation=act))
        if begin:
            self.w2 = [np.random.rand(*w.shape) for w in self.model.layers[1].get_weights()]
        self.model.layers[1].set_weights(self.w2)

        self.model.add(Dense(self.args.num_hidden, name="sparse_3",
                             kernel_regularizer=l2(self.args.wd), bias_regularizer=l2(self.args.wd),
                             kernel_constraint=MaskWeights(self.wm3),
                             activation=act))
        if begin:
            self.w3 = [np.random.rand(*w.shape) for w in self.model.layers[2].get_weights()]
        self.model.layers[2].set_weights(self.w3)

        self.model.add(Dense(self.args.num_classes, name="sparse_4"))
        if begin:
            self.w4 = [np.random.rand(*w.shape) for w in self.model.layers[3].get_weights()]
        self.model.layers[3].set_weights(self.w4)

        self.model.add(Activation('softmax'))

    def select_input_neurons_to_remove(self, W):
        print("---> Update Input layer: select_input_neurons_to_remove")
        print("# Removed features until now = ", self.dim_removed, " out of ", self.args.dim_to_remove)
        strength_input_neurons = np.abs(W).sum(axis=1)

        # ----- compute number of neurons to update (remove and add) and num neurons to remove
        num_neurons_to_update = int(self.dim_removed * self.args.zeta_in * (1 - self.args.epoch / self.args.epochs))
        if self.dim_removed >= self.args.dim_to_remove:
            num_remove_epoch = 0
        else:
            num_remove_epoch = int(
                np.ceil((self.args.dim_to_remove - self.dim_removed) / (self.args.epoch_remove - self.args.epoch)))
        self.dim_removed += num_remove_epoch
        num_total_remove_epoch = int(num_remove_epoch + num_neurons_to_update)
        num_keep_epoch = self.args.dim - self.dim_removed - num_neurons_to_update

        if self.dim_removed >= self.args.dim_to_remove:
            idx_neurons_to_keep_final = np.argpartition(strength_input_neurons, -num_keep_epoch)[-num_keep_epoch:]
            idx_neurons_to_remove_final = np.setdiff1d(self.idx_active_neurons, idx_neurons_to_keep_final)

            print("# neurons to update (remove and add) at this epoch = ", num_neurons_to_update)

        else:

            # ---------- Compute actual numbers to remove (do not consider the immature neurons)
            idx_neurons_immature = np.where(self.protection < 1)[0]
            idx_neurons_mature = np.setdiff1d(self.idx_active_neurons, idx_neurons_immature)
            idx_neurons_to_keep = np.argpartition(strength_input_neurons, -num_keep_epoch)[-num_keep_epoch:]
            idx_neurons_mature_to_keep = np.intersect1d(idx_neurons_to_keep, idx_neurons_mature)
            num_total_remove_epoch_final = idx_neurons_mature.shape[0] - idx_neurons_mature_to_keep.shape[0]

            strength_mature = strength_input_neurons[idx_neurons_mature]
            num_to_keep_final = idx_neurons_mature.shape[0] - num_total_remove_epoch_final
            idx_top_mature = np.argpartition(strength_mature, -num_to_keep_final)[-num_to_keep_final:]
            idx_neurons_mature_to_keep_final = idx_neurons_mature[idx_top_mature]
            idx_neurons_to_remove_final = np.setdiff1d(idx_neurons_mature, idx_neurons_mature_to_keep_final)

            # ------ determine the neurons to keep (including the top mature ones and the active immature)
            idx_active_immature = np.intersect1d(idx_neurons_immature, self.idx_active_neurons)
            self.protection[idx_active_immature] += 1
            self.protection[idx_neurons_mature_to_keep_final] += 1
            idx_neurons_to_keep_final = np.concatenate((idx_neurons_mature_to_keep_final, idx_active_immature))

            # ------
            if num_total_remove_epoch_final < num_total_remove_epoch:
                num_neurons_to_update = num_total_remove_epoch_final - num_remove_epoch
                if num_neurons_to_update < 0:
                    self.dim_removed -= num_remove_epoch
                    num_remove_epoch = num_total_remove_epoch_final
                    self.dim_removed += num_total_remove_epoch_final
                    num_neurons_to_update = 0

            print("# neurons to decay at this epoch  = ", num_remove_epoch)
            print("# neurons to update (remove and add) at this epoch = ", num_neurons_to_update)
        return idx_neurons_to_keep_final, idx_neurons_to_remove_final, num_neurons_to_update

    def neuron_removal(self, W, idx_neurons_to_remove):
        print("---> Update Input layer: neuron_removal")
        W[idx_neurons_to_remove] = 0
        self.protection[idx_neurons_to_remove] = 0
        print(idx_neurons_to_remove.shape[0], " neurons has been removed")
        return W

    def weight_removal(self, W, idx_neurons_to_keep):
        print("---> Update Input layer: Updating neurons to keep  (Weight removal)  ")

        # ------- compute the thresholds
        W_copy = np.copy(W[idx_neurons_to_keep])
        values = np.sort(W_copy.ravel())
        firstZeroPos = find_first_pos(values, 0)
        lastZeroPos = find_last_pos(values, 0)
        largestNegative = values[int((1 - self.args.zeta_in) * firstZeroPos)]
        smallestPositive = values[
            int(min(values.shape[0] - 1, lastZeroPos + self.args.zeta_in * (values.shape[0] - lastZeroPos)))]
        # --------remove weights
        W2 = W.copy()
        W2[W2 < largestNegative] = 1
        W2[W2 > smallestPositive] = 1
        W2[W2 != 1] = 0

        rewiredWeights = W2.copy()
        rewiredWeights[rewiredWeights > 0] = 1
        rewiredWeights[rewiredWeights < 0] = 1
        weightMaskCore = rewiredWeights.copy()
        return W2, rewiredWeights, weightMaskCore

    def select_neurons_to_add(self, W, num_neurons_to_update):
        print("---> Update Input layer: Select Neurons to add  ")

        if num_neurons_to_update > 0:
            idx_deactive_neurons = np.setdiff1d(np.arange(W.shape[0]), self.idx_active_neurons)
            if len(idx_deactive_neurons) > 0:
                if self.args.gradient_addition:
                    print("Neurons are selected based on gradient")
                    grad_deactive_neurons = np.abs(self.gradients[0][idx_deactive_neurons, :]).ravel()
                    ind = np.argpartition(grad_deactive_neurons, -grad_deactive_neurons.shape[0])[
                          -grad_deactive_neurons.shape[0]:]
                    ind = np.argsort(-grad_deactive_neurons[ind])
                    ncols = self.gradients[0].shape[1]
                    idx_neurons_to_add = []
                    ind = ind / ncols
                    ind = ind.astype(int)
                    index = np.unique(ind, return_index=True)[1]
                    ind = ind[sorted(index)]
                    idx_neurons_to_add = idx_deactive_neurons[ind[:num_neurons_to_update]]
                else:
                    print("Neurons are selected randomly")
                    idx_neurons_to_add = np.array(random.sample(list(idx_deactive_neurons), num_neurons_to_update))
            else:  # ----- no new neurons are added
                idx_neurons_to_add = np.asarray([])
        else:  # ----- no new neurons are added
            idx_neurons_to_add = np.asarray([])
        print(idx_neurons_to_add.shape[0], " neurons are added", flush=True)
        return idx_neurons_to_add

    def weight_addition(self, W2, rewiredWeights, idx_neurons_to_keep, idx_neurons_to_add, num_neurons_to_update):
        if num_neurons_to_update == 0:
            return rewiredWeights

        print("---> Update Input layer: Add weights ")

        num_existing_connections = np.count_nonzero(W2[idx_neurons_to_keep])

        idx_neurons_to_keep = idx_neurons_to_keep.astype(int)
        idx_neurons_to_add = idx_neurons_to_add.astype(int)
        if self.args.gradient_addition:
            array1d_new_connectionspp_kept = np.abs(self.gradients[0][idx_neurons_to_keep]).ravel()
        else:
            array1d_new_connectionspp_kept = np.random.rand(idx_neurons_to_keep.shape[0] * self.args.num_hidden)

        idx_keep_neurons_old_connections = np.array(np.nonzero(W2[idx_neurons_to_keep].ravel())).ravel()
        array1d_new_connectionspp_kept[idx_keep_neurons_old_connections] = 0

        if self.args.gradient_addition:
            array1d_new_connectionspp_new = np.abs(self.gradients[0][idx_neurons_to_add]).ravel()
        else:
            array1d_new_connectionspp_new = np.random.rand(idx_neurons_to_add.shape[0] * self.args.num_hidden)

        array1d_new_connectionspp = np.concatenate((array1d_new_connectionspp_kept, array1d_new_connectionspp_new))

        idx_features_add_connection = np.concatenate((idx_neurons_to_keep, idx_neurons_to_add))

        noRows = W2.shape[0]
        noCols = W2.shape[1]
        total_connections = int((noRows * noCols) * ((noRows + noCols) / (noRows * noCols)))
        num_connections_to_add_overall = total_connections - num_existing_connections

        ind = np.argpartition(array1d_new_connectionspp, -num_connections_to_add_overall)[
              -num_connections_to_add_overall:]
        array1d_new_connectionspp[:] = 0
        array1d_new_connectionspp[ind] = 1
        array2d_new_connectionsolddd = np.reshape(array1d_new_connectionspp,
                                                  (idx_features_add_connection.shape[0], W2.shape[1]))
        rewiredWeights[idx_features_add_connection] += array2d_new_connectionsolddd
        rewiredWeights[rewiredWeights > 0] = 1
        rewiredWeights[rewiredWeights < 0] = 1
        return rewiredWeights

    def rewireMask_input_gradual(self, W, noWeights):
        # -------- Compute strength
        # strength_grad_input_neurons = np.abs(self.gradients[0]).sum(axis=1)
        strength_input_neurons = np.abs(W).sum(axis=1)
        idx_selected_features = np.argpartition(strength_input_neurons, -self.args.K)[-self.args.K:]
        # -------- Active Neurons
        self.idx_active_neurons = np.array(np.nonzero(strength_input_neurons)).ravel()
        print("# active neurons = ", self.idx_active_neurons.shape[0])
        # -------- Select Neurons to keep/remove
        idx_neurons_to_keep, idx_neurons_to_remove, num_neurons_to_update = self.select_input_neurons_to_remove(W)
        # -------- Remove Neurons
        W = self.neuron_removal(W, idx_neurons_to_remove)
        # -------- Remove weights
        W2, rewiredWeights, weightMaskCore = self.weight_removal(W, idx_neurons_to_keep)
        # -------- Select Neurons to add
        idx_neurons_to_add = self.select_neurons_to_add(W2, num_neurons_to_update)
        # -------- Add weights
        rewiredWeights = self.weight_addition(W2, rewiredWeights, idx_neurons_to_keep, idx_neurons_to_add,
                                              num_neurons_to_update)

        strength_input_neurons = np.abs(rewiredWeights).sum(axis=1)
        self.idx_active_neurons = np.array(np.nonzero(strength_input_neurons)).ravel()

        return [rewiredWeights, weightMaskCore], idx_selected_features

    def rewireMask(self, weights, noWeights, idx_grad=0, mode="grad"):
        # rewire weight matrix

        # remove zeta largest negative and smallest positive weights
        values = np.sort(weights.ravel())
        firstZeroPos = find_first_pos(values, 0)
        lastZeroPos = find_last_pos(values, 0)
        largestNegative = values[int((1 - self.args.zeta_hid) * firstZeroPos)]
        smallestPositive = values[
            int(min(values.shape[0] - 1, lastZeroPos + self.args.zeta_hid * (values.shape[0] - lastZeroPos)))]
        rewiredWeights = weights.copy()
        rewiredWeights[rewiredWeights > smallestPositive] = 1
        rewiredWeights[rewiredWeights < largestNegative] = 1
        rewiredWeights[rewiredWeights != 1] = 0
        weightMaskCore = rewiredWeights.copy()

        noRewires = int(noWeights - np.sum(rewiredWeights))

        if mode == "grad":
            array1d_new_connections = np.abs(self.gradients[idx_grad])
            rewiredWeights2 = np.copy(rewiredWeights)
            rewiredWeights2 = 1 - rewiredWeights2
            array1d_new_connections = rewiredWeights2.ravel() * array1d_new_connections.ravel()
            ind = np.argpartition(array1d_new_connections, -noRewires)[-noRewires:]
            array1d_new_connections[:] = 0
            array1d_new_connections[ind] = 1
            array2d_new_connections = np.reshape(array1d_new_connections,
                                                 (rewiredWeights.shape[0], rewiredWeights.shape[1]))

            rewiredWeights = rewiredWeights + array2d_new_connections
        else:

            old_weights = weights.copy()
            old_weights[old_weights > 0] = 1
            old_weights[old_weights < 0] = 1
            old_weights = old_weights.ravel()
            indices_zero_weights_old = np.where(old_weights == 0)[0]
            print("indices_zero_weights_old.shape[0] = ", indices_zero_weights_old.shape[0])
            print("noRewires = ", noRewires)
            index = np.random.choice(np.arange(indices_zero_weights_old.shape[0]), int(noRewires), replace=False)
            idx_to_add = indices_zero_weights_old[index]

            rewiredWeights = rewiredWeights.ravel()
            rewiredWeights[idx_to_add] = 1
            rewiredWeights = rewiredWeights.reshape((weights.shape[0], weights.shape[1]))
        return [rewiredWeights, weightMaskCore]

    def weightsEvolution(self):
        self.w1 = self.model.get_layer("sparse_1").get_weights()
        self.w2 = self.model.get_layer("sparse_2").get_weights()
        self.w3 = self.model.get_layer("sparse_3").get_weights()
        self.w4 = self.model.get_layer("sparse_4").get_weights()

        # ---------------------- compute sparsity level ----------------------
        density_l1 = (np.count_nonzero(self.wm1) / (self.wm1.shape[0] * self.wm1.shape[1]))
        density_l2 = (np.count_nonzero(self.wm2) / (self.wm2.shape[0] * self.wm2.shape[1]))
        density_l3 = (np.count_nonzero(self.wm3) / (self.wm3.shape[0] * self.wm3.shape[1]))
        density_l4 = (np.count_nonzero(self.wm4) / (self.wm4.shape[0] * self.wm4.shape[1]))
        print("-----------------------------------------------------------")
        print("------------           Network Sparsity         -----------")
        print("Density of layer 1 = ", density_l1, ", sparsity = ", 1 - density_l1)
        print("Density of layer 2 = ", density_l2, ", sparsity = ", 1 - density_l2)
        print("Density of layer 3 = ", density_l3, ", sparsity = ", 1 - density_l3)
        print("Density of layer 4 = ", density_l4, ", sparsity = ", 1 - density_l4)

        print("-----------------------------------------------------------")
        print("------------         Update Connectivity        -----------")
        # ---------------------- Rewire input layer ----------------------
        if self.args.epoch < 2:
            [self.wm1, self.wm1Core] = self.rewireMask(self.w1[0], self.noPar1, 0)
            strength_input_neurons = np.abs(self.w1[0]).sum(axis=1)
            num_all_features = strength_input_neurons.shape[0]
            self.idx_active_neurons = np.array(np.nonzero(strength_input_neurons)).ravel()
            idx_selected_features = np.argpartition(strength_input_neurons, -self.args.K)[-self.args.K:]
            self.protection += np.ones(self.args.dim)
        else:
            [self.wm1, self.wm1Core], idx_selected_features = self.rewireMask_input_gradual(self.w1[0], self.noPar1)

        # ---------------------- Rewire other layers ----------------------
        if self.args.gradient_addition:
            # gradient-based addition
            [self.wm2, self.wm2Core] = self.rewireMask(self.w2[0], self.noPar2, idx_grad=2, mode="grad")
            [self.wm3, self.wm3Core] = self.rewireMask(self.w3[0], self.noPar3, idx_grad=4, mode="grad")
        else:
            # Random addition
            [self.wm2, self.wm2Core] = self.rewireMask(self.w2[0], self.noPar2, mode="rand")
            [self.wm3, self.wm3Core] = self.rewireMask(self.w3[0], self.noPar3, mode="rand")

        # ---------------------- Finalize weights ----------------------
        self.w1[0] = self.w1[0] * self.wm1Core
        self.w2[0] = self.w2[0] * self.wm2Core
        self.w3[0] = self.w3[0] * self.wm3Core
        return idx_selected_features

    def train(self, args):
        print("------- Data info")
        x_train = args.data[0]
        x_test = args.data[1]
        y_train = args.data[2]
        y_test = args.data[3]
        y_train2 = np.array([np.where(r == 1)[0][0] for r in y_train])
        y_test2 = np.array([np.where(r == 1)[0][0] for r in y_test])
        print("x_train shape = " + str(x_train.shape))
        print("y_train shape = " + str(y_train2.shape))
        print("x_test shape = " + str(x_test.shape))
        print("y_test shape = " + str(y_test2.shape))

        print("------------ Network Structure  ")
        self.model.summary()
        print("\n\n")

        for epoch in range(0, self.args.epochs):
            self.args.epoch = epoch
            print("\n\n#######################################################################")
            print("########                       epoch ", epoch, "/", self.args.epochs,
                  "                       ########")
            t_start_epoch = datetime.datetime.now()

            opt = optimizers.SGD(learning_rate=self.args.lr, momentum=self.args.momentum)
            self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
            early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=1, mode='min')

            print("-----------------------------------------------------------")
            print("----------              Fit the model             ---------")
            historytemp = self.model.fit(x_train, y_train,
                                         steps_per_epoch=None,
                                         epochs=epoch, verbose=2,
                                         validation_data=(x_test, y_test),
                                         initial_epoch=epoch - 1,
                                         validation_steps=None,
                                         # callbacks=[early_stopping]
                                         # callbacks=[CustomCallback()]
                                         )

            # --------- log the model loss/acc
            print("t (fit) = ", datetime.datetime.now() - t_start_epoch)
            args.results["ACC_train_model"].append(historytemp.history['accuracy'][0])
            args.results["Loss_train_model"].append(historytemp.history['loss'][0])
            args.results["ACC_test_model"].append(historytemp.history['val_accuracy'][0])
            args.results["Loss_test_model"].append(historytemp.history['val_loss'][0])

            # -------- get gradients for weight updating
            with tf.GradientTape(persistent=True) as tape:
                predictions = self.model(x_train)  # Forward pass
                loss = tf.keras.losses.categorical_crossentropy(y_train, predictions)  # Compute the loss
            self.gradients = tape.gradient(loss, self.model.trainable_weights)
            self.gradients = [gradient.numpy() for gradient in self.gradients]
            del tape
            # -------- update connectivity
            if epoch % self.args.update_interval == 0:
                t_start_weight_update = datetime.datetime.now()
                indices = self.weightsEvolution()
                selected_columns = [args.column_names[x] for x in sorted(indices)]
                print("t (update connectivity) = ", datetime.datetime.now() - t_start_weight_update, flush=True)
                t_start_evaluation = datetime.datetime.now()

                # --------- evaluate feature selection performance
                if epoch == self.args.epochs - 1 or (epoch % 10 == 0):
                    print("-----------------------------------------------------------")
                    print("----------        Evaluate Feature Selection      ---------")

                    knn_acc, et_acc, svc_acc = eval_subset_supervised([x_train[:, indices], x_train, y_train2.ravel()],
                                                                      [x_test[:, indices], x_test, y_test2.ravel()])
                    print("t (evaluation) = ", datetime.datetime.now() - t_start_evaluation, flush=True)
                else:
                    knn_acc = 0
                    et_acc = 0
                    svc_acc = 0

                args.results["SVC"].append(svc_acc)
                args.results["KNN"].append(knn_acc)
                args.results["EXT"].append(et_acc)
                save_obj(args.save_path + "results_NeuroFS", args.results)
                with open(args.columns_path, 'a') as the_file:
                    the_file.write("Selected columns: {}, ({})\n"
                                   .format(selected_columns, round(historytemp.history['val_loss'][0], 5)))
                with open(args.result_path, 'a') as the_file:
                    the_file.write('KNNacc = {:.3f}, ETacc = {:.3f}, SVCacc = {:.3f}, '
                                   .format(knn_acc, et_acc, svc_acc))
                plt_loss(args)
                plt_accuracy_supervised_feature_selection(args)

            K.clear_session()
            self.create_model(False)
            print("\nt_total (epoch) = ", datetime.datetime.now() - t_start_epoch)
