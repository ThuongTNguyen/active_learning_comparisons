import tensorflow as tf
import gc
import numpy as np
from keras import backend as backend

class DAL(object):
    '''Adapt from low-resource-text-classification-framework/lrtc_lib/active_learning/core/strategy/discriminative_representation_sampling.py'''
    def __init__(self, additional_transform=None):
        self.sub_batches = 10
        self.additional_transform = additional_transform

    def acquire(self, X_L, y_L, X_U, X_L_trf, X_U_trf, clf, size, **kwargs):
        if self.additional_transform:
            X_L_trf_discr, X_U_trf_discr = self.additional_transform(self, X_L, y_L, X_U, X_L_trf, X_U_trf, clf, size, kwargs)
        else:
            X_L_trf_discr, X_U_trf_discr = X_L_trf, X_U_trf

        if not isinstance(X_L_trf_discr, np.ndarray):  # needed for CountVec output
            X_L_trf_discr = X_L_trf_discr.toarray()
            X_U_trf_discr = X_U_trf_discr.toarray()
        print(X_L_trf_discr.shape, X_U_trf_discr.shape)
        X_train = np.vstack((X_L_trf_discr, X_U_trf_discr))
        labeled_idx = np.arange(len(X_L_trf_discr))
        unlabeled_idx = np.arange(len(X_L_trf_discr), len(X_train))

        # from the original repo
        selected_unlabeled_idx = np.random.choice(unlabeled_idx, np.min([len(labeled_idx) * 10, len(unlabeled_idx)]),
                                                  replace=False)
        labeled_so_far = 0
        additional_to_predict_idx = []
        sub_sample_size = int(size / self.sub_batches)
        while labeled_so_far < size:
            if labeled_so_far + sub_sample_size > size:
                sub_sample_size = size - labeled_so_far
            backend.clear_session()
            model = train_discriminative_model(X_train[labeled_idx], X_train[selected_unlabeled_idx], len(X_train[0]))
            idx_to_predict = selected_unlabeled_idx
            predictions = model.predict(X_train[idx_to_predict])
            selected_indices = np.argpartition(predictions[:, 1], -sub_sample_size)[-sub_sample_size:]
            labeled_so_far += sub_sample_size
            unlabeled_idx = [i for i in unlabeled_idx if i not in idx_to_predict[selected_indices]]
            labeled_idx = np.hstack((labeled_idx, idx_to_predict[selected_indices]))
            additional_to_predict_idx = \
                np.hstack((additional_to_predict_idx, idx_to_predict[selected_indices])).astype(int)
            selected_unlabeled_idx = np.random.choice(unlabeled_idx,
                                                      np.min([len(labeled_idx) * 10, len(unlabeled_idx)]),
                                                      replace=False)

            # if labeled_so_far==sample_size:
            #     additional_to_predict_idx = np.sort(additional_to_predict_idx)
            #     predictions = model.predict(Train[additional_to_predict_idx])

            # delete the model to free GPU memory:
            del model
            gc.collect()
        return additional_to_predict_idx - len(X_L)


def train_discriminative_model(labeled, unlabeled, input_shape):
    """
    A function that trains and returns a discriminative model on the labeled and unlabeled data.
    """

    # create the binary dataset:
    y_L = np.zeros((labeled.shape[0], 1), dtype='int')
    y_U = np.ones((unlabeled.shape[0], 1), dtype='int')
    X_train = np.vstack((labeled, unlabeled))
    Y_train = np.vstack((y_L, y_U))
    Y_train = tf.keras.utils.to_categorical(Y_train)

    # build the model:
    model = get_discriminative_model(input_shape)

    # train the model:
    batch_size = 100
    epochs = 10
    optimizer = tf.keras.optimizers.Adam(lr=0.0001)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    callbacks = [DiscriminativeEarlyStopping()]
    model.fit(X_train, Y_train,
              epochs=epochs,
              batch_size=batch_size,
              shuffle=True,
              callbacks=callbacks,
              class_weight={0: float(X_train.shape[0]) / Y_train[Y_train == 0].shape[0],
                            1: float(X_train.shape[0]) / Y_train[Y_train == 1].shape[0]},
              verbose=2)

    return model


def get_discriminative_model(input_shape):
    """
    The MLP model for discriminative active learning, without any regularization techniques.
    """
    width = input_shape
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(width, activation='relu'))
    model.add(tf.keras.layers.Dense(width, activation='relu'))
    model.add(tf.keras.layers.Dense(width, activation='relu'))
    model.add(tf.keras.layers.Dense(2, activation='softmax', name='softmax'))
    return model


class DiscriminativeEarlyStopping(tf.keras.callbacks.Callback):
    """
    A custom callback for discriminative active learning, to stop the training a little bit before the classifier is
    able to get 100% accuracy on the training set. This makes sure examples which are similar to ones already in the
    labeled set won't have a very high confidence.
    """

    def __init__(self, monitor='accuracy', threshold=0.98, verbose=0):
        super(tf.keras.callbacks.Callback, self).__init__()
        self.monitor = monitor
        self.threshold = threshold
        self.verbose = verbose
        self.improved = 0

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)

        if current > self.threshold:
            if self.verbose > 0:
                print("Epoch {e}: early stopping at accuracy {a}".format(e=epoch, a=current))
            self.model.stop_training = True


if __name__ == '__main__':
    pass