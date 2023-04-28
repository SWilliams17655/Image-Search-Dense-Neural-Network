import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pickle


def save_to_csv(data_frame, file_location):
    """
    Saves a data set to a .csv file.
    :param data_frame: [Required] DataFrame representing the file to be saved.
    :param file_location: [Required] String representing file location for where .csv will be saved.
    :return: none
    """

    os_loc = input(file_location)
    data_frame.to_csv(os_loc)


def train_neural_network(x_train, y_train, x_eval_set, y_eval_set):
    """
    Trains a baseline neural network to classify data using test and training.

    :param x_train : DataFrame with attributes used for classification.
    :param y_train : Series with solutions for x_train attributes.
    :param x_eval_set : DataFrame with attributes used for verification.
    :param y_eval_set : Series with solutions for x_test attributes.
    :return: Returns the trained model.
    """

    print("\n\n Training neural network model for classification...")

    num_epoch = 30
    num_batch = 500
    drop_out_rate = .2
    initializer = tf.keras.initializers.HeUniform()
    num_hidden_layer_1 = len(x_train.columns)
    num_hidden_layer_2 = len(x_train.columns) // 2
    num_hidden_layer_3 = len(x_train.columns) // 2
    num_output_layer = 10

    # creates the neural network model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(num_hidden_layer_1, input_shape=(784,),
                              activation=tf.nn.relu,
                              kernel_initializer=initializer),
        tf.keras.layers.Dropout(drop_out_rate,
                                input_shape=(num_hidden_layer_1,)),
        tf.keras.layers.Dense(num_hidden_layer_2, activation=tf.nn.relu,
                              kernel_initializer=initializer),
        tf.keras.layers.Dense(num_hidden_layer_3, activation=tf.nn.relu,
                              kernel_initializer=initializer),
        tf.keras.layers.Dense(num_output_layer, activation=tf.nn.softmax,
                              kernel_initializer=initializer)
    ])

    # compiles the model using accuracy as the metric and adam as the optimizer
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    csv_logger = tf.keras.callbacks.CSVLogger("NN_log.csv")

    # Trains model using x_test and y_test.
    model.fit(x_train, y_train,
              epochs=num_epoch,
              batch_size=num_batch,
              validation_data=(x_eval_set, y_eval_set),
              callbacks=[csv_logger])

    return model
