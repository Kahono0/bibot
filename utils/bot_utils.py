import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


def trash_file(filename=None):
    try:
        os.remove(filename)
        return True
    except:
        pass


def parse_output(data=None):
    try:
        target_output = np.array(data[-1:])
        assert (target_output.shape[1] == 4)
        indices = ["high", "low", "open", "close"]
        results = dict()
        results["high"], results["low"] = target_output[0][0], target_output[0][1]
        results["open"], results["close"] = target_output[0][2], target_output[0][3]
        return results
    except Exception as error:
        print("An error occured in bot-utils: ", error)
        return False


def parse_outputs(data=None):
    try:
        data = np.expand_dims(np.array(data), 1)
        assert (data[0].shape[1] == 4)
        results = pd.DataFrame(columns=["high", "low", "open", "close"])
        for target in data:
            results.loc[results.shape[0] + 1] = list(target[0])
        return results
    except Exception as error:
        print("An error occured in bot-utils: ", error)
        return False


class AnalystCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        print("loss: ", logs.get("loss"))
        if (logs.get("loss") < 0.000998):
            self.model.stop_training = True


AnalystCallbacks = AnalystCallback()


def PriceAnalyst(input_shape, MODEL_NAME):
    X_input = tf.keras.Input(shape=input_shape)

    X = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True),
        name="DIGEST_SEQUENCE")(X_input)

    X = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=False),
        name="RE_DIGEST_SEQUENCE_A")(X)

    X = tf.keras.layers.Dense(128, name="FC")(X)

    X = tf.keras.layers.Dense(4, name="YNODE")(X)

    model = tf.keras.Model(inputs=X_input, outputs=X, name=MODEL_NAME)
    return model


class PriceData:
    def load_price_data(self, filename=None):
        try:
            if filename is not None:
                try:
                    d_frame = pd.read_csv(filename)

                    price_data = d_frame.filter(items=["high", "low", "open", "close"])
                    self.normalizer.fit_transform(price_data)

                    return price_data
                except Exception as error:
                    print(error)
                    return False
            else:
                return False
        except Exception as error:
            print("--from bot_utils:", error)
            return False

    def load_dataset(self, filename=None, interval_window=60, validation_split=0.2):
        try:
            if filename is not None:
                try:
                    d_frame = pd.read_csv(filename)

                    price_data = d_frame.filter(items=["high", "low", "open", "close"])

                    unnormalized_ds = price_data.values

                    normalized_ds = self.normalizer.fit_transform(unnormalized_ds)

                    training_samples = int((1 - validation_split) * int(unnormalized_ds.shape[0]))
                    self.training_samples = training_samples

                    training_data = normalized_ds[:training_samples]
                    testing_data = normalized_ds[int(training_samples - interval_window):]

                    x_train, y_train, x_test = list(), list(), list()
                    for i in range(interval_window, len(training_data)):
                        x_train.append(
                            [training_data[i - interval_window:i, 0], training_data[i - interval_window:i, 1],
                             training_data[i - interval_window:i, 2], training_data[i - interval_window:i, 3]])
                        y_train.append(
                            [training_data[i, 0], training_data[i, 1], training_data[i, 2], training_data[i, 3]])

                    y_test = [[unit for unit in data] for data in unnormalized_ds[training_samples:]]
                    for m in range(interval_window, len(testing_data)):
                        x_test.append([testing_data[m - interval_window:m, 0], testing_data[m - interval_window:m, 1],
                                       testing_data[m - interval_window:m, 2], testing_data[m - interval_window:m, 3]])

                    x_train, y_train, x_test, y_test = np.array(x_train), np.array(y_train), np.array(x_test), np.array(
                        y_test)
                    x_train = np.reshape(x_train, newshape=(x_train.shape[0], x_train.shape[2], x_train.shape[1]))
                    x_test = np.reshape(x_test, newshape=(x_test.shape[0], x_test.shape[2], x_test.shape[1]))

                    return x_train, y_train, x_test, y_test
                except Exception as error:
                    print(error)
                    return False
            else:
                return False
        except Exception as e:
            print(e)
            return False

    def __init__(self):
        self.training_samples = None
        self.normalizer = MinMaxScaler(feature_range=(0, 1))
