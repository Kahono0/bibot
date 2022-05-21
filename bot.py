import time
import io
from utils.klines_api import *
from utils.bot_utils import *
from utils.binance_api import *


STOCKS_DIR = "stocks/"
MODELS_DIR = "models/ai-"
HISTORY_DIR = "logs/"
WINDOW_LENGTH = 60


def data_file(filename=None):
    return str(filename) + ".csv"


def model_file(filename=None):
    return str(filename) + ".h5"


ds_loader = PriceData()
LOGIN_TIMESTAMP = time.time()


def pred_to_history(value=None):
    if value is None:
        return False
    else:
        try:
            value = float(value)
            if os.path.exists(HISTORY_DIR + "history.log"):
                f = io.open(HISTORY_DIR + "history.log", mode="a", encoding="UTF-8")
            else:
                f = io.open(HISTORY_DIR + "history.log", mode="w", encoding="UTF-8")
            f.write(str(value) + "\n")
            f.close()
            return True
        except Exception as error:
            return False


class BotTrader:
    def one_step_predict(self, default_steps=1):
        ready = True
        if not self.model_ready:
            return False

        if os.path.exists(data_file(self.symbol)):
            if not trash_file(STOCKS_DIR + data_file(self.symbol)):
                ready = False

        price_data = PriceUpdates(
            filename=STOCKS_DIR + data_file(self.symbol),
            symbol=self.symbol,
            limit=1000,
            interval=1,
            interval_units="m"
        )
        if not price_data.refresh() == True:
            ready = False

        if not ready:
            return False
        elif ready:
            latest_data = ds_loader.load_price_data(STOCKS_DIR + data_file(self.symbol))
            if type(latest_data) == 'bool':
                print("--from -@-bot- unable to load latest_price-data: {main}")
                return False
            else:
                history = latest_data[-(WINDOW_LENGTH * default_steps):].values
                history = ds_loader.normalizer.transform(history)
                window_length_slices = np.array(np.split(history, default_steps))
                try:
                    predictions = self.model.predict(window_length_slices)
                    predictions = ds_loader.normalizer.inverse_transform(predictions)
                    prediction_frame = parse_outputs(predictions)
                    custom_forecast = float(np.mean([prediction_frame.mean().mean(), prediction_frame.mean()["close"]]))
                    return custom_forecast
                except Exception as fatalError:
                    print(
                        f"\n--warning-from--bot: a problem occured while try to make a prediction. |help|--> {fatalError}\n\n")
                    return False

    def learn(self, summary=True, silent=False):
        ready = True
        if os.path.exists(data_file(self.symbol)):
            if not trash_file(STOCKS_DIR + data_file(self.symbol)):
                ready = False

        price_data = PriceUpdates(
            filename=STOCKS_DIR + data_file(self.symbol),
            symbol=self.symbol,
            limit=1000,
            interval=1,
            interval_units="m"
        )
        if not price_data.refresh() == True:
            ready = False
        if ready:
            if not silent:
                print("--@-bot--preparing to learn...\n")
            try:
                x_train, y_train, x_test, y_test = ds_loader.load_dataset(
                    filename=STOCKS_DIR + data_file(self.symbol),
                    interval_window=WINDOW_LENGTH,
                    validation_split=0.2
                )
                if os.path.exists(model_file(MODELS_DIR + self.symbol)):
                    try:
                        if not silent:
                            print("--@bot--relearning in a few...")
                        this_model = tf.keras.models.load_model(filepath=model_file(MODELS_DIR + self.symbol))
                    except:
                        this_model = PriceAnalyst(x_train.shape[1:],
                                                  f"AI...{self.symbol}....PAIR..PRICE..FORECASTER..MODEL")
                else:
                    this_model = PriceAnalyst(x_train.shape[1:],
                                              f"AI...{self.symbol}....PAIR..PRICE..FORECASTER..MODEL")
                if summary:
                    this_model.summary()
                this_model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00178125),
                    loss="mean_squared_error"
                )
                if not silent:
                    print("--learning...\n")
                this_model.fit(
                    x_train,
                    y_train,
                    batch_size=20,
                    epochs=100,
                    verbose=0,
                    callbacks=[AnalystCallbacks]
                )
                this_model.save(model_file(MODELS_DIR + self.symbol))
                if not silent:
                    print("\n", "-" * 20, "done training-->", "-" * 20, "\n")
                return True
            except Exception as error:
                print("from --bot: I got a problem gathering price data csv file", error)
                return False

    def __init__(self, base_asset="BTC", quote_asset="USDT"):
        self.symbol = str(base_asset) + str(quote_asset)
        self.base_asset = base_asset
        self.quote_asset = quote_asset
        self.model_ready = True
        if not self.learn(summary=False) == True:
            self.model_ready = False
        else:
            self.model = tf.keras.models.load_model(filepath=model_file(MODELS_DIR + self.symbol))
            self.model_ready = True
