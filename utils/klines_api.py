import pandas as pd
from binance.spot import Spot as server


def live_data(symbol):
    link = server()
    data = link.klines(symbol=symbol, interval="1m", limit=1)
    # link.close_connection()
    dframe = pd.DataFrame(data)
    dframe.columns = ['openTime', 'open', 'high', 'low', 'close', 'volume', 'closeTime',
                      'quoteVolume', 'numTrades', 'buyBaseAssetV', 'buyQuoteAssetV', 'Ignore']
    dframe = dframe.filter(items=["close"]).astype(float)
    data = float(dframe.values)
    return data


class PriceUpdates:
    def refresh(self):
        try:
            client = server()
            klines = client.klines(symbol=self.symbol,
                                   interval=str(str(self.interval) + self.interval_units),
                                   limit=self.limit)
            # client.close_connection()
            dframe = pd.DataFrame(klines)
            dframe.columns = ['openTime', 'open', 'high', 'low', 'close', 'volume', 'closeTime',
                              'quoteVolume', 'numTrades', 'buyBaseAssetV', 'buyQuoteAssetV', 'Ignore']
            dframe = dframe.filter(items=["open", "high", "low", "close"]).astype(float)
            dframe.to_csv(self.filename, index=False, header=True)
            return True
        except Exception as error:
            print("Fatal error! Could not refresh previous data!", error)
            return False

    def __init__(self, filename=None, symbol="BTCUSDT", limit=1000, interval=1, interval_units="m"):
        self.symbol = symbol
        self.filename = filename
        self.limit = limit
        self.interval = interval
        self.interval_units = interval_units

x = PriceUpdates()
x.refresh()