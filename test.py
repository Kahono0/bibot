from turtle import color
from bot import BotTrader
from requests import get
import matplotlib.pyplot as plt

symbol = "BNBUSDT"
bt = BotTrader("BNB","USDT")

def proof(sym):
    y = get("https://api.binance.com/api/v3/ticker/price?symbol=" + sym)
    return float(y.json()['price'])
arr1 = []
arr2 = []

def check():
    a = bt.one_step_predict()
    # arr2.append(a)
    w = 0
    for h in range(1000):
        # time.sleep()
        x = proof(symbol)
        print("Correct:", x)
        arr1.append(x)
        print("Predict:", a)
        a = bt.one_step_predict()
        arr2.append(a)
def pt():
    plt.plot(arr1,'r', arr2,'b')
    plt.show()
check()
pt()
