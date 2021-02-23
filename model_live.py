import pickle
import pandas as pd
import numpy as np
from binance.client import Client
from model_tools import processFrame,loadDataFrame,addFeatures,recordLiveResults

CLIENT = Client("<>","<>")

def truncate(n, decimals=0):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier

validation_frame = loadDataFrame(pull_date='July 27, 2020')

validation_frame = validation_frame.drop(len(validation_frame)-1,axis=0)
latest_entry_info = validation_frame.iloc[[-1]]
validation_frame = validation_frame.drop(['ignored','openTime','closeTime'],axis=1)
validation_frame = addFeatures(validation_frame)

X_set = np.array(validation_frame)
loaded_model = pickle.load(open('pickle_files/pipe_line.pkl', 'rb'))
prediction = loaded_model.predict_proba(X_set)[-1]

balance_USDT = CLIENT.get_asset_balance(asset='USDT')
balance_BTC = CLIENT.get_asset_balance(asset='BTC')

money_USDT = float(balance_USDT['free'])
money_BTC = float(balance_BTC['free'])

ticker_info = CLIENT.get_ticker(symbol="BTCUSDT")
last_price = float(ticker_info['askPrice'])
bid_price = float(ticker_info['bidPrice'])

USDT_quantity = truncate(money_USDT/last_price,6)
BTC_quantity = truncate(money_BTC,6)

if prediction[1] >= 0.6 and USDT_quantity > BTC_quantity:
    order = CLIENT.order_market_buy(
        symbol='BTCUSDT',
        quantity=USDT_quantity
        )
elif prediction[1] < 0.6 and BTC_quantity > USDT_quantity:
    order = CLIENT.order_market_sell(
        symbol='BTCUSDT',
        quantity=BTC_quantity
        )
recordLiveResults(latest_entry_info,prediction)
