# class BotWarning:
#     def transactions(description=''):
#         return f"{description}"
#
#     def binanceException(description=''):
#         return f"{description}"
#
#
# import math
# import asyncio
# from binance import AsyncClient
# from binance.exceptions import BinanceAPIException
#
# """*******************************************************************************************************************"""
# """*******************************************************************************************************************"""
# TESTNET = True
# API_KEY = None
# API_SECRET = None
# TESTNET_API_KEY = None
# TESTNET_API_SECRET = None
# """*******************************************************************************************************************"""
# """*******************************************************************************************************************"""
#
#
# async def bal(base=None, quote=None):
#     if TESTNET:
#         client = await AsyncClient.create(api_key=TESTNET_API_KEY, api_secret=TESTNET_API_SECRET, testnet=True)
#     else:
#         client = await AsyncClient.create(api_key=API_KEY, api_secret=API_SECRET, testnet=False)
#
#     pair_symbol = str(base + quote)
#     balances = await client.get_account()
#     balances = balances["balances"]
#     base_balance, quote_balance = 0, 0
#     for data in balances:
#         if data["asset"] == base:
#             base_balance = float(data["free"])
#         elif data["asset"] == quote:
#             quote_balance = float(data["free"])
#     await client.close_connection()
#     return base_balance, quote_balance
#
#
# def float_precision(f, n):
#     n = int(math.log10(1 / float(n)))
#     f = math.floor(float(f) * 10 ** n) / 10 ** n
#     f = "{:0.0{}f}".format(float(f), n)
#     return str(int(f)) if int(n) == 0 else f
#
#
# def push_receipt(data=dict()):
#     fills = data["fills"][0]
#     print(f"\n\n {data['symbol']} {data['side']} ORDER PLACED")
#     print("-" * 30, "\n")
#     print(f"ORDER-ID: {data['orderId']} STATUS: {data['status']} STAMP: {data['transactTime']}")
#     print(f"QUANTITY: {fills['qty']} \nPRICE: {fills['price']} \nTYPE: {data['type']}")
#     print("-" * 30, "\n\n")
#     return True
#
#
# async def buy(base=None, quote=None, max_amount='auto'):
#     if TESTNET:
#         client = await AsyncClient.create(api_key=TESTNET_API_KEY, api_secret=TESTNET_API_SECRET, testnet=True)
#     else:
#         client = await AsyncClient.create(api_key=API_KEY, api_secret=API_SECRET, testnet=False)
#
#     pair_symbol = str(base + quote)
#     balances = await client.get_account()
#     balances = balances["balances"]
#     quote_balance = 0
#     for data in balances:
#         if data["asset"] == quote:
#             quote_balance = float(data["free"])
#     if not quote_balance > 0:
#         raise Warning(
#             BotWarning.transactions(quote + " balance is too low in your account! Couldn't buy " + pair_symbol))
#
#     symbol_info = await client.get_symbol_info(pair_symbol)
#     if symbol_info["isSpotTradingAllowed"] == True:
#         base_asset_precision = int(symbol_info["baseAssetPrecision"])
#         quantity, maxQty, minQty, minPrice, maxPrice, tickSize = 0, 0, 0, 0, 0, 0
#         notional_value = 0
#         avg_price = await client.get_avg_price(symbol=pair_symbol)
#         avg_price = float(avg_price["price"])
#         step_size = 0
#         for pfilter in symbol_info["filters"]:
#             if pfilter["filterType"] == "PRICE_FILTER":
#                 minPrice, maxPrice, tickSize = float(pfilter["minPrice"]), float(pfilter["maxPrice"]), float(
#                     pfilter["tickSize"])
#             elif pfilter["filterType"] == "MARKET_LOT_SIZE":
#                 maxQty, minQty = float(pfilter["maxQty"]), float(pfilter["minQty"])
#             elif pfilter["filterType"] == "MIN_NOTIONAL":
#                 notional_value = float(pfilter["minNotional"])
#             elif pfilter["filterType"] == "LOT_SIZE":
#                 step_size = float(pfilter["stepSize"])
#
#         if quote_balance < notional_value:
#             raise Warning(BotWarning.transactions(
#                 quote + " balance is too less than the minimum required of {notional_value}. Couldn't buy " + pair_symbol))
#         else:
#             if not max_amount == 'auto':
#                 max_amount = float(max_amount)
#                 if max_amount < notional_value:
#                     raise Warning(BotWarning.transactions(
#                         quote + " balance is too less than the minimum required of {notional_value}. Couldn't buy " + pair_symbol))
#                 else:
#                     quantity = (max_amount / avg_price)
#             else:
#                 quantity = (quote_balance / avg_price)
#             if quantity > minQty and quantity <= maxQty:
#                 try:
#                     quantity = float_precision(quantity, step_size)
#                     buy_order = await client.order_market_buy(
#                         symbol=pair_symbol,
#                         quantity=quantity)
#                     push_receipt(buy_order)
#                 except BinanceAPIException as errorBuying:
#                     raise Warning(BotWarning.binanceException("Fatal error: " + str(errorBuying)))
#             else:
#                 raise Warning(BotWarning.transactions(
#                     str(quantity) + " of " + base + " is either more than the maximum tradable or less than the minimum allowable."))
#     else:
#         raise Warning(BotWarning.transactions(pair_symbol + " is not tradable on SPOT market"))
#
#     await client.close_connection()
#
#
# async def sell(base=None, quote=None, max_amount='auto'):
#     if TESTNET == True:
#         client = await AsyncClient.create(api_key=TESTNET_API_KEY, api_secret=TESTNET_API_SECRET, testnet=True)
#     else:
#         client = await AsyncClient.create(api_key=API_KEY, api_secret=API_SECRET, testnet=False)
#
#     pair_symbol = str(base + quote)
#     balances = await client.get_account()
#     balances = balances["balances"]
#     base_balance = 0
#     for data in balances:
#         if data["asset"] == base:
#             base_balance = float(data["free"])
#     if not base_balance > 0:
#         raise Warning(
#             BotWarning.transactions(base + " balance is too low in your account! Couldn't sell " + pair_symbol))
#
#     symbol_info = await client.get_symbol_info(pair_symbol)
#     if symbol_info["isSpotTradingAllowed"] == True:
#         base_asset_precision = int(symbol_info["baseAssetPrecision"])
#         quantity, maxQty, minQty, minPrice, maxPrice, tickSize = 0, 0, 0, 0, 0, 0
#         notional_value = 0
#         avg_price = await client.get_avg_price(symbol=pair_symbol)
#         avg_price = float(avg_price["price"])
#         step_size = 0
#         for pfilter in symbol_info["filters"]:
#             if pfilter["filterType"] == "PRICE_FILTER":
#                 minPrice, maxPrice, tickSize = float(pfilter["minPrice"]), float(pfilter["maxPrice"]), float(
#                     pfilter["tickSize"])
#             elif pfilter["filterType"] == "MARKET_LOT_SIZE":
#                 maxQty, minQty = float(pfilter["maxQty"]), float(pfilter["minQty"])
#             elif pfilter["filterType"] == "MIN_NOTIONAL":
#                 notional_value = float(pfilter["minNotional"])
#             elif pfilter["filterType"] == "LOT_SIZE":
#                 step_size = float(pfilter["stepSize"])
#         if base_balance < (notional_value / avg_price):
#             raise Warning(BotWarning.transactions(base + " balance of " + str(
#                 int((notional_value / avg_price))) + " is too low. Couldn't sell " + pair_symbol))
#         else:
#             if not max_amount == 'auto':
#                 max_amount = float(max_amount)
#                 if max_amount < (notional_value / avg_price):
#                     raise Warning(BotWarning.transactions(base + " balance is too low. Couldn't sell " + pair_symbol))
#                 else:
#                     quantity = max_amount
#             else:
#                 quantity = base_balance
#             if quantity > minQty and quantity <= maxQty:
#                 try:
#                     quantity = float_precision(quantity, step_size)
#                     sell_order = await client.order_market_sell(
#                         symbol=pair_symbol,
#                         quantity=quantity)
#                     push_receipt(sell_order)
#                 except BinanceAPIException as errorSelling:
#                     raise Warning(BotWarning.binanceException("Fatal error: " + errorSelling))
#             else:
#                 raise Warning(BotWarning.transactions(
#                     str(quantity) + " of " + base + " is either more than the maximum tradable or less than the minimum allowable."))
#     else:
#         raise Warning(BotWarning.transactions(pair_symbol + " is not tradable on SPOT market"))
#
#     await client.close_connection()
