from futu import *
import talib as ta
import time
import numpy as np
import pandas as pd
import sys
import os
class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
# import matplotlib.pyplot as plt
# import scipy.signal as fp

TRADING_ENVIRONMENT = TrdEnv.SIMULATE  # 交易环境：真实 / 模拟
TRADING_SECURITY = list()  # 交易标的
FUTUOPEND_ADDRESS = '127.0.0.1'  # FutuOpenD 监听地址
FUTUOPEND_PORT = 11111  # FutuOpenD 监听端口
trade_context = OpenHKTradeContext(host=FUTUOPEND_ADDRESS, port=FUTUOPEND_PORT,
                                   security_firm=SecurityFirm.FUTUSECURITIES)  # 交易对象，根据交易标的修改交易对象类型

a = time.localtime()
# pip install TA-Lib 技术分析库
# pip install futu-api 富途api
# https://openapi.futunn.com/futu-api-doc/quote/sub.html#5009
# https://openapi.futunn.com/futu-api-doc/quick/strategy-sample.html#7871 如需盘中实时运行策略并交易可参考
quote_ctx = OpenQuoteContext(host=FUTUOPEND_ADDRESS, port=FUTUOPEND_PORT)  # 行情对象

simple_filter,simple_filter1 = SimpleFilter(),SimpleFilter()
# 市值条件
simple_filter.stock_field = StockField.MARKET_VAL
simple_filter.filter_min = 18000000000
simple_filter.filter_max = 180000000000000
simple_filter.is_no_filter = False

simple_filter1.stock_field = StockField.MARKET_VAL
simple_filter1.filter_min = 18000000000
simple_filter1.filter_max = 180000000000000
simple_filter1.is_no_filter = False
stocklist,ussklist = list(),list()
nBegin = 0
last_page = False
ret_list = list()
while not last_page:
    nBegin += len(ret_list)
    ret, ls = quote_ctx.get_stock_filter(market=Market.HK, filter_list=[simple_filter], begin=nBegin)  # 对香港市场的股票做简单和财务筛选
    ret1, ls1 = quote_ctx.get_stock_filter(market=Market.US, filter_list=[simple_filter1],
                                         begin=nBegin)  # 对香港市场的股票做简单和财务筛选
    if ret == RET_OK:
        last_page, all_count, ret_list = ls
        for item in ret_list:
            # print(item.stock_code)  # 取股票代码
            stocklist.append(item.stock_code)
    else:
        print('error: ', ls)
    if ret1 == RET_OK:
        last_page, all_count, ret_list = ls1
        for item in ret_list:
            # print(item.stock_code)  # 取股票代码
            ussklist.append(item.stock_code)
    else:
        print('error: ', ls1)
sys.stdout = Logger('log.txt')
print(a)
print('初步筛选的HK股票为：',stocklist)
print('初步筛选的US股票为：',ussklist)
# 确定bs list
def bs_list(stocklist):
    b_list, s_list = [], []
    b2_list = list()
    s2_list = list()
    cdt1, cdt2, cdt3, cdt4, cdt6, cdt7, cdt8, cdt9, cdt11, =[], [], [], [], [], [], [], [], []
    cdt12, cdt13, cdt14, cdt16, cdt18, cdt19 = [], [], [], [], [], []
    for code in stocklist:
        num = int(stocklist.index(code)) + 1
        print('目前进行第', num, '只股票筛选，总共股票数量为：', len(stocklist))
        if num % 20 == 0:
            print('api暂时中断,程序暂停30秒.')
            time.sleep(30)
        try:
            ret, data, page_req_key = quote_ctx.request_history_kline(code, ktype=KLType.K_DAY,
                                                                      max_count=500)  # 每页500个，请求第一页
        except:
            print('error:', data)
        # print(data)
        if ret == RET_OK:
            # Buy  # Condition1:sar 指标
            try:
                sar = ta.SAR(data['high'].values, data['low'].values)
                if data['high'].values[-1] > sar[-1] and data['high'].values[-2] < sar[-2]:
                    b_list.append(data['code'][0])
                    cdt1.append(data['code'][0])
                # Se11#Condition11  sar指標線cross High SAR
                if data['high'].values[-1] < sar[-1] and data['high'].values[-2] > sar[-2]:
                    s_list.append(data['code'][0])
                    cdt11.append(data['code'][0])
            except:
                print(data['code'][0], 'sar指标筛选不成功！')
            # Buy  # Condition2:Close>前6天的Close，建續滿足此保件13天
            # Se11#Condition12:Close<前6天的Close，速續滿足此修件13天
            c2 = []
            c12 = []
            dataclose = data['close'].values
            try:
                for i in range(13):
                    if dataclose[-13 + i] > dataclose[-19 + i]:
                        c2.append(1)
                    if dataclose[-13 + i] < dataclose[-19 + i]:
                        c12.append(1)
                if c2 == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]:
                    b_list.append(data['code'][0])
                    cdt2.append(data['code'][0])
                if c12 == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]:
                    s_list.append(data['code'][0])
                    cdt12.append(data['code'][0])
            except:
                print(data['code'][0], '13天 close筛选不成功！')

            # Buy# Condition 4:CS cross 0值 (CS=(close-EMA20)/EMA20*100)
            # Sell# Condition 14:0值cross CS (CS=(close-EMA20)/EMA20*100)
            try:
                EMA20 = ta.EMA(dataclose, 20)
                CS1 = (dataclose[-1] - EMA20[-1]) / EMA20[-1] * 100
                CS2 = (dataclose[-2] - EMA20[-2]) / EMA20[-2] * 100

                if CS1 > 0 and CS2 < 0:
                    cdt4.append(data['code'][0])

                if CS1 < 0 and CS2 > 0:
                    cdt14.append(data['code'][0])
            except:
                print(data['code'][0], 'EMA5 & MA10筛选不成功！')
            # Buy# Condition 6: 週期形态突破上限線最大3個月
            # Buy# Condition 7: Volume Contraction Pattern(VCP) 突破一刻最大3個月
            # Sell# Condition 16: 週期形态突破下限線最大3個月
            if len(dataclose) >= 60:
                n1, n2, n3 = 0, 0, 0
                for i in range(10):
                    maxnum = [max(dataclose[-60 - i:40 - i]), max(dataclose[-40 - i:20 - i]),
                              max(dataclose[-20 - i:-2])]
                    minnum = [min(dataclose[-60 - i:40 - i]), min(dataclose[-40 - i:20 - i]),
                              min(dataclose[-20 - i:-2])]
                    maxavg, minavg, m1, m2 = np.mean(maxnum), np.mean(minnum), list(), list()
                    if min(maxnum) > max(minnum):
                        for i in maxnum:
                            if 0.995 < (i / maxavg) < 1.005:
                                m1.append(1)
                        for i in minavg:
                            if 0.995 < (i / minavg) < 1.005:
                                m2.append(1)
                        if m1 == m2 and m2 == [1, 1, 1]:
                            if dataclose[-1] > maxavg and n1 == 0:
                                cdt6.append(data['code'][0])  # 週期形态突破上限線
                                n1 = 1
                            if dataclose[-1] < minavg and n2 == 0:
                                cdt16.append(data['code'][0])  # 週期形态突破下限線
                                n2 = 1
                        if m1 == [1, 1, 1] and minnum[0] < minnum[1] < minnum[2] and dataclose[
                            -1] > 0.8 * maxavg and n3 == 0:  # Volume Contraction Pattern(VCP) 4点定位+3低点确认
                            cdt7.append(data['code'][0])
            try:
                open, high, low, close = data['open'].values[-1], data['high'].values[-1], data['low'].values[-1], \
                                         data['close'].values[-1]
                # integer = ta.CDLHAMMER(data)
                # integer2 = ta.CDLGRAVESTONEDOJI(data)
                integer = ta.CDLHAMMER(open, high, low, close)
                integer2 = ta.CDLGRAVESTONEDOJI(open, high, low, close)
                rsi1 = ta.RSI(dataclose, timeperiod=6)
                EMA5 = ta.EMA(dataclose, 5)
                MA10 = ta.MA(dataclose, 10)
                # Buy  # Condition9:出现HAMMER Pattern且RSI<=35
                if rsi1 <= 35 and integer >= 60:
                    b_list.append(data['code'][0])
                    cdt9.append(data['code'][0])
                # Se11#Condition19:出现 GRAVESTONEDOJIPattern 且RSI>=65
                if rsi1 >= 65 and integer2 >= 60:
                    s_list.append(data['code'][0])
                    cdt19.append(data['code'][0])
                # Buy# Condition 3:EMA5 cross MA10
                if EMA5[-1] > MA10[-1] and EMA5[-2] < MA10[-2]:
                    cdt3.append(data['code'][0])
                # Sell# Condition 13:MA10 cross EMA5
                if rsi1 >= 65 and integer2 >= 60:
                    cdt13.append(data['code'][0])
            except:
                print(data['code'][0], 'RSI指标筛选不成功！')

            try:
                # Buy  # Condition8:MACD日線图出现黄金交叉，MACD周及月線图呈上趣势(DIF>DEA)
                # Se1l#Condition18:MACD日線图出现死亡交叉，MACD周及月線图呈下趣势(DEA>DIF)

                ret, Mdata, page_req_key = quote_ctx.request_history_kline(code, ktype=KLType.K_MON,
                                                                           max_count=500)
                ret, Wdata, page_req_key = quote_ctx.request_history_kline(code, ktype=KLType.K_WEEK,
                                                                           max_count=500)
                # macd 参数
                MACD, DIF, DEA = ta.MACD(dataclose, fastperiod=12, slowperiod=26, signalperiod=9)
                MMACD, MDIF, MDEA = ta.MACD(Mdata['close'].value, fastperiod=12, slowperiod=26, signalperiod=9)
                WMACD, WDIF, WDEA = ta.MACD(Wdata['close'].value, fastperiod=12, slowperiod=26, signalperiod=9)
                if MACD[-2] < 0 and MACD[-1] > 0 and MDIF[-1] > MDEA[-1] and WDIF[-1] > WDEA[-1]:
                    b_list.append(data['code'][0])
                    cdt8.append(data['code'][0])
                if MACD[-2] > 0 and MACD[-1] < 0 and MDIF[-1] < MDEA[-1] and WDIF[-1] < WDEA[-1]:
                    s_list.append(data['code'][0])
                    cdt18.append(data['code'][0])
            except:
                print(data['code'][0], 'MACD指标筛选不成功！')
    print('当前选出符合BUY条件的股票列表为：', b_list)
    print('当前选出符合SELL条件的股票列表为：', s_list)
    # 觸發多單情況1: Condition 1 +3 +4 +(下单需要判定:当前的Bar 升幅<3%)
    # 觸發多單情況2: Condition 2 +3 +4 +(下单需要判定:当前的Bar 升幅<3%)
    # 觸發多單情況3: Condition 6+(下单需要判定:当前的Bar 升幅<5%)
    # 觸發多單情況4: Condition 7+(下单需要判定:当前的Bar 升幅<5%)
    #
    # 觸發空單情況1: Condition 11 +13 +14 +(下单需要判定:当前的Bar 跌幅<3%)
    # 觸發空單情況2: Condition 12 +13 +14 +(下单需要判定:当前的Bar 跌幅<3%)
    # 觸發空單情況3: Condition 16+(下单需要判定:当前的Bar 跌幅<5%)
    #
    # 加入自选股列表:
    # 1.每週一加入分組前先清空分組
    # 2.出現以上多單情況1或2或3或4將股票列表加入自选股列表分組”BUY2”中
    # 3.出現以上空單情況1或2或3將股票列表加入自选股列表分組”SELL2”中
    b3, b5, s3, s5 = [], [], [], []
    for i in cdt6:
        b2_list.append(i)
        b5.append(i)
    for i in cdt7:
        b2_list.append(i)
        b5.append(i)
    for i in cdt1:
        if (i in cdt3) and (i in cdt4):
            b2_list.append(i)
            b3.append(i)
    for i in cdt2:
        if (i in cdt3) and (i in cdt4):
            b2_list.append(i)
            b3.append(i)

    for i in cdt16:
        s2_list.append(i)
        s5.append(i)
    for i in cdt11:
        if (i in cdt13) and (i in cdt14):
            s2_list.append(i)
            s3.append(i)
    for i in cdt12:
        if (i in cdt13) and (i in cdt14):
            s2_list.append(i)
            s3.append(i)
    return b_list, s_list, b2_list, s2_list, b3, s3
    # return b_list, s_list, b2_list, s2_list, b3, s3,b5,s5


# 修改自选股列表
def modify(fx, numlist):
    print('执行自选股操作！')
    fx = str(fx)
    print('执行', fx, '自选股操作,结果为：')
    ret, data = quote_ctx.modify_user_security(fx, ModifyUserSecurityOp.ADD, numlist)
    if ret == RET_OK:
        print(data)
    else:
        print('error:', data)


if time.strftime("%A", a) == 'Monday':

    data = quote_ctx.get_user_security("SELL")[1]
    data1 = quote_ctx.get_user_security("BUY")[1]
    data3 = quote_ctx.get_user_security("SELL2")[1]
    data4 = quote_ctx.get_user_security("BUY2")[1]
    try:
        selllist = data['code'].values.tolist()  # 转为 list
        quote_ctx.modify_user_security('SELL', ModifyUserSecurityOp.DEL, selllist)
        buylist = data1['code'].values.tolist()
        quote_ctx.modify_user_security('BUY', ModifyUserSecurityOp.DEL, buylist)
        selllist = data3['code'].values.tolist()  # 转为 list
        quote_ctx.modify_user_security('SELL2', ModifyUserSecurityOp.DEL, selllist)
        buylist = data4['code'].values.tolist()
        quote_ctx.modify_user_security('BUY2', ModifyUserSecurityOp.DEL, buylist)
    except:
        print('error:', data, data1)


# 觸發多單情況1: Condition 1 +3 +4 +(下单需要判定:当前的Bar 升幅<3%)
# 觸發多單情況2: Condition 2 +3 +4 +(下单需要判定:当前的Bar 升幅<3%)
# 觸發多單情況3: Condition 6+(下单需要判定:当前的Bar 升幅<5%)
# 觸發多單情況4: Condition 7+(下单需要判定:当前的Bar 升幅<5%)
# 觸發空單情況1: Condition 11 +13 +14 +(下单需要判定:当前的Bar 跌幅<3%)
# 觸發空單情況2: Condition 12 +13 +14 +(下单需要判定:当前的Bar 跌幅<3%)
# 觸發空單情況3: Condition 16+(下单需要判定:当前的Bar 跌幅<5%)
# 展示订单回调
def show_order_status(data):
    order_status = data['order_status'][0]
    order_info = dict()
    order_info['代码'] = data['code'][0]
    order_info['价格'] = data['price'][0]
    order_info['方向'] = data['trd_side'][0]
    order_info['数量'] = data['qty'][0]
    print('【订单状态】', order_status, order_info)


# 获取持仓数量
def get_holding_position(code):
    holding_position = 0
    ret, data = trade_context.position_list_query(code=code, trd_env=TRADING_ENVIRONMENT)
    if ret != RET_OK:
        print('获取持仓数据失败：', data)
        return None
    else:
        if data.shape[0] > 0:
            holding_position = data['qty'][0]
        print('【持仓状态】 {} 的持仓数量为：{}'.format(TRADING_SECURITY, holding_position))
    return holding_position


# 获取一档摆盘的 ask1 和 bid1
def get_ask_and_bid(code):
    ret, data = quote_ctx.get_order_book(code, num=1)
    if ret != RET_OK:
        print('获取摆盘数据失败：', data)
        return None, None
    return data['Ask'][0][0], data['Bid'][0][0]


# 开仓函数
def open_position(code, side):
    # 获取摆盘数据
    ask, bid = get_ask_and_bid(code)

    # 计算下单量
    open_quantity = calculate_quantity()

    # 判断购买力是否足够
    if is_valid_quantity(TRADING_SECURITY, open_quantity, ask):
        # 下单
        ret, data = trade_context.place_order(price=ask, qty=open_quantity, code=code, trd_side=side,
                                              order_type=OrderType.NORMAL, trd_env=TRADING_ENVIRONMENT)
        if ret != RET_OK:
            print('开仓失败：', data)
        if ret == RET_OK:
            show_order_status(data)
            cdata=time.strftime("%Y-%m-%d", time.localtime())
            code=code
            df=pd.DataFrame({'cdata':cdata,'code':code,'qty':open_quantity,'trd_side':side})
            df1=pd.read_csv('trade.csv')
            df=pd.concat([df, df1], axis=0)
            df.to_csv('trade.csv')

    else:
        print('下单数量超出最大可买数量。')


# 平仓函数
def close_position(code, quantity, side):
    # 获取摆盘数据
    ask, bid = get_ask_and_bid(code)
    # 检查平仓数量
    if quantity == 0:
        print('无效的下单数量。')
        return False
    # 平仓
    ret, data = trade_context.place_order(price=bid, qty=quantity, code=code, trd_side=side,
                                          order_type=OrderType.NORMAL, trd_env=TRADING_ENVIRONMENT)
    if ret != RET_OK:
        print('平仓失败：', data)
        return False
    return True


# 计算下单数量
def calculate_quantity():
    price_quantity = 0
    # 使用最小交易量
    ret, data = quote_ctx.get_market_snapshot([TRADING_SECURITY])
    if ret != RET_OK:
        print('获取快照失败：', data)
        return price_quantity
    price_quantity = data['lot_size'][0]
    return price_quantity

# 判断购买力是否足够
def is_valid_quantity(code, quantity, price):
    ret, data = trade_context.acctradinginfo_query(order_type=OrderType.NORMAL, code=code, price=price,
                                                   trd_env=TRADING_ENVIRONMENT)
    if ret != RET_OK:
        print('获取最大可买可卖失败：', data)
        return False
    max_can_buy = data['max_cash_buy'][0]
    max_can_sell = data['max_sell_short'][0]
    if quantity > 0:
        return quantity < max_can_buy
    elif quantity < 0:
        return abs(quantity) < max_can_sell
    else:
        return False


def trade(fx, slist, n):
    fx1 = ''
    # pwd_unlock = '123456'
    trd_ctx = trade_context
    # ret, data = trd_ctx.unlock_trade(pwd_unlock)  # 先解锁交易
    qr = 3
    ret = RET_OK
    if ret == RET_OK:
        for stock in slist:
            ret_sub, err_message = quote_ctx.subscribe(stock, [SubType.QUOTE], subscribe_push=False)
            # 先订阅 K 线类型。订阅成功后 FutuOpenD 将持续收到服务器的推送，False 代表暂时不需要推送给脚本
            if ret_sub == RET_OK:  # 订阅成功
                ret, data = quote_ctx.get_stock_quote(stock)  # 获取订阅股票报价的实时数据
                if ret == RET_OK:
                    zf = (data['last_price'] / data['open_price'] - 1) * 100
                    if n == 3:
                        if abs(zf) < n:
                            qr = 1
                    elif n == 5:
                        if abs(zf) < n:
                            qr = 1
                    else:
                        qr = 0
                else:
                    print('error:', data)
            else:
                print('subscription failed', err_message)
            ret, data = trd_ctx.position_list_query()
            if ret == RET_OK and qr == 1:
                holdstock = data['code'].values.to_list()
                if stock in holdstock:
                    if data['position_side'][holdstock[holdstock.index(stock)]] == 'LONG':
                        fx1 = TrdSide.BUY
                    if data['position_side'][holdstock[holdstock.index(stock)]] == 'SHORT':
                        fx1 = TrdSide.SELL_SHORT
                    if fx1 != '':
                        if fx1 != fx:
                            if fx1 == TrdSide.SELL_SHORT:
                                side = TrdSide.BUY_BACK
                            elif fx1 == TrdSide.BUY:
                                side = TrdSide.SELL
                            else:
                                side = TrdSide.NONE
                            quantity = get_holding_position(stock)
                            close_position(stock, quantity, side)  # 平仓
                            open_position(stock, fx)  # 开仓
                else:
                    open_position(stock, fx)  # 开仓
            else:
                print('position_list_query error: ', data)
    else:
        print('unlock_trade failed！ ')
        # print('unlock_trade failed: ', data)
    trd_ctx.close()


if len(stocklist) > 0:
    stocklistnum = len(stocklist)
    num = int(stocklistnum / 160) + 1
    print('总共', num, '批股票需要筛选！')
    for i in range(num):
        a, b = 160 * i, 160 * (i + 1)
        if 160 * (i + 1) > stocklistnum:
            a, b = 160 * i, stocklistnum
        try:
            print('当前进行第', i + 1, '批股票筛选：')
            b_list, s_list, b2_list, s2_list, b3, s3 = bs_list(stocklist[a:b])
            # b_list, s_list, b2_list, s2_list, b3, s3 ,b5,s5= bs_list(stocklist[a:b])
            modify('SELL', s_list)
            modify('BUY', b_list)
            modify('SELL2', s2_list)
            modify('BUY2', b2_list)
            # 做多
            trade(TrdSide.BUY, b3, 3)
            # trade(TrdSide.BUY, b5,5)
            # 做空
            trade(TrdSide.SELL_SHORT, s3, 3)
            # trade(TrdSide.SELL_SHORT, s5, 5)
        except OSError as err:
            print("OS error: {0}".format(err))
            quote_ctx.close()
quote_ctx.close()
print('链接关闭')
