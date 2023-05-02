#港股 美股 富途牛牛api 国内股票 期货量化代写，vx 1985159637，备注：量化需求，其他不通过
from futu import *
quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)
data = quote_ctx.get_user_security("SELL")[1]
data1 = quote_ctx.get_user_security("BUY")[1]
try:
    selllist = data['code'].values.tolist()  # 转为 list
    quote_ctx.modify_user_security('SELL', ModifyUserSecurityOp.DEL, selllist)
    buylist = data1['code'].values.tolist()
    quote_ctx.modify_user_security('BUY', ModifyUserSecurityOp.DEL, buylist)
except:
    print('error:', data, data1)
quote_ctx.close()
