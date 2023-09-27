import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from program.Function import *

pd.set_option('expand_frame_repr', False)

atm_straddle = pd.read_csv(os.path.join(portfolio_path, 'atm_straddle.csv'), encoding='gbk')

call_otm = pd.read_csv(os.path.join(option_path, 'call_otm_m_ret.csv'), encoding='gbk')
call_itm = pd.read_csv(os.path.join(option_path, 'call_itm_m_ret.csv'), encoding='gbk')
call_atm = pd.read_csv(os.path.join(option_path, 'call_atm_m_ret.csv'), encoding='gbk')
put_otm = pd.read_csv(os.path.join(option_path, 'put_otm_m_ret.csv'), encoding='gbk')
put_itm = pd.read_csv(os.path.join(option_path, 'put_itm_m_ret.csv'), encoding='gbk')
put_atm = pd.read_csv(os.path.join(option_path, 'put_atm_m_ret.csv'), encoding='gbk')

ivix = pd.read_csv(r'C:\Users\86156\Desktop\option\data\ETF数据\ivix.csv', delimiter=';', encoding='gbk')
ivix['cal_date'] = pd.to_datetime(ivix['cal_date'], format='%Y%m%d')
ivix['交易日期'] = ivix['cal_date'].dt.strftime('%Y-%m-%d')

call_96_100 = call_itm[(call_itm['在值程度'] >= 0.96) & (call_itm['在值程度'] < 1.00)]
call_atm_otm = pd.concat([call_atm, call_otm])
call_100_104 = call_atm_otm[(call_atm_otm['在值程度'] >= 1.00) & (call_atm_otm['在值程度'] < 1.04)]
call_104_108 = call_otm[(call_otm['在值程度'] >= 1.04) & (call_otm['在值程度'] < 1.08)]

put_92_96 = put_otm[(put_otm['在值程度'] >= 0.92) & (put_otm['在值程度'] < 0.96)]
put_96_100 = put_otm[(put_otm['在值程度'] >= 0.96) & (put_otm['在值程度'] < 1.00)]
put_atm_itm = pd.concat([put_atm, put_itm])
put_100_104 = put_atm_itm[(put_atm_itm['在值程度'] >= 1.00) & (put_atm_itm['在值程度'] < 1.04)]

# 合并筛选后的看涨和看跌期权
calls_filtered = pd.concat([call_96_100, call_100_104, call_104_108])
puts_filtered = pd.concat([put_92_96, put_96_100, put_100_104])

# 筛选straddle组合
straddle_94_98 = calls_filtered[(calls_filtered['在值程度'] >= 0.94) & (calls_filtered['在值程度'] < 0.98)].merge(
    puts_filtered[(puts_filtered['在值程度'] >= 0.94) & (puts_filtered['在值程度'] < 0.98)], on=['行权价', '交易日期'], suffixes=('_call', '_put'))

straddle_98_102 = calls_filtered[(calls_filtered['在值程度'] >= 0.98) & (calls_filtered['在值程度'] < 1.02)].merge(
    puts_filtered[(puts_filtered['在值程度'] >= 0.98) & (puts_filtered['在值程度'] < 1.02)], on=['行权价', '交易日期'], suffixes=('_call', '_put'))

straddle_102_106 = calls_filtered[(calls_filtered['在值程度'] >= 1.02) & (calls_filtered['在值程度'] < 106)].merge(
    puts_filtered[(puts_filtered['在值程度'] >= 1.02) & (puts_filtered['在值程度'] < 1.06)], on=['行权价', '交易日期'], suffixes=('_call', '_put'))

straddle_list = [straddle_94_98, straddle_98_102, straddle_102_106]
for straddle_type in straddle_list:
    for i in range(len(straddle_type)):
        call_price = straddle_type.at[i, '收盘价_call']
        put_price = straddle_type.at[i, '收盘价_put']
        if call_price != 0 and put_price != 0:
            ret = pd.to_numeric(
                ((max(straddle_type.at[i, '到期ETF收盘价_call'] - straddle_type.at[i, '行权价'], 0) + max(
                    straddle_type.at[i, '行权价'] - straddle_type.at[i, '到期ETF收盘价_put'],
                    0)) / (call_price + put_price)) ** (30 / straddle_type.at[i, '到期剩余天数_call']) - 1)
            straddle_type.at[i, 'Return'] = ret
        else:
            straddle_type.at[i, 'Return'] = np.nan

straddle_98_102_ivix = straddle_98_102.merge(ivix, on=['交易日期'])
straddle_98_102_ivix = straddle_98_102_ivix.drop(columns=['Unnamed: 0_call', 'Unnamed: 0_put', 'exchange', 'cal_date'])
straddle_98_102_ivix['VRP'] = straddle_98_102_ivix['monthly_RV_call'] - straddle_98_102_ivix['ivix'] / 100

straddle_94_98_ivix = straddle_94_98.merge(ivix, on=['交易日期'])
straddle_94_98_ivix = straddle_94_98_ivix.drop(columns=['Unnamed: 0_call', 'Unnamed: 0_put', 'exchange', 'cal_date'])
straddle_94_98_ivix['VRP'] = straddle_94_98_ivix['monthly_RV_call'] - straddle_94_98_ivix['ivix'] / 100

straddle_102_106_ivix = straddle_102_106.merge(ivix, on=['交易日期'])
straddle_102_106_ivix = straddle_102_106_ivix.drop(columns=['Unnamed: 0_call', 'Unnamed: 0_put', 'exchange', 'cal_date'])
straddle_102_106_ivix['VRP'] = straddle_102_106_ivix['monthly_RV_call'] - straddle_102_106_ivix['ivix'] / 100

straddle_ivix_list = [straddle_94_98_ivix, straddle_98_102_ivix, straddle_102_106_ivix]
for straddle_ivix in straddle_ivix_list:
    straddle_ivix = portfolio_vomma(straddle_ivix)
    data = pd.DataFrame()
    data['returns'] = straddle_ivix['Return']
    data['VRP'] = straddle_ivix['VRP']
    data['组合Vomma'] = straddle_ivix['组合Vomma']
    data['交易日期'] = pd.to_datetime(straddle_ivix['交易日期'])
    # data = data[['returns', 'VRP', '交易日期']].dropna(how='any')
    data = data[['returns', '组合Vomma', '交易日期', 'VRP']].dropna(how='any')

    y_straddle = data['returns']
    # X_straddle = data['VRP']
    X_straddle = data[['组合Vomma', 'VRP']]
    X_straddle = sm.add_constant(X_straddle)
    model_straddle = sm.OLS(y_straddle, X_straddle).fit()

    # 计算Newey-West标准误
    nw_cov = cov_hac(model_straddle, nlags=None)
    nw_std_err = np.sqrt(np.diag(nw_cov))

    # data['residuals'] = model_straddle.resid
    # data['fitted_values'] = model_straddle.fittedvalues
    #
    # plt.figure(figsize=(8, 6))
    # sns.scatterplot(x=data['fitted_values'], y=data['residuals'])
    # plt.axhline(y=0, color='r', linestyle='--')
    # plt.xlabel('Fitted Values')
    # plt.ylabel('Residuals')
    # plt.title('Residual Plot')
    # plt.show()

    print(model_straddle.summary())
    print("Newey-West Standard Errors:", nw_std_err)
    print('****************************************************************************************************************************')

# opt_list = [call_96_100, call_100_104, call_104_108, put_92_96, put_96_100, put_100_104]
# for option in opt_list:
#     option = option.merge(ivix, on='交易日期')
#     option = option.drop(columns=['Unnamed: 0', 'exchange', 'cal_date'])
#     option['VRP'] = option['monthly_RV'] - option['ivix'] / 100
#     data = pd.DataFrame()
#     data['returns'] = option['期权收益率']
#     data['VRP'] = option['VRP']
#     data['交易日期'] = pd.to_datetime(option['交易日期'])
#     data = data[['returns', 'VRP', '交易日期']].dropna(how='any')
#
#     y_option = data['returns']
#     x_option = data['VRP']
#     x_option = sm.add_constant(x_option)
#     model_option = sm.OLS(y_option, x_option).fit()
#
#     # 计算Newey-West标准误
#     nw_cov = cov_hac(model_option, nlags=None)
#     nw_std_err = np.sqrt(np.diag(nw_cov))
#
#     # data['residuals'] = model_option.resid
#     # data['fitted_values'] = model_option.fittedvalues
#     #
#     # plt.figure(figsize=(8, 6))
#     # sns.scatterplot(x=data['fitted_values'], y=data['residuals'])
#     # plt.axhline(y=0, color='r', linestyle='--')
#     # plt.xlabel('Fitted Values')
#     # plt.ylabel('Residuals')
#     # plt.title('Residual Plot')
#     # plt.show()
#     print(model_option.summary())
#     print("Newey-West Standard Errors:", nw_std_err)
#     print('****************************************************************************************************************************')
