import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as si
import matplotlib.dates as mdates
import scipy.stats as stats
import os
from path import script_directory
import statsmodels.api as sm

# 定义路径
pkl_path = os.path.join(script_directory, 'data/classification_results/pkl/')
option_path = os.path.join(script_directory, 'data/classification_results/csv/option/')
portfolio_path = os.path.join(script_directory, 'data/classification_results/csv/option_portfolio/')

def merge_option(df1, df_option):
    df1 = df1.merge(df_option[['交易日期', '行权价', '成交量(手)', '到期剩余天数', '期权代码', '收盘价']],
                    on=['交易日期', '行权价'], how='left')
    return df1


def fill_closing_price(df_option):
    zero_price_options = df_option[(df_option['收盘价'] == 0) | (df_option['收盘价'] == 0.01)]
    zero_price_options.loc[:, '交易日期'] = pd.to_datetime(zero_price_options['交易日期'],
                                                           format='%Y/%m/%d %H:%M:%S').dt.strftime('%Y-%m-%d')
    for index, row in zero_price_options.iterrows():
        try:
            date = row['交易日期']
            code = row['期权代码']
            file_path = f"D:/PyCharm project/The-Pricing-of-Volatility-and-Jump-Risks-in-Index-Options/data/classification_results/ETF_data/min_options_etf50/{date}.csv"
            option_data = pd.read_csv(file_path, skiprows=1, encoding='gbk')
            option_row = option_data[option_data['期权代码'] == code]
            closing_price = option_row[option_row['交易日期'] == '%s' % date + ' 15:00:00']['收盘价'].values[0]
            df_option.at[index, '收盘价'] = closing_price
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            continue

    return df_option


def pkl_to_frame(pkl_name):
    with open(pkl_path + '%s' % pkl_name + '.pkl', 'rb') as f:
        data = pickle.load(f)
    df = pd.DataFrame(data)
    return df


def merge_with_etf(df_option, df_etf):
    df_option['到期ETF收盘价'] = 0.0
    df_option['现期ETF收盘价'] = 0.0
    df_option['monthly_RV'] = 0.0
    for index, row in df_option.iterrows():
        remaining_days = row['到期剩余天数']
        expiration_date = row['交易日期'] + pd.Timedelta(days=remaining_days)
        closing_price_due = df_etf.loc[df_etf['交易日期'] == expiration_date]['收盘价'].values[0]
        closing_price_now = df_etf.loc[df_etf['交易日期'] == row['交易日期']]['收盘价'].values[0]
        monthly_RV = df_etf.loc[df_etf['交易日期'] == row['交易日期']]['monthly_RV'].values[0]
        df_option.loc[index, 'monthly_RV'] = monthly_RV
        df_option.loc[index, '到期ETF收盘价'] = closing_price_due
        df_option.loc[index, '现期ETF收盘价'] = closing_price_now

    return df_option


# Ret = max(ST-K, 0) / C - 1
def call_opt_return(df_option):
    df_option['option_type'] = 'call'
    df_option['收盘价'] = pd.to_numeric(df_option['收盘价'])
    df_option['期权收益率'] = pd.to_numeric(df_option.apply(
        lambda row: -1 if row['收盘价'] == 0 else (max(row['到期ETF收盘价'] - row['行权价'], 0) / row['收盘价'] - 1) * (
                    30 / row['到期剩余天数']),
        axis=1))
    return df_option


# Ret = max(K-ST, 0) / P - 1
def put_opt_return(df_option):
    df_option['option_type'] = 'put'
    df_option['收盘价'] = pd.to_numeric(df_option['收盘价'])
    df_option['期权收益率'] = pd.to_numeric(df_option.apply(
        lambda row: -1 if row['收盘价'] == 0 else (max(row['行权价'] - row['到期ETF收盘价'], 0) / row['收盘价'] - 1) * (
                    30 / row['到期剩余天数']),
        axis=1))
    return df_option


def draw_curve(df_option, moneyness, option_type):
    df_option = df_option[df_option['在值程度'].isin(moneyness)]
    df_option = df_option.sort_values(by='在值程度')
    groups = df_option.groupby('在值程度')
    mean_returns = groups['期权收益率'].mean()
    q1_returns = groups['期权收益率'].quantile(0.25)
    q2_returns = groups['期权收益率'].quantile(0.5)
    q3_returns = groups['期权收益率'].quantile(0.75)
    plt.rcParams['figure.figsize'] = [12, 8]
    plt.rcParams['font.size'] = 14
    fig, ax = plt.subplots()
    ax.plot(mean_returns.index, mean_returns.values, label='Mean Return')
    # ax.fill_between(mean_returns.index, q1_returns, q3_returns, alpha=0.3, label='Interquartile Range')
    ax.plot(mean_returns.index, q2_returns, linestyle='--', label='Median Return')
    ax.set_xlabel('Moneyness')
    ax.set_ylabel('Expected Returns')
    ax.set_title('Expected ' + '%s' % option_type + ' Option Returns')
    ax.legend()
    plt.show()


def draw_portfolio_curve(df1, portfolio):
    if portfolio == 'Atm Straddle':
        average_return_df = df1.groupby('交易日期')['Return'].mean().reset_index()
        closing_prices_df = pd.DataFrame(
            {'交易日期': df1['交易日期'], 'ETF收盘价': df1['现期ETF收盘价_call'], 'ETF波动率': df1['monthly_RV_call']})
        merged_df = pd.merge(average_return_df, closing_prices_df, on='交易日期')
        merged_df['交易日期'] = pd.to_datetime(merged_df['交易日期'])
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        date_fmt = mdates.DateFormatter('%Y-%m')
        ax1.xaxis.set_major_formatter(date_fmt)
        ax1.plot(merged_df['交易日期'], merged_df['ETF波动率'], color='blue', label='ETF Volatility', linestyle='--')
        ax1.set_ylabel('ETF Volatility')
        ax2.plot(merged_df['交易日期'], merged_df['Return'], color='red', label='Portfolio Return', linestyle='-')
        ax2.plot(merged_df['交易日期'], merged_df['ETF收盘价'], color='yellow', label='ETF Closing Price', linestyle=':')
        ax2.set_ylabel('Portfolio Return and ETF Closing Price')
        ax2.axhline(y=0, color='black', linewidth=1)
        # ax2.axhline(y=-1, color='black', linewidth=1)
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        plt.title('%s' % portfolio + ' Returns')
        plt.legend()
        plt.show()
    else:
        df1.groupby('交易日期')['Return'].mean().sort_index().plot()
        plt.title('%s' % portfolio + ' Returns')
        plt.axhline(y=0, color='black', linewidth=1)
        # plt.axhline(y=-1, color='black', linewidth=1)
        plt.xlabel('Trade Date')
        plt.ylabel('Portfolio Returns')
        plt.show()


def black_scholes(option_type, S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        price = S * si.norm.cdf(d1) - K * np.exp(-r * T) * si.norm.cdf(d2)
    elif option_type == 'put':
        price = K * np.exp(-r * T) * si.norm.cdf(-d2) - S * si.norm.cdf(-d1)

    return price


def implied_volatility_bisection(option_type, S, K, T, r, option_price, lower_bound=1e-6, upper_bound=2, tol=1e-6,
                                 max_iter=1000):
    if option_type not in ('call', 'put'):
        raise ValueError("Invalid option type. Use 'call' or 'put'")

    if S <= 0 or K <= 0 or T <= 0 or option_price < 0:
        raise ValueError("Invalid input values")

    lower_sigma = lower_bound
    upper_sigma = upper_bound

    for _ in range(max_iter):
        mid_sigma = (lower_sigma + upper_sigma) / 2.0
        mid_price = black_scholes(option_type, S, K, T, r, mid_sigma)

        if abs(mid_price - option_price) < tol:
            return mid_sigma

        if mid_price < option_price:
            lower_sigma = mid_sigma
        else:
            upper_sigma = mid_sigma

    raise ValueError("Failed to converge")


def calculate_iv(row, lower_bound=1e-6, upper_bound=2, tol=1e-8, max_iter=1000):
    option_type = row['option_type']
    S = row['现期ETF收盘价']
    K = row['行权价']
    T = row['到期剩余天数'] / 365
    r_risk_free = 0.02311
    option_price = row['收盘价']

    try:
        return implied_volatility_bisection(option_type, S, K, T, r_risk_free, option_price, lower_bound, upper_bound,
                                            tol,
                                            max_iter)
    except ValueError:
        return np.nan


# Ret = (max(ST-K, 0) + max(K-ST, 0) - P - C) / (P + C)
def straddle(call_data, put_data):
    option_merged = pd.merge(call_data, put_data, on='交易日期', suffixes=('_call', '_put'))
    option_merged['收盘价_call'] = pd.to_numeric(option_merged['收盘价_call'])
    option_merged['收盘价_put'] = pd.to_numeric(option_merged['收盘价_put'])
    for i in range(len(option_merged)):
        call_price = option_merged.at[i, '收盘价_call']
        put_price = option_merged.at[i, '收盘价_put']
        if call_price != 0 and put_price != 0:
            ret = pd.to_numeric(
                ((max(option_merged.at[i, '到期ETF收盘价_call'] - option_merged.at[i, '行权价_call'], 0) + max(
                    option_merged.at[i, '行权价_put'] - option_merged.at[i, '到期ETF收盘价_put'],
                    0)) / (call_price + put_price) - 1) * (30 / option_merged.at[i, '到期剩余天数_call']))
            option_merged.at[i, 'Return'] = ret
        else:
            option_merged.at[i, 'Return'] = np.nan

    return option_merged


def crash_neutral(df_straddle, put_otm, put_moneyness):
    selected_put = put_otm[put_otm['在值程度'] == 1 - put_moneyness]
    selected_put.columns = [col + '_otm' if col != '交易日期' else col for col in selected_put.columns]
    df_straddle.columns = [col + '_straddle' if col != '交易日期' else col for col in df_straddle.columns]
    cns = pd.merge(selected_put, df_straddle, on='交易日期')
    for i in range(len(cns)):
        otm_price = cns.at[i, '收盘价_otm']
        put_price = cns.at[i, '收盘价_put_straddle']
        call_price = cns.at[i, '收盘价_call_straddle']
        if call_price != 0 and put_price != 0 and otm_price != 0:
            ret = pd.to_numeric(
                ((max(cns.at[i, '到期ETF收盘价_call_straddle'] - cns.at[i, '行权价_call_straddle'], 0) + max(
                    cns.at[i, '行权价_put_straddle'] - cns.at[i, '到期ETF收盘价_put_straddle'],
                    0) - max(cns.at[i, '行权价_otm'] - cns.at[i, '到期ETF收盘价_otm'], 0)) / (
                             call_price + put_price - otm_price) - 1) * (30 / cns.at[i, '到期剩余天数_call_straddle']))
            cns.at[i, 'Return'] = ret
        else:
            cns.at[i, 'Return'] = np.nan

    return cns


# Ret = (max(ST-K, 0) + max(K-ST, 0) - P - C) / (P + C)
def otm_strangle(call_otm, call_moneyness, put_otm, put_moneyness):
    selected_call = call_otm[call_otm['在值程度'] == 1 + call_moneyness]
    selected_put = put_otm[put_otm['在值程度'] == 1 - put_moneyness]
    strangle = pd.merge(selected_call, selected_put, on='交易日期', suffixes=('_call', '_put'))
    for i in range(len(strangle)):
        call_price = strangle.at[i, '收盘价_call']
        put_price = strangle.at[i, '收盘价_put']
        if call_price != 0 and put_price != 0:
            ret = pd.to_numeric(((max(strangle.at[i, '到期ETF收盘价_call'] - strangle.at[i, '行权价_call'], 0) + max(
                strangle.at[i, '行权价_put'] - strangle.at[i, '到期ETF收盘价_put'],
                0)) / (call_price + put_price) - 1) * (30 / strangle.at[i, '到期剩余天数_call']))
            strangle.at[i, 'Return'] = ret
        else:
            strangle.at[i, 'Return'] = np.nan

    return strangle


# Ret = (max(K_atm-ST, 0) - max(K_otm-ST, 0) - p_atm + p_otm) / (p_atm - p_otm)
def put_spread(put_atm, put_otm, otm_moneyness):
    # selected_otm = put_otm[put_otm['在值程度'] == 1 - otm_moneyness]
    # selected_otm = put_otm[(put_otm['在值程度'] < 0.98) & (put_otm['在值程度'] >= 0.96)]
    selected_otm = put_otm[(put_otm['在值程度'] < 0.96) & (put_otm['在值程度'] >= 0.94)]
    option_merged = pd.merge(put_atm, selected_otm, on='交易日期', suffixes=('_atm', '_otm'))
    for i in range(len(option_merged)):
        atm_price = option_merged.at[i, '收盘价_atm']
        otm_price = option_merged.at[i, '收盘价_otm']
        if atm_price != 0 and otm_price != 0:
            ret = pd.to_numeric(
                ((max(option_merged.at[i, '行权价_atm'] - option_merged.at[i, '到期ETF收盘价_atm'], 0) - max(
                    option_merged.at[i, '行权价_otm'] - option_merged.at[i, '到期ETF收盘价_otm'],
                    0)) / (atm_price - otm_price) - 1) * (30 / option_merged.at[i, '到期剩余天数_atm']))
            option_merged.at[i, 'Return'] = ret
        else:
            option_merged.at[i, 'Return'] = np.nan

    return option_merged


# Ret = (max(ST-K_atm, 0) - p_atm -max(ST-K_otm, 0) + p_otm) / (p_atm - p_otm)
def call_spread(call_atm, call_otm, otm_moneyness):
    # selected_otm = call_otm[call_otm['在值程度'] == 1 + otm_moneyness]
    # selected_otm = call_otm[(call_otm['在值程度'] > 1.02) & (call_otm['在值程度'] <= 1.04)]
    selected_otm = call_otm[(call_otm['在值程度'] > 1.04) & (call_otm['在值程度'] <= 1.06)]
    option_merged = pd.merge(call_atm, selected_otm, on='交易日期', suffixes=('_atm', '_otm'))
    for i in range(len(option_merged)):
        atm_price = option_merged.at[i, '收盘价_atm']
        otm_price = option_merged.at[i, '收盘价_otm']
        if atm_price != 0 and otm_price != 0:
            ret = pd.to_numeric(
                ((max(option_merged.at[i, '到期ETF收盘价_atm'] - option_merged.at[i, '行权价_atm'], 0) - max(
                    option_merged.at[i, '到期ETF收盘价_otm'] - option_merged.at[i, '行权价_otm'],
                    0)) / (atm_price - otm_price) - 1) * (30 / option_merged.at[i, '到期剩余天数_atm']))
            option_merged.at[i, 'Return'] = ret
        else:
            option_merged.at[i, 'Return'] = np.nan

    return option_merged


def butterfly_call_ret(call_otm, otm_moneyness, call_atm, call_itm, itm_moneyness):
    call_otm.columns = [col + '_otm' if col != '交易日期' else col for col in call_otm.columns]
    call_atm.columns = [col + '_atm' if col != '交易日期' else col for col in call_atm.columns]
    call_itm.columns = [col + '_itm' if col != '交易日期' else col for col in call_itm.columns]
    selected_otm = call_otm[call_otm['在值程度_otm'] == 1 + otm_moneyness]
    selected_itm = call_itm[call_itm['在值程度_itm'] == 1 - itm_moneyness]
    option_merged = selected_otm.merge(call_atm, on='交易日期').merge(selected_itm, on='交易日期')

    for i in range(len(option_merged)):
        atm_price = option_merged.at[i, '收盘价_atm']
        otm_price = option_merged.at[i, '收盘价_otm']
        itm_price = option_merged.at[i, '收盘价_itm']
        if atm_price != 0 and otm_price != 0 and itm_price != 0:
            ret = pd.to_numeric(
                ((max(option_merged.at[i, '到期ETF收盘价_otm'] - option_merged.at[i, '行权价_otm'], 0)
                  + max(option_merged.at[i, '到期ETF收盘价_itm'] - option_merged.at[i, '行权价_itm'], 0)
                  - 2 * max(option_merged.at[i, '到期ETF收盘价_atm'] - option_merged.at[i, '行权价_atm'],
                            0)) - otm_price - itm_price + 2 * atm_price)
                * (30 / option_merged.at[i, '到期剩余天数_atm']))
            option_merged.at[i, 'Return'] = ret
        else:
            option_merged.at[i, 'Return'] = np.nan

    return option_merged


def add_one_month(year, month):
    if month == 12:
        return year + 1, 1
    else:
        return year, month + 1


def filter_by_date_range(df, start_date, end_date):
    return df[(df['交易日期'] >= start_date) & (df['交易日期'] <= end_date)]


# 定义一个函数，用于将dataframe划分为熊市和牛市dataframe
def split_dataframe_by_date_ranges(df, bear_date_ranges):
    df_bear = pd.DataFrame()
    df_bull = df.copy()

    for start_date, end_date in bear_date_ranges:
        filtered_df = filter_by_date_range(df, start_date, end_date)
        df_bear = pd.concat([df_bear, filtered_df], ignore_index=True)
        df_bull = df_bull.drop(filtered_df.index)

    return df_bear, df_bull


def calculate_vomma(s, k, t, r, sigma, option_type):
    d1 = (np.log(s / k) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    if option_type == 'call':
        vomma = s * np.sqrt(t) * stats.norm.pdf(d1) * d1 * d2 / sigma
    elif option_type == 'put':
        vomma = -s * np.sqrt(t) * stats.norm.pdf(-d1) * d1 * d2 / sigma
    return vomma


def portfolio_vomma(df_portfolio):
    df_portfolio['Vomma_call'] = df_portfolio.apply(lambda x: calculate_vomma(
        x['现期ETF收盘价_call'],
        x['行权价_call'],
        x['到期剩余天数_call'] / 365,
        0.02311,  # 假设无风险利率为 0.02311
        x['IV_call'],
        x['option_type_call']
    ), axis=1)

    df_portfolio['Vomma_put'] = df_portfolio.apply(lambda x: calculate_vomma(
        x['现期ETF收盘价_put'],
        x['行权价_put'],
        x['到期剩余天数_put'] / 365,
        0.02311,  # 假设无风险利率为 0.02311
        x['IV_put'],
        x['option_type_put']
    ), axis=1)
    df_portfolio['权重'] = 0.5
    df_portfolio['组合Vomma'] = df_portfolio['权重'] * (df_portfolio['Vomma_put'] + df_portfolio['Vomma_call'])

    return df_portfolio
