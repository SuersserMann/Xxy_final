from program.Function import *

pd.set_option('expand_frame_repr', False)

df_opt = pd.read_csv(r'C:\Users\86156\Desktop\option\data\ETF数据\50ETF_option.csv',
                     encoding='gbk',
                     parse_dates=['交易日期']
                     )
df_etf = pd.read_csv(r'C:\Users\86156\Desktop\option\data\ETF数据\510050.OF.csv',
                     encoding='gbk',
                     parse_dates=['交易日期'],
                     skiprows=1
                     )

df_option = df_opt[(df_opt['到期剩余天数'] > 25) & (df_opt['到期剩余天数'] < 51)]
df_option = df_option[df_option['交易日期'] >= '2016-02-01']
df_etf['daily_returns'] = np.log(df_etf['收盘价'] / df_etf['前收盘价'])
df_etf['monthly_RV'] = df_etf['daily_returns'].rolling(window=22).std()*np.sqrt(252)

df_atm = pd.DataFrame()

df_call_otm = pd.DataFrame()
df_call_itm = pd.DataFrame()
df_call_atm = pd.DataFrame()

df_put_otm = pd.DataFrame()
df_put_itm = pd.DataFrame()
df_put_atm = pd.DataFrame()

df_call = df_option[df_option['合约名称'].str.contains('购')]
df_put = df_option[df_option['合约名称'].str.contains('沽')]

trade_date = df_option['交易日期'].drop_duplicates(keep='first')

for date in trade_date:
    dates = pd.date_range(date, periods=1)
    spread = []
    t_day = df_option.groupby('交易日期').get_group(date)
    if t_day['收盘价'].isnull().any():
        rows = t_day[t_day['收盘价'].isna()].index
        k_to_drop1 = t_day.loc[rows, '合约名称']
        k_to_drop = list(k_to_drop1.str.split().str[-1].str[-4:])
        rows_to_drop = t_day[t_day['合约名称'].str.contains(k_to_drop[0])].index
        t_day = t_day.drop(rows_to_drop)
    # t_day['Strike'] = t_day['合约名称'].str.extract('(\d{4})[A-Za-z]*$')
    result = t_day['行权价'].tolist()
    p_strike = []
    for p in result:
        if p not in p_strike:
            p_strike.append(p)

    for k in p_strike:
        call_put = t_day.loc[t_day['行权价'] == k, '收盘价'].astype(float)
        spread.append(abs(call_put.diff(1)).iloc[1])
    min_diff = min(spread)
    count_min = spread.count(min_diff)

    if count_min > 1:
        indexArr = [i for i, v in enumerate(spread) if v == min_diff]
        k_xq = [p_strike[i] for i in indexArr if i < len(p_strike)]
        volume = []
        for j in range(len(indexArr)):
            vol = df_call.groupby('交易日期').get_group(date)['成交量(手)'].iloc[indexArr[j]] + \
                  df_put.groupby('交易日期').get_group(date)['成交量(手)'].iloc[indexArr[j]]
            volume.append(vol)
        pos = volume.index(max(volume))
        data_atm = {'交易日期': date, '行权价': k_xq[pos]}
        df_atm = df_atm.append(data_atm, ignore_index=True)
        # df_atm = pd.concat([df_atm, data_atm], ignore_index=True)
        call_itm_p = [p for p in p_strike if p < k_xq[pos]]
        call_otm_p = [p for p in p_strike if p > k_xq[pos]]
        put_itm_p = [p for p in p_strike if p > k_xq[pos]]
        put_otm_p = [p for p in p_strike if p < k_xq[pos]]
    else:
        pos = spread.index(min_diff)
        data_atm = {'交易日期': date, '行权价': p_strike[pos]}
        df_atm = df_atm.append(data_atm, ignore_index=True)
        # df_atm = pd.concat([df_atm, data_atm], ignore_index=True)
        call_itm_p = [p for p in p_strike if p < p_strike[pos]]
        call_otm_p = [p for p in p_strike if p > p_strike[pos]]
        put_itm_p = [p for p in p_strike if p > p_strike[pos]]
        put_otm_p = [p for p in p_strike if p < p_strike[pos]]

    data_call_otm = pd.DataFrame({'交易日期': np.repeat(dates, len(call_otm_p)),
                                  '行权价': np.tile(call_otm_p, len(dates))})
    df_call_otm = df_call_otm.append(data_call_otm, ignore_index=True)
    # df_call_otm = pd.concat([df_call_otm, data_call_otm], ignore_index=True)

    data_call_itm = pd.DataFrame({'交易日期': np.repeat(dates, len(call_itm_p)),
                                  '行权价': np.tile(call_itm_p, len(dates))})
    df_call_itm = df_call_itm.append(data_call_itm, ignore_index=True)
    # df_call_itm = pd.concat([df_call_itm, data_call_itm], ignore_index=True)

    data_put_otm = pd.DataFrame({'交易日期': np.repeat(dates, len(put_otm_p)),
                                 '行权价': np.tile(put_otm_p, len(dates))})
    df_put_otm = df_put_otm.append(data_put_otm, ignore_index=True)
    # df_put_otm = pd.concat([df_put_otm, data_put_otm], ignore_index=True)

    data_put_itm = pd.DataFrame({'交易日期': np.repeat(dates, len(put_itm_p)),
                                 '行权价': np.tile(put_itm_p, len(dates))})
    df_put_itm = df_put_itm.append(data_put_itm, ignore_index=True)
    # df_put_itm = pd.concat([df_put_itm, data_put_itm], ignore_index=True)

df_call_otm = merge_option(df_call_otm, df_call)
df_call_itm = merge_option(df_call_itm, df_call)
df_call_atm = merge_option(df_atm, df_call)
df_put_otm = merge_option(df_put_otm, df_put)
df_put_itm = merge_option(df_put_itm, df_put)
df_put_atm = merge_option(df_atm, df_put)

df_call_otm = fill_closing_price(df_call_otm)
df_call_itm = fill_closing_price(df_call_itm)
df_call_atm = fill_closing_price(df_call_atm)
df_put_otm = fill_closing_price(df_put_otm)
df_put_itm = fill_closing_price(df_put_itm)
df_put_atm = fill_closing_price(df_put_atm)

put_atm = merge_with_etf(df_put_atm, df_etf)
put_itm = merge_with_etf(df_put_itm, df_etf)
put_otm = merge_with_etf(df_put_otm, df_etf)
call_atm = merge_with_etf(df_call_atm, df_etf)
call_itm = merge_with_etf(df_call_itm, df_etf)
call_otm = merge_with_etf(df_call_otm, df_etf)

df_call_otm.to_pickle(os.path.join(pkl_path, 'call_otm.pkl'))
df_call_itm.to_pickle(os.path.join(pkl_path, 'call_itm.pkl'))
df_call_atm.to_pickle(os.path.join(pkl_path, 'call_atm.pkl'))
df_put_otm.to_pickle(os.path.join(pkl_path, 'put_otm.pkl'))
df_put_itm.to_pickle(os.path.join(pkl_path, 'put_itm.pkl'))
df_put_atm.to_pickle(os.path.join(pkl_path, 'put_atm.pkl'))
