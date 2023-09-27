from program.Function import *

pd.set_option('expand_frame_repr', False)


etf_data = pd.read_csv(r'C:\Users\86156\Desktop\option\data\ETF数据\510050.OF.csv',
                       encoding='gbk',
                       parse_dates=['交易日期'],
                       skiprows=1
                       )
etf_data['交易日期'] = pd.to_datetime(etf_data['交易日期'])

ivix = pd.read_csv(r'C:\Users\86156\Desktop\option\data\ETF数据\ivix.csv', delimiter=';', encoding='gbk')
ivix['cal_date'] = pd.to_datetime(ivix['cal_date'], format='%Y%m%d')
ivix['交易日期'] = ivix['cal_date'].dt.strftime('%Y-%m-%d')
ivix['交易日期'] = pd.to_datetime(ivix['交易日期'])

put_atm = pd.read_csv(os.path.join(option_path, 'put_atm_m_ret.csv'), encoding='gbk')
put_otm = pd.read_csv(os.path.join(option_path, 'put_otm_m_ret.csv'), encoding='gbk')
put_itm = pd.read_csv(os.path.join(option_path, 'put_itm_m_ret.csv'), encoding='gbk')

put_all = pd.concat([put_otm, put_atm, put_itm])
put_otm_filtered = put_otm[(put_otm['在值程度'] >= 0.90) & (put_otm['在值程度'] <= 0.94)]
put_atm_filtered = put_all[(put_all['在值程度'] >= 0.98) & (put_all['在值程度'] <= 1.02)]

put_otm_avg = put_otm_filtered.groupby('交易日期')['IV'].mean().reset_index()
put_atm_avg = put_atm_filtered.groupby('交易日期')['IV'].mean().reset_index()
merged_data = put_otm_avg.merge(put_atm_avg, on='交易日期', suffixes=('_otm', '_atm'))
merged_data['JUMP_t'] = merged_data['IV_otm'] - merged_data['IV_atm']
merged_data['交易日期'] = pd.to_datetime(merged_data['交易日期'])


etf_data['log_returns'] = np.log(etf_data['收盘价'] / etf_data['前收盘价'])
etf_data['Year'] = etf_data['交易日期'].dt.year
yearly_data = etf_data.groupby('Year')


def calculate_annualized_variance(data, date):
    data_until_date = data[data['交易日期'] <= date]
    variance = data_until_date['log_returns'].var()
    annualized_variance = variance * 252
    return annualized_variance


Straddle = pd.read_csv(os.path.join(portfolio_path, 'atm_straddle.csv'), encoding='gbk')
Straddle['交易日期'] = pd.to_datetime(Straddle['交易日期'])
Straddle['Year'] = Straddle['交易日期'].dt.year

annualized_variances = []
for _, row in Straddle.iterrows():
    year = row['Year']
    trade_date = row['交易日期']
    year_data = yearly_data.get_group(year)
    annualized_variance = calculate_annualized_variance(year_data, trade_date)
    annualized_variances.append(annualized_variance)

Straddle['Annualized Variance'] = annualized_variances
Straddle['TAIL'] = Straddle['Annualized Variance'] - 0.5 * (Straddle['IV_call'] + Straddle['IV_put'])
Straddle = Straddle.merge(merged_data, on='交易日期')
Straddle = Straddle.merge(ivix, on='交易日期')
Straddle['VRP'] = Straddle['monthly_RV_call'] - Straddle['ivix'] / 100
Straddle = portfolio_vomma(Straddle)

# Straddle = Straddle[['Return', 'JTIX']].dropna(how='any')
Straddle = Straddle[['Return', 'VRP', '组合Vomma', 'JUMP_t', 'TAIL']].dropna(how='any')

y_straddle_mean = Straddle['Return'].mean()
y_straddle_std = Straddle['Return'].std()
x_straddle_mean = Straddle[['JUMP_t', 'TAIL', 'VRP', '组合Vomma']].mean()
x_straddle_std = Straddle[['JUMP_t', 'TAIL', 'VRP', '组合Vomma']].std()
Straddle['Return_nor'] = (Straddle['Return'] - y_straddle_mean) / y_straddle_std
Straddle[['JUMP_t_nor', 'TAIL_nor', 'VRP_nor', '组合Vomma_nor']] = (Straddle[['JUMP_t', 'TAIL', 'VRP', '组合Vomma']] - x_straddle_mean) / x_straddle_std

y_straddle = Straddle['Return']
X_straddle = Straddle[['JUMP_t', 'TAIL']]
X_straddle = sm.add_constant(X_straddle)
model_straddle = sm.OLS(y_straddle, X_straddle).fit()
print(model_straddle.summary())
nw_cov = sm.cov_hac(model_straddle, nlags=None)
nw_std_err = np.sqrt(np.diag(nw_cov))
print("Newey-West Standard Errors:", nw_std_err)
