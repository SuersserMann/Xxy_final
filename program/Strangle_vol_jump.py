from program.Function import *

pd.set_option('expand_frame_repr', False)


def calculate_annualized_variance(data, date):
    data_until_date = data[data['交易日期'] <= date]
    variance = data_until_date['log_returns'].var()
    annualized_variance = variance * 252  # Assuming 252 trading days in a year
    return annualized_variance


etf_data = pd.read_csv(r'C:\Users\86156\Desktop\option\data\ETF数据\510050.OF.csv',
                       encoding='gbk',
                       parse_dates=['交易日期'],
                       skiprows=1
                       )
etf_data['交易日期'] = pd.to_datetime(etf_data['交易日期'])
etf_data['log_returns'] = np.log(etf_data['收盘价'] / etf_data['前收盘价'])
etf_data['Year'] = etf_data['交易日期'].dt.year
yearly_data = etf_data.groupby('Year')

ivix = pd.read_csv(r'C:\Users\86156\Desktop\option\data\ETF数据\ivix.csv', delimiter=';', encoding='gbk')
ivix['cal_date'] = pd.to_datetime(ivix['cal_date'], format='%Y%m%d')
ivix['交易日期'] = ivix['cal_date'].dt.strftime('%Y-%m-%d')
ivix['交易日期'] = pd.to_datetime(ivix['交易日期'])

# Strangle_004 = pd.read_csv(os.path.join(portfolio_path, 'otm_strangle_004.csv'), encoding='gbk')
Strangle_006 = pd.read_csv(os.path.join(portfolio_path, 'otm_strangle_006.csv'), encoding='gbk')
Strangle_006['交易日期'] = pd.to_datetime(Strangle_006['交易日期'])
Strangle_006['Year'] = Strangle_006['交易日期'].dt.year
# Strangle_006['交易日期'] = pd.to_datetime(Strangle_006['交易日期'])
# Strangle_006['Year'] = Strangle_006['交易日期'].dt.year

put_atm = pd.read_csv(os.path.join(option_path, 'put_atm_m_ret.csv'), encoding='gbk')
put_otm = pd.read_csv(os.path.join(option_path, 'put_otm_m_ret.csv'), encoding='gbk')
put_itm = pd.read_csv(os.path.join(option_path, 'put_itm_m_ret.csv'), encoding='gbk')

put_all = pd.concat([put_otm, put_atm, put_itm])
put_otm_filtered = put_otm[(put_otm['在值程度'] >= 0.90) & (put_otm['在值程度'] <= 0.94)]
put_atm_filtered = put_all[(put_all['在值程度'] >= 0.98) & (put_all['在值程度'] <= 1.02)]

put_otm_avg = put_otm_filtered.groupby('交易日期')['IV'].mean().reset_index()
put_atm_avg = put_atm_filtered.groupby('交易日期')['IV'].mean().reset_index()
JRP = put_otm_avg.merge(put_atm_avg, on='交易日期', suffixes=('_otm', '_atm'))
JRP['JUMP_t'] = JRP['IV_otm'] - JRP['IV_atm']
JRP['交易日期'] = pd.to_datetime(JRP['交易日期'])

annualized_variances = []
for _, row in Strangle_006.iterrows():
    year = row['Year']
    trade_date = row['交易日期']
    year_data = yearly_data.get_group(year)
    annualized_variance = calculate_annualized_variance(year_data, trade_date)
    annualized_variances.append(annualized_variance)

Strangle_006['Annualized Variance'] = annualized_variances
Strangle_006['TAIL'] = Strangle_006['Annualized Variance'] - 0.5 * (Strangle_006['IV_call'] + Strangle_006['IV_put'])
Strangle_006 = Strangle_006.merge(JRP, on='交易日期')
Strangle_006 = Strangle_006.merge(ivix, on='交易日期')
Strangle_006['VRP'] = Strangle_006['monthly_RV_call'] - Strangle_006['ivix'] / 100
Strangle = portfolio_vomma(Strangle_006)

Strangle = Strangle[['Return', 'TAIL', 'JUMP_t', 'VRP', '组合Vomma']].dropna(how='any')
y_strangle_mean = Strangle['Return'].mean()
y_strangle_std = Strangle['Return'].std()
x_strangle_mean = Strangle[['TAIL', 'JUMP_t', 'VRP', '组合Vomma']].mean()
x_strangle_std = Strangle[['TAIL', 'JUMP_t', 'VRP', '组合Vomma']].std()

Strangle['Return_nor'] = (Strangle['Return'] - y_strangle_mean) / y_strangle_std
Strangle[['TAIL_nor', 'JUMP_t_nor', 'VRP_nor', '组合Vomma_nor']] = (Strangle[['TAIL', 'JUMP_t', 'VRP', '组合Vomma']] - x_strangle_mean) / x_strangle_std

y_strangle = Strangle['Return']
X_strangle = Strangle[['VRP', '组合Vomma', 'JUMP_t','TAIL']]
X_strangle = sm.add_constant(X_strangle)  # 非标准化
model_strangle = sm.OLS(y_strangle, X_strangle).fit()
print(model_strangle.summary())
nw_cov = sm.cov_hac(model_strangle, nlags=None)
nw_std_err = np.sqrt(np.diag(nw_cov))
print("Newey-West Standard Errors:", nw_std_err)
