from program.Function import *

pd.set_option('expand_frame_repr', False)

csp_002_004 = pd.read_csv(os.path.join(portfolio_path, 'CSP_002_004.csv'), encoding='gbk')
csp_004_006 = pd.read_csv(os.path.join(portfolio_path, 'CSP_004_006.csv'), encoding='gbk')
psp_002_004 = pd.read_csv(os.path.join(portfolio_path, 'PSP_002_004.csv'), encoding='gbk')
psp_004_006 = pd.read_csv(os.path.join(portfolio_path, 'PSP_004_006.csv'), encoding='gbk')

ivix = pd.read_csv(r'C:\Users\86156\Desktop\option\data\ETF数据\ivix.csv', delimiter=';', encoding='gbk')
ivix['cal_date'] = pd.to_datetime(ivix['cal_date'], format='%Y%m%d')
ivix['交易日期'] = ivix['cal_date'].dt.strftime('%Y-%m-%d')

csp_002_004_ivix = pd.merge(csp_002_004, ivix, on='交易日期')
csp_004_006_ivix = pd.merge(csp_004_006, ivix, on='交易日期')
psp_002_004_ivix = pd.merge(psp_002_004, ivix, on='交易日期')
psp_004_006_ivix = pd.merge(psp_004_006, ivix, on='交易日期')

# df_portfolio = [csp_002_004_ivix, csp_004_006_ivix, psp_002_004_ivix, psp_004_006_ivix]
bear_date_ranges = [
    ('2018-01-25', '2019-01-02'),
    ('2020-01-15', '2020-03-19'),
    ('2021-02-18', '2022-05-10'),
    ('2022-07-05', '2022-10-31')
]

csp_002_004_bear, csp_002_004_bull = split_dataframe_by_date_ranges(csp_002_004_ivix, bear_date_ranges)
csp_004_006_bear, csp_004_006_bull = split_dataframe_by_date_ranges(csp_004_006_ivix, bear_date_ranges)
psp_002_004_bear, psp_002_004_bull = split_dataframe_by_date_ranges(psp_002_004_ivix, bear_date_ranges)
psp_004_006_bear, psp_004_006_bull = split_dataframe_by_date_ranges(psp_004_006_ivix, bear_date_ranges)

portfolio_list = [csp_002_004_bear, csp_002_004_bull, csp_004_006_bear, csp_004_006_bull, psp_002_004_bear, psp_002_004_bull, psp_004_006_bear, psp_004_006_bull]
for df in portfolio_list:
    df['VRP'] = df['monthly_RV_otm'] - df['ivix'] / 100
    df['slope'] = df['IV_otm'] - df['IV_atm']
    # df['kurtosis'] = (df['IV_otm'] - 2 * df['IV_atm']) ** 2 / df['IV_atm']
    df = portfolio_vomma(df)
    # df = df[['VRP', 'Return', '交易日期']].dropna(how='any')
    # df = df[['skew', 'Return', '交易日期']].dropna(how='any')
    df = df[['组合Vomma', 'VRP', 'slope', 'Return', '交易日期']].dropna(how='any')

    y_option_mean = df['Return'].mean()
    y_option_std = df['Return'].std()
    x_option_mean = df[['组合Vomma', 'VRP', 'slope']].mean()
    x_option_std = df[['组合Vomma', 'VRP', 'slope']].std()

    df['Return_nor'] = (df['Return'] - y_option_mean) / y_option_std
    df[['组合Vomma_nor', 'VRP_nor', 'slope_nor']] = (df[['组合Vomma', 'VRP', 'slope']] - x_option_mean) / x_option_std

    y_option = df['Return_nor']
    x_option = df[['组合Vomma_nor', 'VRP_nor', 'slope_nor']]
    model_option = sm.OLS(y_option, x_option).fit()
    # y_option = df['Return']
    # x_option = df[['组合Vomma', 'VRP', 'slope']]
    # x_option = sm.add_constant(x_option)
    # model_option = sm.OLS(y_option, x_option).fit()

    nw_cov = cov_hac(model_option, nlags=None)
    nw_std_err = np.sqrt(np.diag(nw_cov))
    print(model_option.summary())
    print("Newey-West Standard Errors:", nw_std_err)
    print('****************************************************************************************************************************')
