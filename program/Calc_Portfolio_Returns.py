from program.Function import *

pd.set_option('expand_frame_repr', False)

call_atm = pd.read_csv(os.path.join(option_path, 'call_atm_m_ret.csv'), encoding='gbk')
call_otm = pd.read_csv(os.path.join(option_path, 'call_otm_m_ret.csv'), encoding='gbk')
call_itm = pd.read_csv(os.path.join(option_path, 'call_itm_m_ret.csv'), encoding='gbk')
put_atm = pd.read_csv(os.path.join(option_path, 'put_atm_m_ret.csv'), encoding='gbk')
put_otm = pd.read_csv(os.path.join(option_path, 'put_otm_m_ret.csv'), encoding='gbk')
put_itm = pd.read_csv(os.path.join(option_path, 'put_itm_m_ret.csv'), encoding='gbk')

df_call = pd.concat([call_itm, call_atm, call_otm], axis=0)
df_put = pd.concat([put_otm, put_atm, put_itm], axis=0)

call_moneyness = [0.96, 0.98, 1, 1.02, 1.04, 1.06, 1.08]
put_moneyness = [0.92, 0.94, 0.96, 0.98, 1, 1.02, 1.04]

# draw_curve(df_call, call_moneyness, 'Call')
# draw_curve(df_put, put_moneyness, 'Put')

atm_straddle = straddle(call_atm, put_atm)  # 波动率风险方向
# atm_straddle.to_csv(os.path.join(csv_path, 'atm_straddle.csv'), encoding='gbk')
atm_straddle_avgret = atm_straddle['Return'].mean()
atm_straddle_mid = atm_straddle['Return'].quantile(0.5)
atm_straddle_std = atm_straddle['Return'].std()
# print(atm_straddle_avgret)
# print(atm_straddle_mid)
print(atm_straddle.columns)

# cns = crash_neutral(atm_straddle, put_otm, 0.06)
# # cns.to_csv(os.path.join(csv_path, 'CNS.csv'), encoding='gbk')
# cns_avgret = cns['Return'].mean()
# cns_mid = cns['Return'].quantile(0.5)
# cns_std = cns['Return'].std()

otm_strangle = otm_strangle(call_otm, 0.04, put_otm, 0.04)
otm_strangle = otm_strangle.drop(columns=['Unnamed: 0_call', 'Unnamed: 0_put'])
# otm_strangle.to_csv(os.path.join(csv_path, 'otm_strangle_004.csv'), encoding='gbk')
otm_strangle_avgret = otm_strangle['Return'].mean()
otm_strangle_mid = otm_strangle['Return'].quantile(0.5)
otm_strangle_std = otm_strangle['Return'].std()
# print(otm_strangle_avgret)
# print(otm_strangle_mid)
print(otm_strangle_std)

psp = put_spread(put_atm, put_otm, 0.04)
# psp.to_csv(os.path.join(csv_path, 'PSP_004_006.csv'), encoding='gbk')
psp_avgret = psp['Return'].mean()
psp_mid = psp['Return'].quantile(0.5)
psp_std = psp['Return'].std()
# print(psp_avgret)
# print(psp_mid)
print(psp_std)

csp = call_spread(call_atm, call_otm, 0.04)
# csp.to_csv(os.path.join(csv_path, 'CSP_004_006.csv'), encoding='gbk')
csp_avgret = csp['Return'].mean()
csp_mid = csp['Return'].quantile(0.5)
csp_std = csp['Return'].std()
# print(csp_avgret)
# print(csp_mid)
print(csp_std)

butterfly_call = butterfly_call_ret(call_otm, 0.04, call_atm, call_itm, 0.04)
# butterfly_call.to_csv(os.path.join(csv_path, 'Butterfly.csv'), encoding='gbk')
butterfly_call_avgret = butterfly_call['Return'].mean()
butterfly_call_mid = butterfly_call['Return'].quantile(0.5)
butterfly_call_std = butterfly_call['Return'].std()
print(butterfly_call_avgret)
print(butterfly_call_mid)
print(butterfly_call_std)

draw_portfolio_curve(atm_straddle, 'Atm Straddle')
draw_portfolio_curve(otm_strangle, 'Otm Strangle')
draw_portfolio_curve(psp, 'Put Spread')
draw_portfolio_curve(csp, 'Call Spread')
# draw_portfolio_curve(butterfly_call, 'Butterfly Call')
# draw_portfolio_curve(cns, 'Crash Neutral')
