from program.Function import *

pd.set_option('expand_frame_repr', False)

put_atm = pkl_to_frame('put_atm')
put_itm = pkl_to_frame('put_itm')
put_otm = pkl_to_frame('put_otm')
call_atm = pkl_to_frame('call_atm')
call_itm = pkl_to_frame('call_itm')
call_otm = pkl_to_frame('call_otm')

call_atm = call_opt_return(call_atm)
call_otm = call_opt_return(call_otm)
call_itm = call_opt_return(call_itm)
put_atm = put_opt_return(put_atm)
put_otm = put_opt_return(put_otm)
put_itm = put_opt_return(put_itm)

call_atm['IV'] = call_atm.apply(calculate_iv, axis=1)
call_otm['IV'] = call_otm.apply(calculate_iv, axis=1)
call_itm['IV'] = call_itm.apply(calculate_iv, axis=1)
put_atm['IV'] = put_atm.apply(calculate_iv, axis=1)
put_otm['IV'] = put_otm.apply(calculate_iv, axis=1)
put_itm['IV'] = put_itm.apply(calculate_iv, axis=1)

call_itm['在值程度'] = (call_itm['行权价'] / call_itm['现期ETF收盘价']).apply(lambda x: round(x, 2))
call_otm['在值程度'] = (call_otm['行权价'] / call_otm['现期ETF收盘价']).apply(lambda x: round(x, 2))
call_atm['在值程度'] = 1
put_otm['在值程度'] = (put_otm['行权价'] / put_otm['现期ETF收盘价']).apply(lambda x: round(x, 2))
put_itm['在值程度'] = (put_itm['行权价'] / put_itm['现期ETF收盘价']).apply(lambda x: round(x, 2))
put_atm['在值程度'] = 1

call_atm.to_csv(os.path.join(option_path, 'call_atm_m_ret.csv'), encoding='gbk')
call_otm.to_csv(os.path.join(option_path, 'call_otm_m_ret.csv'), encoding='gbk')
call_itm.to_csv(os.path.join(option_path, 'call_itm_m_ret.csv'), encoding='gbk')
put_atm.to_csv(os.path.join(option_path, 'put_atm_m_ret.csv'), encoding='gbk')
put_otm.to_csv(os.path.join(option_path, 'put_otm_m_ret.csv'), encoding='gbk')
put_itm.to_csv(os.path.join(option_path, 'put_itm_m_ret.csv'), encoding='gbk')
