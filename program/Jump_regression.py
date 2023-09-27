from program.Function import *
from program.Function_jump import *

pd.set_option('expand_frame_repr', False)

txt_path = 'C:/Users/86156/Desktop/option/data/回归结果/'

atm_straddle = pd.read_csv(os.path.join(portfolio_path, 'atm_straddle.csv'), encoding='gbk')
otm_strangle_004 = pd.read_csv(os.path.join(portfolio_path, 'otm_strangle_004.csv'), encoding='gbk')
otm_strangle_006 = pd.read_csv(os.path.join(portfolio_path, 'otm_strangle_006.csv'), encoding='gbk')
Jump_22 = pd.read_csv(os.path.join(jump_path, 'Jump_22.csv'), encoding='gbk')

Jump_22['交易日期'] = pd.to_datetime(Jump_22['交易日期'])
Jump_22['year'] = Jump_22['交易日期'].dt.year
Jump_22['month'] = Jump_22['交易日期'].dt.month

atm_straddle['交易日期'] = pd.to_datetime(atm_straddle['交易日期'])
atm_straddle['year'] = atm_straddle['交易日期'].dt.year
atm_straddle['month'] = atm_straddle['交易日期'].dt.month
atm_straddle[['year', 'month']] = atm_straddle.apply(lambda x: add_one_month(x['year'], x['month']), axis=1, result_type='expand')
straddle_jump = atm_straddle.merge(Jump_22, left_on=['year', 'month'], right_on=['year', 'month'])
straddle_jump = straddle_jump.drop(columns=['year', 'month', 'Unnamed: 0_x', 'Unnamed: 0_y'])
# straddle_jump.to_csv(os.path.join(jump_path, 'straddle_jump.csv'), encoding='gbk')

otm_strangle_004['交易日期'] = pd.to_datetime(otm_strangle_004['交易日期'])
otm_strangle_004['year'] = otm_strangle_004['交易日期'].dt.year
otm_strangle_004['month'] = otm_strangle_004['交易日期'].dt.month
otm_strangle_004[['year', 'month']] = otm_strangle_004.apply(lambda x: add_one_month(x['year'], x['month']), axis=1, result_type='expand')
strangle_004_jump = otm_strangle_004.merge(Jump_22, left_on=['year', 'month'], right_on=['year', 'month'])
strangle_004_jump = strangle_004_jump.drop(columns=['year', 'month', 'Unnamed: 0_x', 'Unnamed: 0_y'])
# strangle_004_jump.to_csv(os.path.join(jump_path, 'strangle_004_jump.csv'), encoding='gbk')

otm_strangle_006['交易日期'] = pd.to_datetime(otm_strangle_006['交易日期'])
otm_strangle_006['year'] = otm_strangle_006['交易日期'].dt.year
otm_strangle_006['month'] = otm_strangle_006['交易日期'].dt.month
otm_strangle_006[['year', 'month']] = otm_strangle_006.apply(lambda x: add_one_month(x['year'], x['month']), axis=1, result_type='expand')
strangle_006_jump = otm_strangle_006.merge(Jump_22, left_on=['year', 'month'], right_on=['year', 'month'])
strangle_006_jump = strangle_006_jump.drop(columns=['year', 'month', 'Unnamed: 0_x', 'Unnamed: 0_y'])
# strangle_006_jump.to_csv(os.path.join(jump_path, 'strangle_006_jump.csv'), encoding='gbk')

##############################################################################
y_straddle_mean = straddle_jump['Return'].mean()
y_straddle_std = straddle_jump['Return'].std()
# X_straddle_mean = straddle_jump[['CV_22', 'monthly_RV_call', 'intensity_22', 'size_22', 'sizevol_22']].mean()
# X_straddle_std = straddle_jump[['CV_22', 'monthly_RV_call', 'intensity_22', 'size_22', 'sizevol_22']].std()
X_straddle_mean = straddle_jump[['monthly_RV_call', 'intensity_22', 'size_22', 'sizevol_22']].mean()
X_straddle_std = straddle_jump[['monthly_RV_call', 'intensity_22', 'size_22', 'sizevol_22']].std()

straddle_jump['Return_normalized'] = (straddle_jump['Return'] - y_straddle_mean) / y_straddle_std
# straddle_jump[['CV_22_normalized', 'monthly_RV_call_normalized', 'intensity_22_normalized', 'size_22_normalized', 'sizevol_22_normalized']] = (straddle_jump[['CV_22', 'monthly_RV_call', 'intensity_22', 'size_22', 'sizevol_22']] - X_straddle_mean) / X_straddle_std
straddle_jump[['monthly_RV_call_normalized', 'intensity_22_normalized', 'size_22_normalized', 'sizevol_22_normalized']] = (straddle_jump[['monthly_RV_call', 'intensity_22', 'size_22', 'sizevol_22']] - X_straddle_mean) / X_straddle_std
#
y_straddle_normalized = straddle_jump['Return_normalized']
# X_straddle_normalized = straddle_jump[['CV_22_normalized', 'monthly_RV_call_normalized', 'intensity_22_normalized', 'size_22_normalized', 'sizevol_22_normalized']]
X_straddle_normalized = straddle_jump[['monthly_RV_call_normalized', 'intensity_22_normalized', 'size_22_normalized', 'sizevol_22_normalized']]
model_straddle_normalized = sm.OLS(y_straddle_normalized, X_straddle_normalized).fit()
#
# file_path = os.path.join(txt_path, "straddle_summary.txt")
# with open(file_path, "w", encoding="utf-8") as f:
#     f.write(model_straddle_normalized.summary().as_text())

y_straddle = straddle_jump['Return']
X_straddle = straddle_jump[['intensity_22', 'size_22', 'sizevol_22']]
X_straddle = sm.add_constant(X_straddle)
model_straddle = sm.OLS(y_straddle, X_straddle).fit()
print(model_straddle.summary())

y_strangle_004_mean = strangle_004_jump['Return'].mean()
y_strangle_004_std = strangle_004_jump['Return'].std()
# X_strangle_004_mean = strangle_004_jump[['CV_22', 'monthly_RV_call', 'intensity_22', 'size_22', 'sizevol_22']].mean()
# X_strangle_004_std = strangle_004_jump[['CV_22', 'monthly_RV_call', 'intensity_22', 'size_22', 'sizevol_22']].std()
X_strangle_004_mean = strangle_004_jump[['monthly_RV_call', 'intensity_22', 'size_22', 'sizevol_22']].mean()
X_strangle_004_std = strangle_004_jump[['monthly_RV_call', 'intensity_22', 'size_22', 'sizevol_22']].std()

strangle_004_jump['Return_normalized'] = (strangle_004_jump['Return'] - y_strangle_004_mean) / y_strangle_004_std
# strangle_004_jump[['CV_22_normalized', 'monthly_RV_call_normalized', 'intensity_22_normalized', 'size_22_normalized', 'sizevol_22_normalized']] = (strangle_004_jump[['CV_22', 'monthly_RV_call', 'intensity_22', 'size_22', 'sizevol_22']] - X_strangle_004_mean) / X_strangle_004_std
strangle_004_jump[['monthly_RV_call_normalized', 'intensity_22_normalized', 'size_22_normalized', 'sizevol_22_normalized']] = (strangle_004_jump[['monthly_RV_call', 'intensity_22', 'size_22', 'sizevol_22']] - X_strangle_004_mean) / X_strangle_004_std

y_strangle_004_normalized = strangle_004_jump['Return_normalized']
# X_strangle_004_normalized = strangle_004_jump[['CV_22_normalized', 'monthly_RV_call_normalized', 'intensity_22_normalized', 'size_22_normalized', 'sizevol_22_normalized']]
X_strangle_004_normalized = strangle_004_jump[['monthly_RV_call_normalized', 'intensity_22_normalized', 'size_22_normalized', 'sizevol_22_normalized']]
model_strangle_004_normalized = sm.OLS(y_strangle_004_normalized, X_strangle_004_normalized).fit()

file_path = os.path.join(txt_path, "strangle_004_summary.txt")
with open(file_path, "w", encoding="utf-8") as f:
    f.write(model_strangle_004_normalized.summary().as_text())

# y_strangle_004 = strangle_004_jump['Return']
# X_strangle_004 = strangle_004_jump[['CV_22', 'intensity_22', 'size_22', 'sizevol_22']]
# X_strangle_004 = sm.add_constant(X_strangle_004)
# model_strangle_004 = sm.OLS(y_strangle_004, X_strangle_004).fit()
# print(model_strangle_004.summary())

y_strangle_006_mean = strangle_006_jump['Return'].mean()
y_strangle_006_std = strangle_006_jump['Return'].std()
# X_strangle_006_mean = strangle_006_jump[['CV_22', 'monthly_RV_call', 'intensity_22', 'size_22', 'sizevol_22']].mean()
# X_strangle_006_std = strangle_006_jump[['CV_22', 'monthly_RV_call', 'intensity_22', 'size_22', 'sizevol_22']].std()
X_strangle_006_mean = strangle_006_jump[['monthly_RV_call', 'intensity_22', 'size_22', 'sizevol_22']].mean()
X_strangle_006_std = strangle_006_jump[['monthly_RV_call', 'intensity_22', 'size_22', 'sizevol_22']].std()

strangle_006_jump['Return_normalized'] = (strangle_006_jump['Return'] - y_strangle_006_mean) / y_strangle_006_std
# strangle_006_jump[['CV_22_normalized', 'monthly_RV_call_normalized', 'intensity_22_normalized', 'size_22_normalized', 'sizevol_22_normalized']] = (strangle_006_jump[['CV_22', 'monthly_RV_call', 'intensity_22', 'size_22', 'sizevol_22']] - X_strangle_006_mean) / X_strangle_006_std
strangle_006_jump[['monthly_RV_call_normalized', 'intensity_22_normalized', 'size_22_normalized', 'sizevol_22_normalized']] = (strangle_006_jump[['monthly_RV_call', 'intensity_22', 'size_22', 'sizevol_22']] - X_strangle_006_mean) / X_strangle_006_std

y_strangle_006_normalized = strangle_006_jump['Return_normalized']
# X_strangle_006_normalized = strangle_006_jump[['CV_22_normalized', 'monthly_RV_call_normalized', 'intensity_22_normalized', 'size_22_normalized', 'sizevol_22_normalized']]
X_strangle_006_normalized = strangle_006_jump[['monthly_RV_call_normalized', 'intensity_22_normalized', 'size_22_normalized', 'sizevol_22_normalized']]
model_strangle_006_normalized = sm.OLS(y_strangle_006_normalized, X_strangle_006_normalized).fit()

file_path = os.path.join(txt_path, "strangle_006_summary.txt")
with open(file_path, "w", encoding="utf-8") as f:
    f.write(model_strangle_006_normalized.summary().as_text())

# y_strangle_006 = strangle_006_jump['Return']
# X_strangle_006 = strangle_006_jump[['CV_22', 'intensity_22', 'size_22', 'sizevol_22']]
# X_strangle_006 = sm.add_constant(X_strangle_006)
# model_strangle_006 = sm.OLS(y_strangle_006, X_strangle_006).fit()
# print(model_strangle_006.summary())
##############################################################################
