import os.path

from program.Function_jump import *

pd.set_option('expand_frame_repr', False)

df_etf = pd.read_csv(r'C:\Users\86156\Desktop\option\data\ETF数据\50ETF_1min.csv',
                     encoding='gbk',
                     parse_dates=['bob']
                     )

df_etf['bob'] = pd.to_datetime(df_etf['bob'], utc=True)
df_etf['交易日期'] = df_etf['bob'].dt.date
df_etf = df_etf[df_etf['bob'] >= '2016-02-01']
df_etf.set_index('bob', inplace=True)

df_etf['min_returns'] = np.log(df_etf['close'] / df_etf['pre_close'])
grouped_ret = df_etf.groupby('交易日期')

n_days = len(grouped_ret)
ret_matrix = np.empty((0, 240))

for date, group in grouped_ret:
    returns = group['min_returns'].values[1:]
    group_median = np.median(returns)
    if len(returns) < 240:
        returns = np.pad(returns, (0, 240 - len(returns)), mode='constant', constant_values=group_median)
    ret_matrix = np.vstack((ret_matrix, returns))

median_returns = np.median(ret_matrix, axis=1)

RV_t = []
for row in ret_matrix:
    rv = np.sum(row ** 2)
    RV_t.append(rv)
RV_t = np.array(RV_t)

BV_t = []
for row in ret_matrix:
    bv = 0.5 * np.pi * np.sum(np.abs(row[:-1]) * np.abs(row[1:]))
    BV_t.append(bv)
BV_t = np.array(BV_t)

n = len(ret_matrix[0])
TQ_t = []
for row in ret_matrix:
    tq = (1 / n) * 1.7432 * np.sum(
        np.abs(row[:-2]) ** (3 / 4) * np.abs(row[1:-1]) ** (3 / 4) * np.abs(row[2:]) ** (3 / 4))
    TQ_t.append(tq)
TQ_t = np.array(TQ_t)

Z_score = []
for row, rv, bv, tq in zip(ret_matrix, RV_t, BV_t, TQ_t):
    z = ((n ** 0.5) * (rv - bv) * (rv ** (-1))) / (0.6087 * max(1, tq * (bv ** (-2)))) ** 0.5
    Z_score.append(z)
Z_score = np.array(Z_score)

alpha = 0.05
critical_value = norm.ppf(1 - alpha)

J_mark = []
for rv, bv, z in zip(RV_t, BV_t, Z_score):
    j = (rv - bv) * (z > critical_value)
    J_mark.append(j)
J_mark = np.array(J_mark)

CV = []
cv = 0
for rv, bv, j in zip(RV_t, BV_t, J_mark):
    if np.abs(j) == 0:
        cv = rv
    elif np.abs(j) > 0:
        cv = bv
    CV.append(cv)
CV = np.array(CV)

# jumps = find_jumps(ret_matrix, alpha)
# jumps.to_csv('jumps.csv', encoding='gbk')
# print("Jump locations:", jumps)
# exit()

T = len(ret_matrix)
h_values = [1, 5, 22]

CV_kh_dict = {}
for h in h_values:
    M = T // h
    CV_kh = compute_cv_kh(CV, h, M)
    CV_kh_dict[h] = CV_kh

intensity_kh_dict = {}
for h in h_values:
    M = T // h
    intensity_kh = compute_intensity_kh(J_mark, h, M)
    intensity_kh_dict[h] = intensity_kh

J_ti = calculate_j_ti(ret_matrix, Z_score, alpha)

size_kh_dict = {}
for h in h_values:
    M = T // h
    size_kh = calculate_size_kh(J_ti, h, M, intensity_kh_dict[h])
    size_kh_dict[h] = size_kh

sizevol_kh_dict = {}
for h in h_values:
    M = T // h
    sizevol_kh = calculate_sizevol_kh(J_ti, h, M, intensity_kh_dict[h], size_kh_dict[h])
    sizevol_kh_dict[h] = sizevol_kh

data_1 = {
    'CV_1': CV_kh_dict[1],
    'intensity_1': intensity_kh_dict[1],
    'size_1': size_kh_dict[1],
    'sizevol_1': sizevol_kh_dict[1],
}

data_5 = {
    'CV_5': CV_kh_dict[5],
    'intensity_5': intensity_kh_dict[5],
    'size_5': size_kh_dict[5],
    'sizevol_5': sizevol_kh_dict[5],
}

data_22 = {
    'CV_22': CV_kh_dict[22],
    'intensity_22': intensity_kh_dict[22],
    'size_22': size_kh_dict[22],
    'sizevol_22': sizevol_kh_dict[22],
}

jump_1 = pd.DataFrame(data_1)
jump_5 = pd.DataFrame(data_5)
jump_22 = pd.DataFrame(data_22)

unique_dates = df_etf['交易日期'].drop_duplicates().reset_index(drop=True)

jump_1['交易日期'] = unique_dates
jump_5_dates = unique_dates[4::5]
jump_5['交易日期'] = jump_5_dates.reset_index(drop=True)

jump_22_dates = unique_dates[21::22]
jump_22['交易日期'] = jump_22_dates.reset_index(drop=True)

jump_1 = jump_1[['交易日期'] + [col for col in jump_1.columns if col != '交易日期']]
jump_5 = jump_5[['交易日期'] + [col for col in jump_5.columns if col != '交易日期']]
jump_22 = jump_22[['交易日期'] + [col for col in jump_22.columns if col != '交易日期']]

jump_1.to_csv(os.path.join(jump_path, 'Jump_1.csv'), encoding='gbk')
jump_5.to_csv(os.path.join(jump_path, 'Jump_5.csv'), encoding='gbk')
jump_22.to_csv(os.path.join(jump_path, 'Jump_22.csv'), encoding='gbk')
