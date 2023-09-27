# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 11:58:47 2023

@author: lenovo
"""
import tushare as ts
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
ts.set_token( '1c8b06446534ae510c8c68e38fc248b99f89ac3814cb55645ae2be72') 
pro = ts.pro_api()

def deltak(k):
    if k<=3:
        return 0.05
    elif k<=5:
        return 0.1
    elif k<=10:
        return 0.25
    elif k<=20:
        return 0.5
    elif k<=50:
        return 1
    elif k<=100:
        return 2.5
    else :
        return 5



df_ivix=pro.trade_cal(**{
    "exchange": "SSE",
    "cal_date": "",
    "start_date": 20150209,
    "end_date": 20230308,
    "is_open": 1,
    "limit": "",
    "offset": ""
}, fields=[
    "exchange",
    "cal_date"
])
df_shibor = pro.shibor(**{"start_date": '20150209',"end_date":'20230308'}, fields=["date","1w"])
df_ivix['ivix']=''


for i in range(1873,len(df_ivix['cal_date'])):
     date=df_ivix['cal_date'][i]

     df_basic = pro.opt_basic(exchange='SSE', fields='ts_code,name,per_unit,call_put,exercise_price,list_date,delist_date')
     df_basic = df_basic.loc[df_basic['name'].str.contains('50ETF')]
     df_basic = df_basic[(df_basic.list_date<=date)&(df_basic.delist_date>date)]
     
     df_basic['endday'] = pd.to_datetime(df_basic['delist_date'])
     df_basic['today'] = datetime.strptime(date,'%Y%m%d')
     df_basic['maturity'] = df_basic['endday']-df_basic['today']
#     if df_basic.drop(df_basic[df_basic.per_unit!=10000].index).empty == False:
#          df_basic = df_basic.drop(df_basic[df_basic.per_unit!=10000].index)
     dftest = df_basic.drop(df_basic[df_basic.per_unit!=10000].index)
     if len(set(dftest['maturity'].tolist())) < 2:
         if (df_basic['maturity'].min().days >= 7) or (len(set(df_basic['maturity'].tolist())) == 2):
              d1 = df_basic['maturity'].min().days
              d2 = np.sort(list(set(df_basic['maturity'].tolist())))[1].days
              T1 = d1/365
              T2 = d2/365
         else :
              d1 = np.sort(list(set(df_basic['maturity'].tolist())))[1].days
              d2 = np.sort(list(set(df_basic['maturity'].tolist())))[2].days
              T1 = d1/365
              T2 = d2/365
     else :
          df_basic.drop(df_basic[df_basic.per_unit!=10000].index,inplace=True)
          if(df_basic['maturity'].min().days >= 7) or (len(set(df_basic['maturity'].tolist())) == 2):
         
              d1 = df_basic['maturity'].min().days
              d2 = np.sort(list(set(df_basic['maturity'].tolist())))[1].days
              T1 = d1/365
              T2 = d2/365
          else :
              d1 = np.sort(list(set(df_basic['maturity'].tolist())))[1].days
              d2 = np.sort(list(set(df_basic['maturity'].tolist())))[2].days
              T1 = d1/365
              T2 = d2/365
         
     df_basic['d1'] = timedelta(d1)
     df_basic['d2'] = timedelta(d2)
     
     df_basic = df_basic.drop(['name','list_date','delist_date'],axis=1)
     
     df_basic = df_basic.drop(df_basic[(df_basic.maturity!=df_basic.d1)&(df_basic.maturity!=df_basic.d2)].index)
     
     
     opt_list = df_basic['ts_code'].tolist() # 获取50ETF期权合约列表
     df_daily = pro.opt_daily(trade_date=date,exchange = 'SSE',fields='ts_code,trade_date,settle,vol')
     df_daily = df_daily[df_daily['ts_code'].isin(opt_list)]
     
     df = pd.merge(df_basic,df_daily,how='left',on=['ts_code'])
     # risk-free risk 
     r = float(df_shibor.loc[df_shibor.date==date,'1w'])/100
     df = df.rename(columns={'exercise_price':'k', 'settle':'c'})
     
     df1 = df.drop(df[df.maturity==df.d2].index)
     df2 = df.drop(df[df.maturity==df.d1].index)
     
     df_gap1 = df1.groupby('k')['c'].aggregate(lambda group:group.max()-group.min()).to_frame()
     df_gap1['k'] = df_gap1.index
     df_gap2 = df2.groupby('k')['c'].aggregate(lambda group:group.max()-group.min()).to_frame()
     df_gap2['k'] = df_gap2.index
     
     
     
     min_gap1 = df_gap1['c'].min()
     df_gap1 = df_gap1.set_index('c')
     
     
     min_gap2 = df_gap2['c'].min()
     df_gap2 = df_gap2.set_index('c')
 
     
     if isinstance(k1,float):
          k1 = df_gap1.loc[min_gap1,'k']
     else:
          sum1 = df1[df1.k==k1.tolist()[0]]['vol'].sum()
          sum2 = df1[df2.k==k1.tolist()[1]]['vol'].sum()
          if sum1 > sum2:
               k1 = k1.tolist()[0]
          else:
               k1 = k1.tolist()[1]
     if isinstance(k2,float):
          k2 = df_gap1.loc[min_gap1,'k']
     else:
          sum1 = df2[df2.k==k2.tolist()[0]]['vol'].sum()
          sum2 = df2[df2.k==k2.tolist()[1]]['vol'].sum()
          if sum1 > sum2:
               k2 = k2.tolist()[0]
          else:
               k2 = k2.tolist()[1]
          
          
     F1 = k1+math.exp(r*T1)*min_gap1
     F2 = k2+math.exp(r*T2)*min_gap2
     if np.isnan(F1).any() or np.isnan(F2).any() :
          continue
     
     df_gap1['F1'] = F1
     df_gap2['F2'] = F2
     
     df_gap1['F1-k'] = df_gap1['F1']-df_gap1['k']
     df_gap2['F2-k'] = df_gap2['F2']-df_gap2['k']
     
     #求出K01
     df_temp1 = df_gap1[df_gap1['F1-k']>0]
     temp1 = df_temp1['F1-k'].min()
     df_temp1 = df_temp1.set_index('F1-k')
     K01 = df_temp1.loc[temp1,'k']
     
     #求出K02
     df_temp2 = df_gap2[df_gap2['F2-k']>0]
     temp2 = df_temp2['F2-k'].min()
     df_temp2 = df_temp2.set_index('F2-k')
     K02 = df_temp2.loc[temp2,'k']
     
     df1_put = df1.drop(df1[df1.call_put=='C'].index)
     df1_call = df1.drop(df1[df1.call_put=='P'].index)
     
     df2_put = df2.drop(df2[df2.call_put=='C'].index)
     df2_call = df2.drop(df2[df2.call_put=='P'].index)
     
     df1_put = df1_put.set_index('k')
     df1_call = df1_call.set_index('k')
     df2_put = df2_put.set_index('k')
     df2_call = df2_call.set_index('k')
     
     df_mix1=pd.DataFrame()
     df_mix2=pd.DataFrame()
     
     df_mix1['k'] = df1['k'].drop_duplicates().sort_values()
     df_mix2['k'] = df2['k'].drop_duplicates().sort_values()
     
     df_mix1 = df_mix1.set_index(np.arange(0,len(df_mix1['k'])))
     df_mix1['mix']=''
     df_mix1['delta_K']=''
     df_mix1['fcgx']=''
     df_mix2 = df_mix2.set_index(np.arange(0,len(df_mix2['k'])))
     df_mix2['mix']=''
     df_mix2['delta_K']=''
     df_mix2['fcgx']=''
     
     #近月合约
     for j in range(len(df_mix1['k'])):
         if df_mix1['k'][j] < K01:
             df_mix1.iloc[j,1] = df1_put.loc[df_mix1['k'][j],'c'] #虚值看跌期权价格
         elif df_mix1['k'][j] > K01:
             df_mix1.iloc[j,1] = df1_call.loc[df_mix1['k'][j],'c'] #虚值看涨期权价格
         else:
             df_mix1.iloc[j,1] = (df1_call.loc[df_mix1['k'][j],'c']+df1_put.loc[df_mix1['k'][j],'c'])/2 #平值期权价格
         
         if j==0:
             df_mix1.iloc[j,2]=deltak(df_mix1.iloc[j,0])
         else:
             df_mix1.iloc[j,2]=df_mix1.iloc[j,0]-df_mix1.iloc[j-1,0]
         df_mix1.iloc[j,3]=(df_mix1.iloc[j,2]/df_mix1.iloc[j,0]**2)*math.exp(r*T1)*df_mix1.iloc[j,1]
         
     #次近月合约
     for l in range(len(df_mix2['k'])):
         if df_mix2['k'][l] < K02:
             df_mix2.iloc[l,1] = df2_put.loc[df_mix2['k'][l],'c'] #虚值看跌期权价格
         elif df_mix2['k'][l] > K02:
             df_mix2.iloc[l,1] = df2_call.loc[df_mix2['k'][l],'c'] #虚值看涨期权价格
         else:
             df_mix2.iloc[l,1] = (df2_call.loc[df_mix2['k'][l],'c']+df2_put.loc[df_mix2['k'][l],'c'])/2 #平值期权价格
             
         if l==0:
             df_mix2.iloc[l,2]=deltak(df_mix2.iloc[l,0])
         else:
             df_mix2.iloc[l,2]=df_mix2.iloc[l,0]-df_mix2.iloc[l-1,0]
         df_mix2.iloc[l,3]=(df_mix2.iloc[l,2]/df_mix2.iloc[l,0]**2)*math.exp(r*T2)*df_mix2.iloc[l,1]
         
     
     variance1=(2/T1)*df_mix1['fcgx'].sum()-(1/T1)*(F1/K01-1)**2
     variance2=(2/T2)*df_mix2['fcgx'].sum()-(1/T2)*(F2/K02-1)**2
     
     iVIX=100*math.sqrt((T1*variance1*((d2-30)/(d2-d1))+T2*variance2*((30-d1)/(d2-d1)))*(365/30))
     df_ivix.iloc[i,2]=iVIX

df_ivix.to_excel('C:\\Users\\lenovo\\Desktop\\ivix.xlsx',index=False)
df_ivix.to_csv('C:\\Users\\lenovo\\Desktop\\ivix_2.csv',index=False)
