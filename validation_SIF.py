# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def matlab_datenum_to_datetime(datenum):
    # MATLAB datenum 是从 0000-01-01 开始的天数，Python datetime 从 1970-01-01 开始
    # MATLAB datenum 的整数部分是天数，小数部分是一天中的时间
    days = int(datenum)
    fraction = datenum - days
    dt = datetime(1, 1, 1) + pd.to_timedelta(days - 366, unit='D') + pd.to_timedelta(fraction * 24 * 3600, unit='s')
    return dt

# function to matlab datetime to python datetime
def matlab2datetime(matlab_datenum):
    # e.g. 739449.88170052 to 2024-07-17T05:09:40
    python_datetime = pd.to_datetime(matlab_datenum - 719529, unit='D')
    # plus 8 hours for timezone correction (if needed)
    python_datetime += pd.Timedelta(hours=2)
    return python_datetime

sif_mat = r'E:\Datahub\Barbeau\Data_SIF\SIF3data\2022\PROCESSED_matlab\L2\SIF_2022.csv'
sif_py = r'E:\Datahub\Barbeau\Data_SIF\SIF3data\2022\PROCESSED\L2\Yearly\PROSIF_SIFresults_2022_Yearly.csv'

df_mat = pd.read_csv(sif_mat)
df_mat['SIF_3FLD_O2A'] = df_mat['SIF_3FLD_O2A'] * 1e3 # 转换为 mW/m^2/sr/nm

df_py = pd.read_csv(sif_py)
df_py['SIF_3FLD'] = df_py['SIF_3FLD']/ np.pi
print(df_mat.loc[0, 'Time_start'], df_py.loc[152, 'Time_start'])

# # 根据df_mat的Time_start列，匹配df_py的Time_start列，找到对应的行
# df_match = df_mat.copy()
# for i in range(len(df_mat)):
#     time_mat = df_mat.loc[i, 'Time_start']
#     # 在df_py中找到与time_mat_dt最接近的时间
#     df_py['Time_diff'] = abs(df_py['Time_start'] - time_mat)
#     idx_closest = df_py['Time_diff'].idxmin()
#     if df_py.loc[idx_closest, 'Time_diff'] < 0.0001:
#         # 将df_py中的SIF_3FLD值赋给df_match的SIF_3FLD列
#         df_match.loc[i, 'Time_start_py'] = df_py.loc[idx_closest, 'Time_start']
#         df_match.loc[i, 'SIF_3FLD_py'] = df_py.loc[idx_closest, 'SIF_3FLD']
#         df_match.loc[i, 'Time_diff'] = df_py.loc[idx_closest, 'Time_diff']
#     else:
#         df_match.loc[i, 'Time_start_py'] = np.nan
#         df_match.loc[i, 'SIF_3FLD_py'] = np.nan
#         df_match.loc[i, 'Time_diff'] = np.nan

# df_match.loc[(df_match['SIF_3FLD_O2A'] > 10) | (df_match['SIF_3FLD_O2A'] < 0), 'SIF_3FLD_O2A'] = np.nan
# df_match.loc[(df_match['SIF_3FLD_py'] > 10) | (df_match['SIF_3FLD_py'] < 0), 'SIF_3FLD_py'] = np.nan
# df_match.to_csv(r'E:\Datahub\Barbeau\Data_SIF\SIF3data\2022\PROCESSED\L2\Yearly\PROSIF_SIFresults_2022_Yearly_matched.csv', index=False)

df_match = pd.read_csv(r'E:\Datahub\Barbeau\Data_SIF\SIF3data\2022\PROCESSED\L2\Yearly\PROSIF_SIFresults_2022_Yearly_matched.csv')
idx = (df_match['prcdarkpix_sig'] < 0.025) & (df_match['I1_prcdarkpix_sig'] < 0.025) \
    & (df_match['I2_prcdarkpix_sig'] < 0.025)
df_match = df_match.loc[idx, :]
df_match = df_match.dropna(subset=['SIF_3FLD_O2A', 'SIF_3FLD_py'])

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(df_match['SIF_3FLD_O2A'], df_match['SIF_3FLD_py'], marker = '.', color = 'b')
ax.plot([0, 10], [0, 10], 'r--') # 添加 y=x 的参考线
ax.set_xlim([0, 10])
ax.set_ylim([0, 10])
# r2
r2 = np.corrcoef(df_match['SIF_3FLD_O2A'], df_match['SIF_3FLD_py'])[0, 1] ** 2
ax.text(0.5, 0.9, f'$R^2$ = {r2:.4f}', transform=ax.transAxes, fontsize=12)
ax.set_xlabel('SIF 3-FLD O2A (MATLAB) [mW/m^2/sr/nm]')
ax.set_ylabel('SIF 3-FLD (Python) [mW/m^2/sr/nm]')
ax.set_title('Comparison of SIF 3-FLD Retrievals')
plt.show()


# fig, ax = plt.subplots(figsize=(10, 6))
# # idx_mat =
# idx_py = df_py['prcsppix_sat'] == 0
# ax.scatter(df_mat['Time_start'], df_mat['SIF_3FLD_O2A'], marker = '.', 
#            color = 'b',label='MATLAB')
# ax.scatter(df_py.loc[idx_py, 'Time_start'], df_py.loc[idx_py, 'SIF_3FLD'], 
#            marker = '.', color= 'orange',label='Python', linestyle='--')
# ax.set_ylim([0, 5])
# ax.legend()
# ax.set_xlabel('Time')
# plt.show()