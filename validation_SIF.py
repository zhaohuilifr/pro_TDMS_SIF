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
    # days = int(datenum)
    # fraction = datenum - days
    # dt = datetime(1, 1, 1) + pd.to_timedelta(days - 366, unit='D') + pd.to_timedelta(fraction * 24 * 3600, unit='s')
    dt = pd.to_datetime(datenum - 719529, unit='D')
    hour = dt.hour + dt.minute / 60 + dt.second / 3600 + 7.0  # offset 7 hours
    doy = dt.timetuple().tm_yday + hour / 24
    return dt, doy, hour

# function to matlab datetime to python datetime
def matlab2datetime(matlab_datenum):
    # e.g. 739449.88170052 to 2024-07-17T05:09:40
    python_datetime = pd.to_datetime(matlab_datenum - 719529, unit='D')
    # plus 8 hours for timezone correction (if needed)
    python_datetime += pd.Timedelta(hours=2)
    return python_datetime

def filter_3sigma(array):
    mean = np.nanmean(array)
    std = np.nanstd(array)
    array[~(array >= mean - 3 * std) & (array <= mean + 3 * std)] = np.nan
    if np.sum(~(array >= mean - 3 * std) & (array <= mean + 3 * std)) > 0:
        print('number of outliers filtered:', np.sum(~(array >= mean - 3 * std) & (array <= mean + 3 * std)))
    return array

path_flux = r'E:\Datahub\Barbeau\Data_flux\Daniel'
df_flux = pd.read_excel(os.path.join(path_flux, 'Barbeau_2022_LI7500_noAoA_uthvar_hh_DoubleInstrumentGF.xls'))
sif_mat = r'E:\Datahub\Barbeau\Data_SIF\SIF3data\2022\PROCESSED_matlab\L2\SIF_2022.csv'
sif_py = r'E:\Datahub\Barbeau\Data_SIF\SIF3data\2022\PROCESSED\L2\Yearly\PROSIF_SIFresults_2022_Yearly.csv'
savepath = r'E:\Datahub\Barbeau\Data_SIF\SIF3data\2022\PROCESSED\L2\Yearly'

df_mat = pd.read_csv(sif_mat)
df_mat['SIF_3FLD_O2A'] = df_mat['SIF_3FLD_O2A'] * 1e3 # 转换为 mW/m^2/sr/nm

df_py = pd.read_csv(sif_py)
df_py['SIF_3FLD'] = df_py['SIF_3FLD']/ np.pi
print(df_mat.loc[0, 'Time_start'], df_py.loc[152, 'Time_start'])

# %% 根据df_mat的Time_start列，匹配df_py的Time_start列，找到对应的行
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

# df_match['DOY'] = df_match['Time_start'].apply(lambda x: matlab_datenum_to_datetime(x)[1])
# df_match['Hour'] = df_match['Time_start'].apply(lambda x: matlab_datenum_to_datetime(x)[2])
# # idx = (df_match['prcdarkpix_sig'] < 0.025) & (df_match['I1_prcdarkpix_sig'] < 0.025) \
# #     & (df_match['I2_prcdarkpix_sig'] < 0.025) & (df_match['Hour'] >= 7) & \
# #         (df_match['Hour'] <= 18) & (df_match['DOY'] >= 100) & (df_match['DOY'] <= 300)
# idx = (df_match['Hour'] >= 7) & (df_match['Hour'] <= 18) & \
#     (df_match['DOY'] >= 100) & (df_match['DOY'] <= 300)
# df_match = df_match.loc[idx, :]
# df_match = df_match.reset_index(drop=True)
# # filter sigma outliers for each halfhour window from 7:00 to 18:00
# dt_start = int(df_match.loc[0, 'DOY'])
# dt_end = int(df_match.loc[len(df_match)-1, 'DOY'])
# for doy in range(dt_start, dt_end+1):
#     for hour in np.arange(7, 18, 0.5):
#         idx_window = (df_match['DOY'] >= doy) & (df_match['DOY'] < doy + 1) & (df_match['Hour'] >= hour) & (df_match['Hour'] < hour + 0.5)
#         df_match.loc[idx_window, 'SIF_3FLD_O2A'] = filter_3sigma(df_match.loc[idx_window, 'SIF_3FLD_O2A'])
#         df_match.loc[idx_window, 'SIF_3FLD_py'] = filter_3sigma(df_match.loc[idx_window, 'SIF_3FLD_py'])
#     idx_day_window = (df_match['DOY'] >= doy) & (df_match['DOY'] < doy + 1)
#     df_match.loc[idx_day_window, 'SIF_3FLD_O2A'] = filter_3sigma(df_match.loc[idx_day_window, 'SIF_3FLD_O2A'])
#     df_match.loc[idx_day_window, 'SIF_3FLD_py'] = filter_3sigma(df_match.loc[idx_day_window, 'SIF_3FLD_py'])

# df_match.to_csv(r'E:\Datahub\Barbeau\Data_SIF\SIF3data\2022\PROCESSED\L2\Yearly\PROSIF_SIFresults_2022_Yearly_matched.csv', index=False)

# %%
# df_match = pd.read_csv(r'E:\Datahub\Barbeau\Data_SIF\SIF3data\2022\PROCESSED\L2\Yearly\PROSIF_SIFresults_2022_Yearly_matched.csv')
# df_match.loc[(df_match['SIF_3FLD_O2A'] > 5) | (df_match['SIF_3FLD_O2A'] < 0), 'SIF_3FLD_O2A'] = np.nan
# df_match.loc[(df_match['SIF_3FLD_py'] > 5) | (df_match['SIF_3FLD_py'] < 0), 'SIF_3FLD_py'] = np.nan

# # %% match df_match with df_flux based on Time_start
# dt_start = int(df_match.loc[0, 'DOY'])
# dt_end = int(df_match.loc[len(df_match)-1, 'DOY'])
# hour_start = 7+0.5 
# hour_end = 18+0.5
# df_flux = df_flux.loc[(df_flux['jj'] >= dt_start) & (df_flux['jj'] <= dt_end) \
#                        & (df_flux['hh'] >= hour_start) & (df_flux['hh'] <= hour_end), :]# [7, 18)
# df_flux = df_flux.reset_index(drop=True)
# for i in range(len(df_flux)):
#     day_flux = df_flux.loc[i, 'jj']
#     hour_flux = df_flux.loc[i, 'hh']
#     # if hour_flux == 0:
#     #     day_flux -= 1
#     #     hour_flux = 24
#     idx_window = (df_match['DOY'] >= day_flux) & (df_match['DOY'] < day_flux + 1) \
#         & (df_match['Hour'] >= hour_flux - 0.5) & (df_match['Hour'] < hour_flux)
#     # # 计算sif_3fld_02a的nan比例
#     # nan_ratio = np.sum(df_match.loc[idx_window, 'SIF_3FLD_O2A'].isna()) / len(df_match.loc[idx_window, 'SIF_3FLD_O2A'])
#     # if nan_ratio > 0.8:
#     #     print(f'Warning: More than 80% of SIF_3FLD_O2A values are NaN for day {day_flux} hour {hour_flux}')
#     #     df_flux.loc[i, 'SIF_3FLD_O2A_mean'] = np.nan
#     # else:
#     #     df_flux.loc[i, 'SIF_3FLD_O2A_mean'] = np.nanmean(df_match.loc[idx_window, 'SIF_3FLD_O2A'])
#     # nan_ratio_py = np.sum(df_match.loc[idx_window, 'SIF_3FLD_py'].isna()) / len(df_match.loc[idx_window, 'SIF_3FLD_py'])
#     # if nan_ratio_py > 0.8:
#     #     print(f'Warning: More than 80% of SIF_3FLD_py values are NaN for day {day_flux} hour {hour_flux}')
#     #     df_flux.loc[i, 'SIF_3FLD_py_mean'] = np.nan
#     # else:
#     #     df_flux.loc[i, 'SIF_3FLD_py_mean'] = np.nanmean(df_match.loc[idx_window, 'SIF_3FLD_py'])
    
#     df_flux.loc[i, 'SIF_3FLD_O2A_mean'] = np.nanmean(df_match.loc[idx_window, 'SIF_3FLD_O2A'])
#     df_flux.loc[i, 'SIF_3FLD_py_mean'] = np.nanmean(df_match.loc[idx_window, 'SIF_3FLD_py'])

# df_flux['DOY'] = df_flux['jj'] + (df_flux['hh'] - 0.5) / 24.0
# # 将DOY列插入到jj之后
# cols = df_flux.columns.tolist()
# cols.insert(cols.index('jj') + 1, cols.pop(cols.index('DOY')))
# df_flux = df_flux[cols]
# # DOY >=104 and DOY<261
# idx = (df_flux['DOY'] >= 104) & (df_flux['DOY'] < 261)
# df_flux = df_flux.loc[idx, :]
# df_flux = df_flux.reset_index(drop=True)
# # outliers filtering 
# df_flux.loc[(df_flux['GPP'] < 0), 'GPP'] = np.nan
# df_flux.to_csv(savepath + '\Barbeau_2022_LI7500_noAoA_uthvar_hh_DoubleInstrumentGF_with_SIF.csv', index=False)

# %% plot the relationship between GPP and SIF_3FLD_O2A_mean and SIF_3FLD_py_mean
df_flux = pd.read_csv(savepath + '\Barbeau_2022_LI7500_noAoA_uthvar_hh_DoubleInstrumentGF_with_SIF.csv')
df_flux = df_flux.loc[(df_flux['hh'] >= 7.5) & (df_flux['hh'] <= 18.5), :]
df_flux = df_flux.reset_index(drop=True)

fig = plt.figure(figsize=(8, 6))
gs = fig.add_gridspec(2, 2)
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
ax_bottom = fig.add_subplot(gs[1, :])  # 跨两列
ax0.scatter(df_flux['SIF_3FLD_O2A_mean'], df_flux['GPP'], marker = '.', color = 'b')
ax0.set_xlabel('SIF 3-FLD O2A (MATLAB) [mW/m^2/sr/nm]')
ax0.set_ylabel('GPP (μmol/m^2/s)')
ax0.set_title('Relationship between GPP and SIF 3-FLD O2A (MATLAB)')
ax1.scatter(df_flux['SIF_3FLD_py_mean'], df_flux['GPP'], marker = '.', color = 'orange')
ax1.set_xlabel('SIF 3-FLD (Python) [mW/m^2/sr/nm]')
ax1.set_ylabel('GPP (μmol/m^2/s)')
ax1.set_title('Relationship between GPP and SIF 3-FLD (Python)')

fig, ax = plt.subplots(2, 8, figsize=(28, 8))
fig.subplots_adjust(wspace=0.6, hspace=0.3)
doys = [106, 133, 167, 191, 199, 215, 222, 225]
for i, doy in enumerate(doys):
    idx = df_flux['jj'] == doy
    ax[0, i].plot(df_flux.loc[idx, 'hh'], df_flux.loc[idx, 'PAR (µmol/m2/s)'], 
                          marker = '.', color = 'b', label='PAR')
    axi = ax[0, i].twinx()
    ax[0, i].set_ylim(0, 2000)
    axi.set_ylim(0, 50)
    axi.plot(df_flux.loc[idx, 'hh'], df_flux.loc[idx, 'GPP'], 
                          marker = '.', color = 'orange', label='GPP')
    ax[1,i].plot(df_flux.loc[idx, 'hh'], df_flux.loc[idx, 'SIF_3FLD_O2A_mean'], 
                          marker = '.', color = 'b', label='SIF 3-FLD (MATLAB)')
    ax[1,i].plot(df_flux.loc[idx, 'hh'], df_flux.loc[idx, 'SIF_3FLD_py_mean'], 
                          marker = '.', color = 'orange', label='SIF 3-FLD (Python)')
    ax[0, i].set_xlabel('Hour')
    ax[1,i].set_ylim([0, 2])
    if i == 0:
        ax[0, i].set_ylabel('PAR (μmol/m^2/s)')
    if i == 7:
        axi.set_ylabel('GPP (μmol/m^2/s)')
    ax[0, i].set_title(f'DOY {doy}')
    ax[1, i].set_xlabel('Hour')
    if i == 0:
        ax[1, i].set_ylabel('SIF (mW/m^2/sr/nm)')
    ax[1, i].set_title(f'DOY {doy}')
    if i == 0:
        ax[0, i].legend(loc='upper left')
        axi.legend(loc='upper right')
        ax[1, i].legend(loc='upper left')


fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(2, 2)
ax0 = fig.add_subplot(gs[0, :])
# ax1 = fig.add_subplot(gs[0, 1])
ax_bottom = fig.add_subplot(gs[1, :])  # 跨两列

ax0.scatter(df_flux['SIF_3FLD_O2A_mean'], df_flux['SIF_3FLD_py_mean'], marker = '.', color = 'b')
ax0.plot([0, 3], [0, 3], 'r--') # 添加 y=x 的参考线
ax0.set_xlim([0, 3])
ax0.set_ylim([0, 3])
dfs = df_flux.loc[:, ['SIF_3FLD_O2A_mean', 'SIF_3FLD_py_mean']].dropna()
r2 = np.corrcoef(dfs['SIF_3FLD_O2A_mean'], dfs['SIF_3FLD_py_mean'])[0, 1] ** 2
ax0.text(0.5, 0.9, f'$R^2$ = {r2:.4f}', transform=ax0.transAxes, fontsize=12)
ax0.set_xlabel('SIF 3-FLD O2A (MATLAB) [mW/m^2/sr/nm]')
ax0.set_ylabel('SIF 3-FLD (Python) [mW/m^2/sr/nm]')
ax0.set_title('Comparison of SIF 3-FLD Retrievals')

# ax1.scatter(df_flux['SIF_3FLD_O2A_YG']/1e2, df_flux['SIF_3FLD_py'], marker = '.', color = 'orange')
# ax1.plot([0, 10], [0, 10], 'r--') # 添加 y=x 的参考线
# ax1.set_xlim([0, 10])
# ax1.set_ylim([0, 10])
# r2 = np.corrcoef(df_flux['SIF_3FLD_O2A_YG']/1e2, df_flux['SIF_3FLD_py'])[0, 1] ** 2
# ax1.text(0.5, 0.9, f'$R^2$ = {r2:.4f}', transform=ax1.transAxes, fontsize=12)
# ax1.set_xlabel('SIF 3-FLD O2A (MATLAB) [mW/m^2/sr/nm]')
# ax1.set_ylabel('SIF 3-FLD (Python) [mW/m^2/sr/nm]')
# ax1.set_title('Comparison of SIF 3-FLD Retrievals')

# ax_bottom合并绘制df_flux['SIF_3FLD_O2A']和df_flux['SIF_3FLD_O2A_YG']的时间序列图
ax_bottom.scatter(df_flux['DOY'], df_flux['SIF_3FLD_O2A_mean'], label='SIF 3-FLD (MATLAB)', marker = '.',color = 'b')
ax_bottom.scatter(df_flux['DOY'], df_flux['SIF_3FLD_py_mean'], label='SIF 3-FLD (Python)', marker = '.', color = 'orange')
ax_bottom.set_xlabel('Time')
ax_bottom.set_ylabel('SIF 3-FLD (mW/m^2/sr/nm)')
ax_bottom.set_title('Time Series of SIF 3-FLD Retrievals')
ax_bottom.legend()


plt.show()