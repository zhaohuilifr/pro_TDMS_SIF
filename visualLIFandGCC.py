# # !/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.signal import savgol_filter

# 全局设置：使用 Times New Roman 字体，字号为 14
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14

def sg_smooth(y, window_length=11, polyorder=2):
    """Apply Savitzky-Golay filter to smooth the data."""
    return savgol_filter(y, window_length=window_length, polyorder=polyorder)

def calculate_fractional_doy(series):
    """高效计算带小数部分的年内天数(DOY)"""
    doy_base = series.dt.dayofyear
    fraction = (
        series.dt.hour * 3600
        + series.dt.minute * 60
        + series.dt.second + series.dt.microsecond / 1e6
    ) / 86400.0
    return doy_base + fraction

def matlab_datenum_to_datetime(yearstr, datenum, offset_hours=5):
    # MATLAB datenum 是从 0000-01-01 开始的天数，Python datetime 从 1970-01-01 开始
    # MATLAB datenum 的整数部分是天数，小数部分是一天中的时间
    # days = int(datenum)
    # fraction = datenum - days
    # dt = datetime(1, 1, 1) + pd.to_timedelta(days - 366, unit='D') + pd.to_timedelta(fraction * 24 * 3600, unit='s')
    if yearstr == '2025':
        offset_hours = 2 # 2025年数据的时间已经是UTC时间，不需要再加上5小时的偏移
    dt = pd.to_datetime(datenum - 719529, unit='D') + pd.Timedelta(hours=offset_hours)
    hour = dt.hour + dt.minute / 60 + dt.second / 3600
    doy = dt.timetuple().tm_yday + hour / 24
    return dt, doy, hour

yearstr = '2026'
path_lif = r'E:\Datahub\Barbeau\Data_LIF\A_LIF_PAR_Time_Cor\µLIDAR_situ_data_Barbeau\PROCESSED\{year}\L2\Yearly\{year}_LIF.csv'.format(year=yearstr)
path_gcc = r'E:\Datahub\Barbeau\Data_Camera\GCC\{year}\all_gcc_{year}_filtered.csv'.format(year=yearstr)
# path_vis = r'E:\Datahub\Barbeau\Data_SIF\SIF3data\{year}\PROCESSED\L2\Yearly\PROSIF_VIsresults_{year}_Yearly.csv'.format(year=yearstr)
save_path = r'E:\Datahub\Barbeau\Data_Camera\GCC\{year}'.format(year=yearstr)

data_lif = pd.read_csv(path_lif)
data_gcc = pd.read_csv(path_gcc)
# data_vis = pd.read_csv(path_vis)
data_lif['datetime'] = pd.to_datetime(data_lif['time_01011904'], unit='s', origin=pd.Timestamp('1904-01-01'))
data_gcc['datetime'] = pd.to_datetime(data_gcc['datetime'], format='%Y-%m-%d %H:%M:%S')
# data_vis['datetime'] = data_vis['Time_mid'].apply(lambda x: matlab_datenum_to_datetime(yearstr, x)[0])
# data_vis['DOY'] = data_vis['Time_mid'].apply(lambda x: matlab_datenum_to_datetime(yearstr, x)[1])

data_lif.set_index('datetime', inplace=True)
data_gcc.set_index('datetime', inplace=True)
# data_vis.set_index('datetime', inplace=True)

# resample to 30 mins (pandas>=2.2 prefers 'min' over 'T')
df_fs_30min = data_lif['Fs'].resample('30min').mean().dropna().to_frame()
df_fs_30min.reset_index(inplace=True)
# df_vis_30min = data_vis[['mNDI705','NDVI','PRI','CIrededge','MTCI']].resample('30min').mean().dropna()
# df_vis_30min.reset_index(inplace=True)

df_fs_30min['DOY'] = calculate_fractional_doy(df_fs_30min['datetime'])
data_gcc['DOY'] = calculate_fractional_doy(data_gcc.index.to_series())
data_gcc = data_gcc.sort_values('DOY')  # 确保按照DOY排序
# df_vis_30min['DOY'] = calculate_fractional_doy(df_vis_30min['datetime'])

# GCC 是 30分钟一个点，一天 48 个点。用 7 天窗口平滑：48 * 7 = 336，取奇数 337
# gcc_window = 48 * 3 + 1
# data_gcc['gcc_smooth'] = savgol_filter(
#     data_gcc['gcc_roi_sq_mean'], window_length=gcc_window, polyorder=2
# )
# Fs 经过重采样后也是 30分钟量级，同样使用 7 天的窗口平滑长期物候趋势
fs_window = 48 * 3 + 1
# 如果数据点不够多，确保窗口长度小于数组总长度
if len(df_fs_30min) > fs_window:
    df_fs_30min['Fs_smooth'] = savgol_filter(
        df_fs_30min['Fs'], window_length=fs_window, polyorder=2
    )
else:
    df_fs_30min['Fs_smooth'] = df_fs_30min['Fs']  # 数据量过少时不平滑

vis_window = 48 * 3 + 1
# if len(df_vis_30min) > vis_window:
#     df_vis_30min['mNDI705_smooth'] = savgol_filter(
#         df_vis_30min['mNDI705'], window_length=vis_window, polyorder=2
#     )
#     df_vis_30min['NDVI_smooth'] = savgol_filter(
#         df_vis_30min['NDVI'], window_length=vis_window, polyorder=2
#     )
#     df_vis_30min['PRI_smooth'] = savgol_filter(
#         df_vis_30min['PRI'], window_length=vis_window, polyorder=2
#     )
#     df_vis_30min['CIrededge_smooth'] = savgol_filter(
#         df_vis_30min['CIrededge'], window_length=vis_window, polyorder=2
#     )
#     df_vis_30min['MTCI_smooth'] = savgol_filter(
#         df_vis_30min['MTCI'], window_length=vis_window, polyorder=2
#     )
# else:
#     df_vis_30min['mNDI705_smooth'] = df_vis_30min['mNDI705']  # 数据量过少时不平滑
#     df_vis_30min['NDVI_smooth'] = df_vis_30min['NDVI']  # 数据量过少时不平滑
#     df_vis_30min['PRI_smooth'] = df_vis_30min['PRI']  # 数据量过少时不平滑
#     df_vis_30min['CIrededge_smooth'] = df_vis_30min['CIrededge']  # 数据量过少时不平滑
#     df_vis_30min['MTCI_smooth'] = df_vis_30min['MTCI']  # 数据量过少时不平滑



# ==========================================
# 5. 绘图部分
# ==========================================
# %% 绘制 Fs 和 GCC 的季节变化图
fig, ax = plt.subplots(figsize=(12, 6))

# %%  绘制 Fs（使用降采样后的数据，既保留日间变化特征，又不会卡顿）
ax.plot(df_fs_30min['DOY'],df_fs_30min['Fs'],label='Fs (30-min Resampled)',color='blue',alpha=0.3,)  # 调淡原始点，突出平滑线
ax.plot(df_fs_30min['DOY'],df_fs_30min['Fs_smooth'],label='Fs (Smoothed)',color='cyan',linewidth=2,)
# 找到df_fs_30min中Fs的最大值对应的DOY
max_fs_idx = df_fs_30min['Fs_smooth'].idxmax()
# 绘制最大值点处的垂直线，并标记出最大值和对应的DOY
max_fs_doy = df_fs_30min.loc[max_fs_idx, 'DOY']
max_fs_date = df_fs_30min.loc[max_fs_idx, 'datetime']
max_fs_value = df_fs_30min.loc[max_fs_idx, 'Fs_smooth']
ax.axvline(x=max_fs_doy, color='blue', linestyle='--', alpha=0.5)
ax.text(max_fs_doy + 1,  0.05, f'Max Fs\nDOY: {max_fs_doy:.1f}\nDate: {max_fs_date}\nValue: {max_fs_value:.3f}', \
        color='blue', fontsize=16, verticalalignment='bottom')
ax.set_ylabel('Fs (-)', color='blue')
ax.set_xlabel('Day of Year (DOY)')
ax.set_ylim([0, 1])
ax.legend(loc='upper left')

# 绘制 GCC
ax1 = ax.twinx()
ax1.plot(data_gcc['DOY'],data_gcc['gcc_roi_sq_mean'],label='GCC',color='green',alpha=0.3,marker = 'o')
# ax1.plot(data_gcc['DOY'],data_gcc['gcc_smooth'],label='GCC (Smoothed)',color='lime',linewidth=2,)
# max_gcc_idx = data_gcc['gcc_smooth'].idxmax()
# max_gcc_doy = data_gcc.loc[max_gcc_idx, 'DOY']
# max_gcc_value = data_gcc.loc[max_gcc_idx, 'gcc_smooth']
# ax1.axvline(x=max_gcc_doy, color='green', linestyle='--', alpha=0.5)
# ax1.text(max_gcc_doy + 1, 0.45, f'Max GCC\nDOY: {max_gcc_doy:.1f}\nDate: {max_gcc_idx}\nValue: {max_gcc_value:.3f}', \
#          color='green', fontsize=16, verticalalignment='bottom')
ax1.set_ylabel('GCC (-)', color='green')
ax1.set_ylim([0, 0.6])
ax1.legend(loc='upper right')

plt.title('Seasonal Variations of Fs and GCC')
fig.savefig(os.path.join(save_path, 'Fs_GCC_seasonal_variation.jpg'), dpi=300)

# # # %% 绘制 mNDI705 和 Fs 的季节变化图
# fig, ax = plt.subplots(4,1, figsize=(18, 18))
# # 绘制 Fs（使用降采样后的数据，既保留日间变化特征，又不会卡顿）
# ax[0].plot(df_fs_30min['DOY'],df_fs_30min['Fs'],label='Fs (30-min Resampled)',color='blue',alpha=0.3,)  # 调淡原始点，突出平滑线
# ax[0].plot(df_fs_30min['DOY'],df_fs_30min['Fs_smooth'],label='Fs (Smoothed)',color='cyan',linewidth=2,)
# # 找到df_fs_30min中Fs的最大值对应的DOY
# max_fs_idx = df_fs_30min['Fs_smooth'].idxmax()
# # 绘制最大值点处的垂直线，并标记出最大值和对应的DOY
# max_fs_doy = df_fs_30min.loc[max_fs_idx, 'DOY']
# max_fs_date = df_fs_30min.loc[max_fs_idx, 'datetime']
# max_fs_value = df_fs_30min.loc[max_fs_idx, 'Fs_smooth']
# ax[0].axvline(x=max_fs_doy, color='blue', linestyle='--', alpha=0.5)
# ax[0].text(max_fs_doy + 1,  0.5, f'Max Fs\nDOY: {max_fs_doy:.1f}\nDate: {max_fs_date}\nValue: {max_fs_value:.3f}', \
#         color='blue', fontsize=16, verticalalignment='bottom')
# ax[0].set_ylabel('Fs (-)', color='blue')
# ax[0].set_ylim([0, 1])
# ax[0].legend(loc='upper left')
# # 绘制 mNDI705
# ax[1].plot(df_vis_30min['DOY'],df_vis_30min['mNDI705'],label='mNDI705',color='red',alpha=0.3,)
# ax[1].plot(df_vis_30min['DOY'],df_vis_30min['mNDI705_smooth'],label='mNDI705 (Smoothed)',color='magenta',linewidth=2,)
# max_mndi_idx = df_vis_30min['mNDI705_smooth'].idxmax()
# max_mndi_doy = df_vis_30min.loc[max_mndi_idx, 'DOY']
# max_mndi_date = df_vis_30min.loc[max_mndi_idx, 'datetime']
# max_mndi_value = df_vis_30min.loc[max_mndi_idx, 'mNDI705_smooth']
# ax[1].axvline(x=max_mndi_doy, color='red', linestyle='--', alpha=0.5)
# ax[1].axvline(x=max_fs_doy, color='blue', linestyle='--', alpha=0.5)
# ax[1].text(max_mndi_doy + 1, 0.05, f'Max mNDI705\nDOY: {max_mndi_doy:.1f}\nDate: {max_mndi_date}\nValue: {max_mndi_value:.3f}', \
#          color='red', fontsize=16, verticalalignment='bottom')
# ax[1].set_ylabel('mNDI705 (-)', color='red')
# ax[1].set_ylim([0, 1])
# ax[1].legend(loc='upper right')
# # 绘制NDVI
# ax[2].plot(df_vis_30min['DOY'],df_vis_30min['NDVI'],label='NDVI',color='orange',alpha=0.3,)
# ax[2].plot(df_vis_30min['DOY'],df_vis_30min['NDVI_smooth'],label='NDVI (Smoothed)',color='darkorange',linewidth=2,)
# max_ndvi_idx = df_vis_30min['NDVI_smooth'].idxmax()
# max_ndvi_doy = df_vis_30min.loc[max_ndvi_idx, 'DOY']
# max_ndvi_date = df_vis_30min.loc[max_ndvi_idx, 'datetime']
# max_ndvi_value = df_vis_30min.loc[max_ndvi_idx, 'NDVI_smooth']
# ax[2].axvline(x=max_ndvi_doy, color='orange', linestyle='--', alpha=0.5)
# ax[2].axvline(x=max_fs_doy, color='blue', linestyle='--', alpha=0.5)
# ax[2].text(max_ndvi_doy + 1, 0.05, f'Max NDVI\nDOY: {max_ndvi_doy:.1f}\nDate: {max_ndvi_date}\nValue: {max_ndvi_value:.3f}', \
#          color='orange', fontsize=16, verticalalignment='bottom')
# ax[2].set_ylabel('NDVI (-)', color='orange')
# ax[2].set_ylim([0, 1])
# ax[2].legend(loc='upper right')
# # 绘制PRI
# ax[3].plot(df_vis_30min['DOY'],df_vis_30min['PRI'],label='PRI',color='green',alpha=0.3,)
# ax[3].plot(df_vis_30min['DOY'],df_vis_30min['PRI_smooth'],label='PRI (Smoothed)',color='lime',linewidth=2,)
# max_pri_idx = df_vis_30min['PRI_smooth'].idxmin()
# max_pri_doy = df_vis_30min.loc[max_pri_idx, 'DOY']
# min_pri_date = df_vis_30min.loc[max_pri_idx, 'datetime']
# max_pri_value = df_vis_30min.loc[max_pri_idx, 'PRI_smooth']
# ax[3].axvline(x=max_pri_doy, color='green', linestyle='--', alpha=0.5)
# ax[3].axvline(x=max_fs_doy, color='blue', linestyle='--', alpha=0.5)
# ax[3].text(max_pri_doy + 1, -0.1, f'Min PRI\nDOY: {max_pri_doy:.1f}\nDate: {min_pri_date}\nValue: {max_pri_value:.3f}', \
#          color='green', fontsize=16, verticalalignment='bottom')
# ax[3].set_ylabel('PRI (-)', color='green')
# ax[3].set_ylim([-0.4, 0.2])
# ax[3].legend(loc='upper right')
# ax[0].set_title('Seasonal Variations of Fs and vegetation indices')
# ax[3].set_xlabel('Day of Year (DOY)')
# fig.savefig(os.path.join(save_path, 'Fs_VIs_seasonal_variation.jpg'), dpi=300)


# # %% Fs during the night (Fo)
# fig, ax = plt.subplots(figsize=(12, 6), sharex=True)
# idx_night = df_fs_30min['datetime'].dt.hour.isin([0, 1, 2])
# idx_day = df_fs_30min['datetime'].dt.hour.isin([10, 11, 12, 13, 14, 15])
# fig, ax = plt.subplots(2,1, figsize=(12, 10))
# ax[0].plot(df_fs_30min.loc[idx_night, 'DOY'], df_fs_30min.loc[idx_night, 'Fs'], marker='o', label='Fo (30-min Resampled)', color='purple', alpha=0.3)
# # ax[0].plot(df_fs_30min.loc[idx_night, 'DOY'], df_fs_30min.loc[idx_night, 'Fs_smooth'], label='Fo (Smoothed)', color='magenta', linewidth=2)
# max_fo_idx = df_fs_30min.loc[idx_night, 'Fs_smooth'].idxmax()
# max_fo_doy = df_fs_30min.loc[max_fo_idx, 'DOY']
# max_fo_date = df_fs_30min.loc[max_fo_idx, 'datetime']
# max_fo_value = df_fs_30min.loc[max_fo_idx, 'Fs_smooth']
# ax[0].axvline(x=max_fo_doy, color='purple', linestyle='--', alpha=0.5)
# ax[0].text(max_fo_doy + 1, 0.05, f'Max Fo\nDOY: {max_fo_doy:.1f}\nDate: {max_fo_date}\nValue: {max_fo_value:.3f}', \
#         color='purple', fontsize=16, verticalalignment='bottom')
# ax[0].set_ylabel('Fo (-)', color='purple')
# ax[0].set_xlabel('Day of Year (DOY)')
# ax[0].set_ylim([0, 0.6])

# ax[1].plot(df_fs_30min.loc[idx_day, 'DOY'], df_fs_30min.loc[idx_day, 'Fs'], marker = 'o', label='Fs (30-min Resampled)', color='blue', alpha=0.3)
# # ax[1].plot(df_fs_30min.loc[idx_day, 'DOY'], df_fs_30min.loc[idx_day, 'Fs_smooth'], label='Fs (Smoothed)', color='darkblue', linewidth=2)
# max_fs_day_idx = df_fs_30min.loc[idx_day, 'Fs_smooth'].idxmax()
# max_fs_day_doy = df_fs_30min.loc[max_fs_day_idx, 'DOY']
# max_fs_day_date = df_fs_30min.loc[max_fs_day_idx, 'datetime']
# max_fs_day_value = df_fs_30min.loc[max_fs_day_idx, 'Fs_smooth']
# ax[1].axvline(x=max_fs_day_doy, color='darkblue', linestyle='--', alpha=0.5)
# ax[1].text(max_fs_day_doy + 1, 0.05, f'Max Fs (Day)\nDOY: {max_fs_day_doy:.1f}\nDate: {max_fs_day_date}\nValue: {max_fs_day_value:.3f}', \
#         color='darkblue', fontsize=16, verticalalignment='bottom')
# ax[1].set_ylabel('Fs (-)', color='blue')
# ax[1].set_xlabel('Day of Year (DOY)')
# ax[1].set_ylim([0, 0.6])
# fig.suptitle('Seasonal Variations of Nighttime (Fo) and Daytime (Fs)', fontsize=18)
# fig.savefig(os.path.join(save_path, 'Fo_Fs_seasonal_variation.jpg'), dpi=300)

# # %% 计算每天Fo和Fs的均值表示日平均的夜间和日间Fs

# fig, ax = plt.subplots(figsize=(12, 6), sharex=True)

# df_fs_30min['is_night'] = df_fs_30min['datetime'].dt.hour.isin([0, 1, 2])
# df_fs_30min['is_day'] = df_fs_30min['datetime'].dt.hour.isin([10, 11, 12, 13, 14, 15])
# daily_fo = df_fs_30min[df_fs_30min['is_night']].groupby(df_fs_30min['datetime'].dt.date)['Fs'].mean()
# daily_fs = df_fs_30min[df_fs_30min['is_day']].groupby(df_fs_30min['datetime'].dt.date)['Fs'].mean()
# # daily_fo = df_fs_30min[df_fs_30min['is_night']].groupby(df_fs_30min['datetime'].dt.date)['Fs_smooth'].mean()
# # daily_fs = df_fs_30min[df_fs_30min['is_day']].groupby(df_fs_30min['datetime'].dt.date)['Fs_smooth'].mean()
# daily = pd.DataFrame({'date': daily_fo.index, 'Fo': daily_fo.values, 'Fs': daily_fs.values})
# daily['fos_ratio'] =  daily['Fs'] / daily['Fo']

# fig, ax = plt.subplots(3,1, figsize=(12, 12), sharex=True)
# ax[0].plot(daily['date'], daily['Fo'], label='Fo', marker='o', color='purple')
# ax[0].set_ylabel('Fo (-)', color='purple')
# ax[0].legend(loc='upper left')
# ax[0].set_ylim([0, 0.6])
# ax[1].plot(daily['date'], daily['Fs'], label='Fs', marker='s', color='blue')
# ax[1].set_ylabel('Fs (-)', color='blue')
# ax[1].legend(loc='upper left')
# ax[1].set_ylim([0, 0.6])
# ax[2].plot(daily['date'], daily['fos_ratio'], label='Fs/Fo', marker='^', color='magenta')
# ax[2].set_ylabel('Fs/Fo (-)', color='magenta')
# ax[2].legend(loc='upper left')
# fig.suptitle('Seasonal Variations of Fo, Fs, and Fs/Fo', fontsize=18)
# fig.savefig(os.path.join(save_path, 'Daily_Fo_Fs_ratio.jpg'), dpi=300)

# plt.show()