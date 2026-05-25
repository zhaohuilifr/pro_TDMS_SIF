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


path_lif = r'E:\Datahub\Barbeau\Data_LIF\A_LIF_PAR_Time_Cor\µLIDAR_situ_data_Barbeau\PROCESSED\2025\L2\Yearly\2025_LIF.csv'
path_gcc = r'E:\Datahub\Barbeau\Data_Camera\GCC\2025\all_gcc_2025_filtered.csv'

data_lif = pd.read_csv(path_lif)
data_gcc = pd.read_csv(path_gcc)
data_lif['datetime'] = pd.to_datetime(data_lif['time_01011904'], unit='s', origin=pd.Timestamp('1904-01-01'))
data_gcc['datetime'] = pd.to_datetime(data_gcc['datetime'], format='%Y-%m-%d %H:%M:%S')

data_lif.set_index('datetime', inplace=True)
data_gcc.set_index('datetime', inplace=True)

# resample to 30 mins (pandas>=2.2 prefers 'min' over 'T')
df_fs_30min = data_lif['Fs'].resample('30min').mean().dropna().to_frame()
df_fs_30min.reset_index(inplace=True)

def calculate_fractional_doy(series):
    """高效计算带小数部分的年内天数(DOY)"""
    doy_base = series.dt.dayofyear
    fraction = (
        series.dt.hour * 3600
        + series.dt.minute * 60
        + series.dt.second + series.dt.microsecond / 1e6
    ) / 86400.0
    return doy_base + fraction


df_fs_30min['DOY'] = calculate_fractional_doy(df_fs_30min['datetime'])
data_gcc['DOY'] = calculate_fractional_doy(data_gcc.index.to_series())

# GCC 是 30分钟一个点，一天 48 个点。用 7 天窗口平滑：48 * 7 = 336，取奇数 337
gcc_window = 48 * 3 + 1
data_gcc['gcc_smooth'] = savgol_filter(
    data_gcc['gcc_roi_mean'], window_length=gcc_window, polyorder=2
)

# Fs 经过重采样后也是 30分钟量级，同样使用 7 天的窗口平滑长期物候趋势
fs_window = 337
# 如果数据点不够多，确保窗口长度小于数组总长度
if len(df_fs_30min) > fs_window:
    df_fs_30min['Fs_smooth'] = savgol_filter(
        df_fs_30min['Fs'], window_length=fs_window, polyorder=2
    )
else:
    df_fs_30min['Fs_smooth'] = df_fs_30min['Fs']  # 数据量过少时不平滑

# ==========================================
# 5. 绘图部分
# ==========================================
fig, ax = plt.subplots(figsize=(12, 6))

# 绘制 Fs（使用降采样后的数据，既保留日间变化特征，又不会卡顿）
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
ax1.plot(data_gcc['DOY'],data_gcc['gcc_roi_mean'],label='GCC',color='green',alpha=0.3,)
ax1.plot(data_gcc['DOY'],data_gcc['gcc_smooth'],label='GCC (Smoothed)',color='lime',linewidth=2,)
max_gcc_idx = data_gcc['gcc_smooth'].idxmax()
max_gcc_doy = data_gcc.loc[max_gcc_idx, 'DOY']
max_gcc_value = data_gcc.loc[max_gcc_idx, 'gcc_smooth']
ax1.axvline(x=max_gcc_doy, color='green', linestyle='--', alpha=0.5)
ax1.text(max_gcc_doy + 1, 0.45, f'Max GCC\nDOY: {max_gcc_doy:.1f}\nDate: {max_gcc_idx}\nValue: {max_gcc_value:.3f}', \
         color='green', fontsize=16, verticalalignment='bottom')
ax1.set_ylabel('GCC (-)', color='green')
ax1.set_ylim([0, 0.6])
ax1.legend(loc='upper right')

plt.title('Seasonal Variations of Fs and GCC')
plt.show()