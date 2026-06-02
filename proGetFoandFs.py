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

def getFoandFs(df_fs_30min):
    # calculate Fo and Fs based on the smoothed Fs
    # # method 01
    # par_threshold = 1 # µmol/m2/s
    # sunrise_df = df[df['PAR'] >= par_threshold]
    # if not sunrise_df.empty:
    #     sunrise_time = sunrise_df.index[0]
    #     # Fo is the average of Fs during the 1 minute before sunrise
    #     idx = (df['datetime'] >= (sunrise_df.loc[sunrise_time, 'datetime'] - pd.Timedelta(minutes=1))) \
    #         & (df['datetime'] < sunrise_df.loc[sunrise_time, 'datetime'])
    #     f0_window = df[idx]
    #     f0_window_dt_index = f0_window.set_index('datetime')
    #     f0_value1 = f0_window['Fs'].mean()
    #     if not f0_window_dt_index.empty:
    #         f0_value1_idx = f0_window_dt_index.index[0]
    #     else:            
    #         f0_value1_idx = None
    # else:
    #     f0_value1 = np.nan
    #     f0_value1_idx = None
    # method 02
    df1 = df_fs_30min.copy()
    df1 = df1.sort_values('datetime').set_index('datetime')
    # df1['Fs_rolling'] = df1['Fs'].rolling(window='10s').mean()
    df1['Fs_slope'] = df1['Fs_smooth'].diff()
    morning_window = df1.between_time('01:00', '05:00')
    # sunrise_idx = morning_window[morning_window['Fs_slope'] > morning_window['Fs_slope'].std() * 1].index
    sunrise_idx = morning_window[morning_window['Fs_slope'] > 0.001].index
    if len(sunrise_idx) > 0:
        f0_value2 = df1.loc[sunrise_idx[0], 'Fs_smooth']
        f0_value2_idx = sunrise_idx[0]
    else:        
        f0_value2 = np.nan
        f0_value2_idx = None
    # method 03
    dawn_stable_zone = df1.between_time('02:30', '03:00')
    if not dawn_stable_zone.empty:
        f0_value3 = dawn_stable_zone['Fs'].median()
        f0_value3_idx = dawn_stable_zone.index[0]
    else:
        f0_value3 = np.nan
        f0_value3_idx = None
    # Fs
    idx_daytime = df_fs_30min['datetime'].dt.hour.isin(range(11, 13))
    if idx_daytime.sum() > 0:
        fs_window = df_fs_30min[idx_daytime]
        fs_window_dt_index = fs_window.set_index('datetime')
        fs_value = fs_window['Fs_smooth'].mean()
        fs_value_idx = fs_window_dt_index.index[0] + pd.Timedelta(hours=1) # set fs_value_idx as the 12:00 time point of that day
        # fs_value = df.loc[idx_daytime, 'Fs'].mean()
        # # set fs_value_idx as the 12:00 time point of that day 
        # fs_value_idx = df.loc[idx_daytime, 'datetime'].dt.floor('D') + pd.Timedelta(hours=12)
    else:
        fs_value = np.nan
        fs_value_idx = None
    fos = [f0_value2, f0_value3, fs_value]
    fos_idx = [f0_value2_idx, f0_value3_idx, fs_value_idx]
    return fos, fos_idx

# rootpath = r'E:\Datahub\Barbeau\Data_LIF\A_LIF_PAR_Time_Cor\µLIDAR_situ_data_Barbeau\PROCESSED'
# yearstrs = ['2022','2023','2024','2025','2026']# '2025'
# for yearstr in yearstrs:
#     savepath = os.path.join(rootpath, yearstr, 'L2', 'Yearly')
#     savepath_figs = os.path.join(rootpath, yearstr, 'L2', 'Daily_figs_Fo_Fs')
#     os.makedirs(savepath_figs, exist_ok=True)
#     path_lif = os.path.join(rootpath, yearstr, 'L2', 'Yearly', f'{yearstr}_LIF.csv')

#     data_lif = pd.read_csv(path_lif)
#     data_lif['datetime'] = pd.to_datetime(data_lif['time_01011904'], unit='s', origin=pd.Timestamp('1904-01-01'))
#     data_lif_copy = data_lif.copy()
#     data_lif.set_index('datetime', inplace=True)
#     # resample to 30 mins (pandas>=2.2 prefers 'min' over 'T')
#     df_fs_30min = data_lif['Fs'].resample('30min').mean().dropna().to_frame()
#     df_fs_30min.reset_index(inplace=True)
#     df_fs_30min['DOY'] = calculate_fractional_doy(df_fs_30min['datetime'])
#     # Fs 经过重采样后也是 30分钟量级，同样使用 7 天的窗口平滑长期物候趋势
#     fs_window = 7 # 7 * 30 mins window for smoothing
#     # 如果数据点不够多，确保窗口长度小于数组总长度
#     if len(df_fs_30min) > fs_window:
#         df_fs_30min['Fs_smooth'] = savgol_filter(
#             df_fs_30min['Fs'], window_length=fs_window, polyorder=2
#         )
#     else:
#         df_fs_30min['Fs_smooth'] = df_fs_30min['Fs']  # 数据量过少时不平滑

#     df_fs_30min.to_csv(os.path.join(savepath, f'{yearstr}_Fs_30min_smooth.csv'), index=False)


#     # %%  calculate Fo and Fs for each day, and plot them colored by DOY
#     doy_start = 80 #80 # df_fs_30min['DOY'].min()
#     doy_end = 310 # df_fs_30min['DOY'].max()

#     df = []
#     for doy in range(int(doy_start), int(doy_end) + 1):
#         df_day = df_fs_30min[(df_fs_30min['DOY'] >= doy) & (df_fs_30min['DOY'] < doy + 1)]
#         df_day_raw = data_lif_copy[(data_lif_copy['DOY'] >= doy) & (data_lif_copy['DOY'] < doy + 1)]
#         df_day_raw_dt = df_day_raw.set_index('datetime')
#         fo_values, fo_indices = getFoandFs(df_day)
#         df.append({
#             'DOY': doy,
#             'Fo2': fo_values[0],
#             'Fo3': fo_values[1],
#             'Fs': fo_values[2],
#         })

#         if not df_day_raw.empty:
#             plt.figure(figsize=(12, 5))
#             plt.plot(df_day_raw['datetime'], df_day_raw['Fs'], alpha=0.3, color='blue', label='Original Fs')
#             plt.plot(df_day['datetime'], df_day['Fs'], marker='.', color = 'blue', alpha=0.5, label='Fs (30min resampled)')
#             plt.plot(df_day['datetime'], df_day['Fs_smooth'], label='Smoothed Fs (30min smoothed)', marker='.', color='red')
#             plt.scatter(fo_indices[0], fo_values[0], color='purple', label='Fo (Estimated)', zorder=5, s = 24)
#             # plt.scatter(fo_indices[1], fo_values[1], color='cyan', label='Fo (m3)', zorder=5)
#             plt.scatter(fo_indices[2], fo_values[2], color='green', label='Fs (Estimated)', zorder=5, s = 24)
#             plt.ylabel('Fs')
#             plt.xlabel('Time')
#             plt.legend(loc = 'upper right')
#             ax = plt.twinx()
#             ax.plot(df_day_raw['datetime'], df_day_raw['PAR'], alpha=0.3, color='red', ls = '--',label='Original PAR')
#             ax.set_ylabel('PAR (µmol/m2/s)')
#             ax.legend(loc = 'upper left')
            
#             plt.title(f'Fs on DOY {doy} ({df_day_raw["datetime"].dt.date.iloc[0]})')
#             plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
#             # show all legends for both y-axes
#             plt.savefig(os.path.join(savepath_figs, f'{yearstr}_Fs_DOY_{doy}.jpg'), dpi=300)
#             # plt.show()
#     df_fo_fs = pd.DataFrame(df)
#     df_fo_fs.to_csv(os.path.join(savepath, f'{yearstr}_Fo_Fs_values.csv'), index=False)

# %% plot Fo and Fs for each day, colored by DOY

rootpath = r'E:\Datahub\Barbeau\Data_LIF\A_LIF_PAR_Time_Cor\µLIDAR_situ_data_Barbeau\PROCESSED'
yearstrs = ['2022','2023','2024','2025','2026']# '2025'
for yearstr in yearstrs:
    savepath = os.path.join(rootpath, yearstr, 'L2', 'Yearly')
    # savepath_figs = os.path.join(rootpath, yearstr, 'L2', 'Daily_figs_Fo_Fs')
    # os.makedirs(savepath_figs, exist_ok=True)
    # path_lif = os.path.join(rootpath, yearstr, 'L2', 'Yearly', f'{yearstr}_LIF.csv')

    data = pd.read_csv(os.path.join(savepath, f'{yearstr}_Fo_Fs_values.csv'))
    fig, ax = plt.subplots(2,1, figsize=(12, 7))
    ax[0].plot(data['DOY'], data['Fo2'], color='red', marker = '.',label='Fo', zorder=5)
    # plt.scatter(data['DOY'], data['Fo3'], color='purple', label='Fo (m3)', zorder=5)
    ax[0].plot(data['DOY'], data['Fs'], color='green', marker = '.',label='Fs', zorder=5)
    ax[0].set_ylabel('Fo and Fs (volts)')
    # ax[0].set_xlabel('DOY')
    ax[0].set_xlim([75, 315])
    ax[0].set_ylim([0, 0.5])
    ax[0].legend(loc = 'upper right')
    ax[0].set_title(f'Fo and Fs in {yearstr}', fontweight='bold', fontsize=16)
    ax[1].plot(data['DOY'], data['Fs']/ data['Fo2'], color='blue', marker = '.',label='Fo/Fs', zorder=5)
    ax[1].set_ylabel('Fs/Fo (-)')
    ax[1].set_xlabel('DOY')
    ax[1].set_xlim([75, 315])
    ax[1].set_ylim([0, 2])
    ax[1].legend(loc = 'upper right')
    # plt.show()
    plt.savefig(os.path.join(savepath, f'{yearstr}_Fo_Fs_DOY.jpg'), dpi=300)