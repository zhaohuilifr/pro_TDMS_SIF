# -*- coding: utf-8 -*-

###
# Problems:
# 1) the filename date is not trustable
# 2) the time recorded inside the filename is not trustable (?)
# To be Solved:
# 1. find the date when it started
# 2. correct the time of each raw file
###

import os
import shutil
import pandas as pd
import numpy as np
import glob as glob
from io import StringIO
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from joblib import Parallel, delayed
import tkinter as tk
from tkinter import filedialog
import matplotlib.dates as mdates

# 设置全局字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14

def read_mfl(mfl_file):
    with open(mfl_file, 'r', encoding = 'utf-8') as f:
        lines = f.readlines()
        data_str = ''.join(lines)
        # 使用 StringIO 将字符串转换为文件流
        data_stream = StringIO(data_str)
        # 跳过前 4 行（文件信息），读取数据
        df = pd.read_csv(data_stream, sep='\t', skiprows=4)
        idx = df['time_01011904'] < 3e9
        # 将时间戳转换为 UTC 时间
        # 在 LabVIEW 系统中，时间 (LabVIEW Timestamp) 是从 UTC 1904 年 1 月 1 日 00:00:00 开始计算的秒数。
        df['UTC_time'] = pd.to_datetime(df['time_01011904'], unit='s', origin=pd.Timestamp('1904-01-01'))
        df['DOY'] = df['UTC_time'].dt.dayofyear + df['UTC_time'].dt.hour / 24 + df['UTC_time'].dt.minute / 1440 + df['UTC_time'].dt.second / 86400
        # 02
        # epoch_lv = datetime(1904, 1, 1)
        # df['UTC_time_2'] = df['time_01011904'].apply(lambda x: epoch_lv + timedelta(seconds=x))
        df.loc[idx, 'UTC_time'] = np.nan
        # df.loc[idx, 'UTC_time_2'] = np.nan
        # reset index
        df = df.reset_index(drop=True)
    return df

def utc_to_doy(utc_time):
    # 将 UTC 时间转换为 DOY（年内日）
    doy = utc_time.dt.dayofyear + utc_time.dt.hour / 24 + utc_time.dt.minute / 1440 + utc_time.dt.second / 86400
    return doy

def find_cutpoint(df):
    split_index = df[df['time_01011904'].diff()<0].index
    if len(split_index) > 0:
        cut_point = split_index[0]
    else:
        cut_point = None
    return cut_point

unitpar = '\n'+'(μmol photon m$^{-2}$ s$^{-1}$)'

if __name__ == "__main__":
    # %% 设置路径
    path_flux = r'E:\Datahub\Barbeau\Data_flux\Daniel'
    root = r'E:\Datahub\Barbeau\Data_LIF\A_LIF_PAR_Time_Cor\µLIDAR_situ_data_Barbeau'
    lif_path = os.path.join(root, 'RAW')
    meteo_path = r'E:\Datahub\Barbeau\Data_LIF\A_LIF_PAR_Time_Cor\PAR_data\Radiations_20s'
    yearstr = '2026' # '2025' # '2022'
    for subfolders in ['L0', 'L1', 'L2']:
        if not os.path.exists(os.path.join(root, 'PROCESSED', yearstr,subfolders)):
            os.makedirs(os.path.join(root, 'PROCESSED', yearstr,subfolders))
    savepath = os.path.join(root, 'PROCESSED', yearstr)
    # %% process .mfl files to csv files and save by date
    # mfl_files = glob.glob(os.path.join(lif_path, yearstr,'*.mfl'))
    # dfs = []
    # for i, mfl_file in enumerate(mfl_files):
    #     df = read_mfl(mfl_file)
    #     dfs.append(df)
    #     # df.to_csv(os.path.join(savepath, os.path.basename(mfl_file).replace('.mfl', '.csv')), index=False)
    #     # print(f'Processing file {i+1}/{len(mfl_files)}: {os.path.basename(mfl_file)}')
    # df_all = pd.concat(dfs, ignore_index=True)
    # # find the date range and save data by date
    # date_start = (df_all.loc[0, 'UTC_time']).date()
    # date_end = (df_all.loc[len(df_all)-1, 'UTC_time']).date()
    # for date in pd.date_range(date_start, date_end):
    #     df_date = df_all[(df_all['UTC_time'] >= date) & (df_all['UTC_time'] < date + pd.Timedelta(days=1))]
    #     if len(df_date) > 0:
    #         df_date.to_csv(os.path.join(savepath, 'L0', f'{date.strftime("%Y%m%d")}.csv'), index=False)

    # %% plot PAR and Fs time series, and compare with flux data PAR data
    # savepath_figs = os.path.join(savepath,'L1','figs')
    # if not os.path.exists(savepath_figs):
    #     os.makedirs(savepath_figs)
    
    # # flux data
    # df_flux = glob.glob(os.path.join(path_flux, 'Barbeau_' + yearstr + '*.xls'))[0]
    # df_flux = pd.read_excel(df_flux, sheet_name='data')
    # df_flux.loc[df_flux['hh'] == 0, 'jj'] -= 1
    # df_flux.loc[df_flux['hh'] == 0, 'hh'] = 24
    # df_flux['hh'] -= 0.5
    # df_flux['DOY'] = df_flux['jj'] + df_flux['hh'] / 24


    # csvfiles = glob.glob(os.path.join(savepath, 'L0', '*.csv'))
    # for i, csvfile in enumerate(csvfiles):
    #     df = pd.read_csv(csvfile)
    #     datestr = os.path.basename(csvfile).split('.')[0]
    #     doy = datetime.strptime(datestr, '%Y%m%d').timetuple().tm_yday
    #     idx_flux = (df_flux['DOY'] >= doy) & (df_flux['DOY'] < doy + 1)
    #     if os.path.getsize(csvfile) < 1000*1024: # 1000 kb
    #         continue
    #     fig, ax = plt.subplots(2,1,figsize=(10,6), sharex=True)

    #     cutpoint = find_cutpoint(df)
    #     if cutpoint is not None:
    #         df1 = df.iloc[:cutpoint, :]
    #         df2 = df.iloc[cutpoint:, :]
    #         ax[0].plot(df1['DOY'], df1['PAR'], 'b-', marker='.',label='PAR (part1)')
    #         ax[0].plot(df2['DOY'], df2['PAR'], 'r-', marker='.',label='PAR (part2)')
    #         ax[0].plot(df_flux.loc[idx_flux, 'DOY'], df_flux.loc[idx_flux, 'PAR (µmol/m2/s)'], 'k-', marker='.', label='PAR (flux)')
    #         ax[1].plot(df1['DOY'], df1['Fs'], 'b-', marker='.',label='Fs (part1)')
    #         ax[1].plot(df2['DOY'], df2['Fs'], 'r-', marker='.',label='Fs (part2)')
    #     else:
    #         ax[0].plot(df['DOY'], df['PAR'], 'b-', marker='.',label='PAR (LIF)')
    #         ax[0].plot(df_flux.loc[idx_flux, 'DOY'], df_flux.loc[idx_flux, 'PAR (µmol/m2/s)'], 'k-', marker='.', label='PAR (flux)')
    #         ax[1].plot(df['DOY'], df['Fs'], 'b-', marker='.',label='Fs')
        
    #     ax[0].set_xlim([df_flux.loc[idx_flux, 'DOY'].min(), df_flux.loc[idx_flux, 'DOY'].max()])
    #     ax[1].set_xlim([df_flux.loc[idx_flux, 'DOY'].min(), df_flux.loc[idx_flux, 'DOY'].max()])
    #     # ax[0].set_xlabel('DOY')
    #     ax[0].set_ylabel('PAR (µmol m$^{-2}$ s$^{-1}$)')
    #     ax[0].legend()
    #     ax[1].set_xlabel('DOY')
    #     ax[1].set_ylabel('Fs (-)')
    #     ax[1].legend()

    #     fig.savefig(os.path.join(savepath_figs, f'{datestr}_PAR_Fs.png'), dpi=300)

    # %% fix data
    # criteria:
    # 1. Normally, the size of each csv file should be around 10000 kb for a complete day measurement. 
    # If the size is larger than 10000 kb, it may be a replicated file. For example, if the size is around 20000 kb, 
    # it may be a replicated file with two days of data. 
    # 2. Copy the L0 csv files to L1 folder and check the size of each file. Remove the replicated data in 
    # each file that is larger than 10000 kb.
    # 3. 

    # %% ==========================================================================================
    # ================================== Correction for data 2022 =================================
    # =============================================================================================
    # datapath = r'E:\Datahub\Barbeau\Data_LIF\A_LIF_PAR_Time_Cor\µLIDAR_situ_data_Barbeau\PROCESSED\2022\L1\Datachecked'
    # savepath = r'E:\Datahub\Barbeau\Data_LIF\A_LIF_PAR_Time_Cor\µLIDAR_situ_data_Barbeau\PROCESSED\2022\L1\DataCorrected'
    # # 20220617
    # # remove the first part of the data
    # # 20220616
    # df = pd.read_excel(os.path.join(datapath, '20220616.xlsx'))
    # sample_df = df.iloc[20000:50000, :]
    # slope = (sample_df['time_01011904'].iloc[-1] - sample_df['time_01011904'].iloc[0]) / (50000 - 20000)
    # print(f'(ΔDOY/point) slope: {slope}')
    # end_doy_idx = (df['DOY'] - 168) < 0
    # end_doy = df.loc[np.argmax(df.loc[end_doy_idx, 'DOY']), 'time_01011904']
    # # time correction
    # df['time_01011904'] = end_doy - np.arange(len(df),0,-1) * slope
    # df['UTC_time'] = pd.to_datetime(df['time_01011904'], unit='s', origin=pd.Timestamp('1904-01-01'))
    # df['DOY'] = df['UTC_time'].dt.dayofyear + df['UTC_time'].dt.hour / 24 + df['UTC_time'].dt.minute / 1440 + df['UTC_time'].dt.second / 86400
    # df = df[df['DOY'] >= 167]
    # df = df.reset_index(drop=True)
    # df.to_csv(os.path.join(savepath, '20220616.csv'), index=False)

    # # 20220707
    # df = pd.read_excel(os.path.join(datapath, '20220707.xlsx'))
    # st, ed = 1000, 30000
    # sample_df = df.iloc[st:ed, :]
    # slope = (sample_df['time_01011904'].iloc[-1] - sample_df['time_01011904'].iloc[0]) / (ed - st)
    # print(f'(ΔDOY/point) slope: {slope}')
    # start_doy = df.loc[0, 'time_01011904']
    # # time correction
    # df['time_01011904'] = start_doy + np.arange(len(df)) * slope
    # df['UTC_time'] = pd.to_datetime(df['time_01011904'], unit='s', origin=pd.Timestamp('1904-01-01'))
    # df['DOY'] = df['UTC_time'].dt.dayofyear + df['UTC_time'].dt.hour / 24 + df['UTC_time'].dt.minute / 1440 + df['UTC_time'].dt.second / 86400
    # df_nextday = df[df['DOY'] >= 189].copy()
    # df = df[df['DOY'] < 189]
    # df.loc[df['PAR']<-500, 'PAR'] = np.nan
    # df = df.reset_index(drop=True)
    # df.to_csv(os.path.join(savepath, '20220707.csv'), index=False)

    # # new method for 20220707
    # df = pd.read_excel(os.path.join(datapath, '20220707.xlsx'))
    # st, ed = 40000, 48000
    # sample_df = df.iloc[st:ed, :]
    # slope = (sample_df['time_01011904'].iloc[-1] - sample_df['time_01011904'].iloc[0]) / (ed - st)
    # print(f'(ΔDOY/point) slope: {slope}')
    # start_doy = df.loc[40000, 'time_01011904'] # assume the time at 40000 is correct, and use it as the start time for correction
    # # time correction
    # df['time_01011904'] = start_doy + (np.arange(len(df))-40000) * slope
    # df['UTC_time'] = pd.to_datetime(df['time_01011904'], unit='s', origin=pd.Timestamp('1904-01-01'))
    # df['DOY'] = df['UTC_time'].dt.dayofyear + df['UTC_time'].dt.hour / 24 + df['UTC_time'].dt.minute / 1440 + df['UTC_time'].dt.second / 86400
    # df_nextday = df[df['DOY'] >= 189].copy()
    # df = df[df['DOY'] < 189]
    # df.loc[df['PAR']<-500, 'PAR'] = np.nan
    # df = df.reset_index(drop=True)
    # df.to_csv(os.path.join(savepath, '20220707.csv'), index=False)

    # # add a offset to the df_nextday
    # df_nextday['time_01011904'] = df_nextday['time_01011904'] + 30*60 # 30 minutes
    # df_nextday['UTC_time'] = pd.to_datetime(df_nextday['time_01011904'], unit='s', origin=pd.Timestamp('1904-01-01'))
    # df_nextday['DOY'] = df_nextday['UTC_time'].dt.dayofyear + df_nextday['UTC_time'].dt.hour / 24 + df_nextday['UTC_time'].dt.minute / 1440 + df_nextday['UTC_time'].dt.second / 86400
    
    # # 20220708
    # df = pd.read_csv(os.path.join(datapath, '20220708.csv'))
    # df = pd.concat([df_nextday, df], ignore_index=True)
    # start_doy = df.loc[0, 'time_01011904']
    # df['time_01011904'] = start_doy + np.arange(len(df)) * slope
    # df['UTC_time'] = pd.to_datetime(df['time_01011904'], unit='s', origin=pd.Timestamp('1904-01-01'))
    # df['DOY'] = df['UTC_time'].dt.dayofyear + df['UTC_time'].dt.hour / 24 + df['UTC_time'].dt.minute / 1440 + df['UTC_time'].dt.second / 86400
    # df_nextday = df[df['DOY'] >= 190].copy()
    # df = df[df['DOY'] < 190]
    # df.loc[df['PAR']<-500, 'PAR'] = np.nan
    # df = df.reset_index(drop=True)
    # df.to_csv(os.path.join(savepath, '20220708.csv'), index=False)

    # # 20220709
    # df = pd.read_csv(os.path.join(datapath, '20220709.csv'))
    # df = pd.concat([df_nextday, df], ignore_index=True)
    # start_doy = df.loc[0, 'time_01011904']
    # df['time_01011904'] = start_doy + np.arange(len(df)) * slope
    # df['UTC_time'] = pd.to_datetime(df['time_01011904'], unit='s', origin=pd.Timestamp('1904-01-01'))
    # df['DOY'] = df['UTC_time'].dt.dayofyear + df['UTC_time'].dt.hour / 24 + df['UTC_time'].dt.minute / 1440 + df['UTC_time'].dt.second / 86400
    # df_nextday = df[df['DOY'] >= 191].copy()
    # df = df[df['DOY'] < 191]
    # df.loc[df['PAR']<-500, 'PAR'] = np.nan
    # df = df.reset_index(drop=True)
    # df.to_csv(os.path.join(savepath, '20220709.csv'), index=False)

    # # 20220710
    # df = pd.read_csv(os.path.join(datapath, '20220710.csv'))
    # df = pd.concat([df_nextday, df], ignore_index=True)
    # start_doy = df.loc[0, 'time_01011904']
    # df['time_01011904'] = start_doy + np.arange(len(df)) * slope
    # df['UTC_time'] = pd.to_datetime(df['time_01011904'], unit='s', origin=pd.Timestamp('1904-01-01'))
    # df['DOY'] = df['UTC_time'].dt.dayofyear + df['UTC_time'].dt.hour / 24 + df['UTC_time'].dt.minute / 1440 + df['UTC_time'].dt.second / 86400
    # df_nextday = df[df['DOY'] >= 192].copy()
    # df = df[df['DOY'] < 192]
    # df.loc[df['PAR']<-500, 'PAR'] = np.nan
    # df = df.reset_index(drop=True)
    # df.to_csv(os.path.join(savepath, '20220710.csv'), index=False)

    # # 20220711
    # df = pd.read_csv(os.path.join(datapath, '20220711.csv'))
    # df = pd.concat([df_nextday, df], ignore_index=True)
    # start_doy = df.loc[0, 'time_01011904']
    # df['time_01011904'] = start_doy + np.arange(len(df)) * slope
    # df['UTC_time'] = pd.to_datetime(df['time_01011904'], unit='s', origin=pd.Timestamp('1904-01-01'))
    # df['DOY'] = df['UTC_time'].dt.dayofyear + df['UTC_time'].dt.hour / 24 + df['UTC_time'].dt.minute / 1440 + df['UTC_time'].dt.second / 86400
    # df_nextday = df[df['DOY'] >= 193].copy()
    # df = df[df['DOY'] < 193]
    # df.loc[df['PAR']<-500, 'PAR'] = np.nan
    # df = df.reset_index(drop=True)
    # df.to_csv(os.path.join(savepath, '20220711.csv'), index=False)

    # # 20220712
    # df = pd.read_csv(os.path.join(datapath, '20220712.csv'))
    # df = pd.concat([df_nextday, df], ignore_index=True)
    # start_doy = df.loc[0, 'time_01011904']
    # df['time_01011904'] = start_doy + np.arange(len(df)) * slope
    # df['UTC_time'] = pd.to_datetime(df['time_01011904'], unit='s', origin=pd.Timestamp('1904-01-01'))
    # df['DOY'] = df['UTC_time'].dt.dayofyear + df['UTC_time'].dt.hour / 24 + df['UTC_time'].dt.minute / 1440 + df['UTC_time'].dt.second / 86400
    # df_nextday = df[df['DOY'] >= 194].copy()
    # df = df[df['DOY'] < 194]
    # df.loc[df['PAR']<-500, 'PAR'] = np.nan
    # df = df.reset_index(drop=True)
    # df.to_csv(os.path.join(savepath, '20220712.csv'), index=False)

    # # 20220713
    # df = pd.read_csv(os.path.join(datapath, '20220713.csv'))
    # end_doy = df.loc[len(df)-1, 'time_01011904']
    # sample_df = df.iloc[10000:40000, :]
    # slope = (sample_df['time_01011904'].iloc[-1] - sample_df['time_01011904'].iloc[0]) / (40000 - 10000)
    # print(f'(ΔDOY/point) slope: {slope}')
    # df['time_01011904'] = end_doy - np.arange(len(df),0,-1) * slope
    # df['UTC_time'] = pd.to_datetime(df['time_01011904'], unit='s', origin=pd.Timestamp('1904-01-01'))
    # df['DOY'] = df['UTC_time'].dt.dayofyear + df['UTC_time'].dt.hour / 24 + df['UTC_time'].dt.minute / 1440 + df['UTC_time'].dt.second / 86400
    # df.loc[df['PAR']<-500, 'PAR'] = np.nan
    # df = df.reset_index(drop=True)
    # df.to_csv(os.path.join(savepath, '20220713.csv'), index=False)

    # # 20220905 - 20220916
    # # remove the first part of the data
    # dates = ['20220905', '20220906', '20220907', '20220908', '20220909', '20220910', \
    #          '20220911', '20220912', '20220913', '20220914', '20220915', '20220916']
    # for datestr in dates:
    #     df = pd.read_csv(os.path.join(datapath, f'{datestr}.csv'))
    #     cutpoint = find_cutpoint(df)
    #     if cutpoint is not None:
    #         df2 = df.iloc[cutpoint:, :]
    #         df2 = df2.reset_index(drop=True)
    #         cutpoint1 = find_cutpoint(df2)
    #         if cutpoint1 is not None:
    #             df2 = df2.iloc[cutpoint1:, :]
    #             df2 = df2.reset_index(drop=True)
    #         df2.to_csv(os.path.join(savepath, f'{datestr}.csv'), index=False)


    # %% plot corrected data
    # savepath = os.path.join(root, 'PROCESSED', yearstr)
    # savepath_figs = os.path.join(savepath,'L1','figs_corrected')
    # if not os.path.exists(savepath_figs):
    #     os.makedirs(savepath_figs)
    # # flux data
    # df_flux = glob.glob(os.path.join(path_flux, 'Barbeau_' + yearstr + '*.xls'))[0]
    # df_flux = pd.read_excel(df_flux, sheet_name='data')
    # df_flux.loc[df_flux['hh'] == 0, 'jj'] -= 1
    # df_flux.loc[df_flux['hh'] == 0, 'hh'] = 24
    # df_flux['hh'] -= 0.5
    # df_flux['DOY'] = df_flux['jj'] + df_flux['hh'] / 24

    # csvfiles = glob.glob(os.path.join(savepath,'L1','DataCorrected','*.csv'))
    # for i, csvfile in enumerate(csvfiles):
    #     df = pd.read_csv(csvfile)
    #     datestr = os.path.basename(csvfile).split('.')[0]
    #     doy = datetime.strptime(datestr, '%Y%m%d').timetuple().tm_yday
    #     idx_flux = (df_flux['DOY'] >= doy) & (df_flux['DOY'] < doy + 1)
    #     if os.path.getsize(csvfile) < 1000*1024: # 1000 kb
    #         continue
    #     fig, ax = plt.subplots(2,1,figsize=(10,6), sharex=True)
    #     ax[0].plot(df['DOY'], df['PAR'], 'b-', marker='.',label='PAR (LIF)')
    #     ax[0].plot(df_flux.loc[idx_flux, 'DOY'], df_flux.loc[idx_flux, 'PAR (µmol/m2/s)'], 'k-', marker='.', label='PAR (flux)')
    #     ax[1].plot(df['DOY'], df['Fs'], 'b-', marker='.',label='Fs')
    #     ax[0].set_xlim([df_flux.loc[idx_flux, 'DOY'].min(), df_flux.loc[idx_flux, 'DOY'].max()])
    #     ax[1].set_xlim([df_flux.loc[idx_flux, 'DOY'].min(), df_flux.loc[idx_flux, 'DOY'].max()])
    #     # ax[0].set_xlabel('DOY')
    #     ax[0].set_ylabel('PAR (µmol m$^{-2}$ s$^{-1}$)')
    #     ax[0].legend()
    #     ax[1].set_xlabel('DOY')
    #     ax[1].set_ylabel('Fs (-)')
    #     ax[1].legend()

    #     fig.savefig(os.path.join(savepath_figs, f'{datestr}_PAR_Fs.png'), dpi=300)

    


    # %% comparison with PAR 1 s
    # # 2022 data
    # meteo_path = r'E:\Datahub\Barbeau\Data_flux\ICOS_PPFD\PPFD_20sec_2022'
    # lif_path = r'E:\Datahub\Barbeau\Data_LIF\A_LIF_PAR_Time_Cor\µLIDAR_situ_data_Barbeau\PROCESSED\2022\L2\Daily'
    # savepath = r'E:\Datahub\Barbeau\Data_LIF\A_LIF_PAR_Time_Cor\µLIDAR_situ_data_Barbeau\PROCESSED\2022\L2\Daily_figs_PAR_validation'

    # if not os.path.exists(savepath):
    #     os.makedirs(savepath)
    # lif_files = glob.glob(os.path.join(lif_path, '*.csv'))
    
    # for i, lif_file in enumerate(lif_files):
    #     date_str = lif_file.split('\\')[-1].replace('.csv', '')
    #     year = date_str[:4]
    #     df = pd.read_csv(lif_file)
    #     print(f'Processing file {i+1}/{len(lif_files)}: {lif_file.split("\\")[-1]}')

    #     # 查找date_str对应的meteo文件
    #     if len(glob.glob(meteo_path + f'/*{date_str}*.dat'))>0:
    #         meteo_file = glob.glob(meteo_path + f'/*{date_str}*.dat')[0]
    #         # 读取meteo文件
    #         meteo = pd.read_csv(meteo_file, sep=',', skiprows=3)
    #         meteo.columns = pd.read_csv(meteo_file, sep=',', skiprows=1).columns
    #         # ICOS 时间戳转换为 UTC 时间
    #         # ICOS时间戳是CET时间，转换为UTC时间（+1h）, CET = UTC + 1h
    #         meteo['UTC_time'] = pd.to_datetime(meteo['TimeStamps'], format='%Y%m%d%H%M%S') -  timedelta(hours=1)
    #         meteo['DOY'] = meteo['UTC_time'].dt.dayofyear + meteo['UTC_time'].dt.hour / 24 + meteo['UTC_time'].dt.minute / 1440 + meteo['UTC_time'].dt.second / 86400
            
    #         fig, axs = plt.subplots(figsize=(12, 6))
    #         fig.subplots_adjust(wspace=0.28, bottom = 0.225)
    #         axs.plot(df['DOY'], df['PAR'], c = 'k', label='LIF PAR (Corrected)')
    #         axs.plot(meteo['DOY'], meteo['PPFD_IN_1_1_1'], c = 'r', ls = '--', label='ICOS PPFD (20s)')
    #         # METEO = pd.concat([METEO, meteo])
    #         axs.set_title('Validation of PAR (LIF vs ICOS)')
    #         axs.set_xlabel('DOY (UTC time)')
    #         axs.set_ylabel('PAR' + unitpar)
    #         axs.legend()
    #         fig.savefig(savepath + '/' + lif_file.split('\\')[-1].replace('.csv', '.jpg'), dpi=300)
    #     else:
    #         print(f'No corresponding meteo file found for {date_str}, only plot LIF data.')
    #         continue
        
    #     print('Processing file {}/{}: {}'.format(i+1, len(lif_files), lif_file.split('\\')[-1]))
    # 
    #  


    # %% PAR validation for 2025 data  
    # # 2025 data
    # meteo_path = r'E:\Datahub\Barbeau\Data_flux\ICOS_PPFD\PPFD_20sec_2025\csvfiles'
    # lif_path = r'E:\Datahub\Barbeau\Data_LIF\A_LIF_PAR_Time_Cor\µLIDAR_situ_data_Barbeau\PROCESSED\2025\L2\Daily'
    # savepath = r'E:\Datahub\Barbeau\Data_LIF\A_LIF_PAR_Time_Cor\µLIDAR_situ_data_Barbeau\PROCESSED\2025\L2\Daily_figs_PAR_validation'
    # if not os.path.exists(savepath):
    #     os.makedirs(savepath)
    # lif_files = glob.glob(os.path.join(lif_path, '*.csv'))
    
    # for i, lif_file in enumerate(lif_files):
    #     date_str = lif_file.split('\\')[-1].replace('.csv', '')
    #     year = date_str[:4]
    #     df = pd.read_csv(lif_file)
    #     print(f'Processing file {i+1}/{len(lif_files)}: {lif_file.split("\\")[-1]}')

    #     # 查找date_str对应的meteo文件
    #     if len(glob.glob(meteo_path + f'/*{date_str}*.csv'))>0:
    #         meteo_file = glob.glob(meteo_path + f'/*{date_str}*.csv')[0]
    #         # 读取meteo文件
    #         meteo = pd.read_csv(meteo_file, sep=',', skiprows=2)
    #         meteo.columns = pd.read_csv(meteo_file, sep=',', skiprows=0).columns
    #         # ICOS 时间戳转换为 UTC 时间
    #         # ICOS时间戳是CET时间，转换为UTC时间（+1h）, CET = UTC + 1h
    #         meteo['UTC_time'] = pd.to_datetime(meteo['TIMESTAMP'], format='%Y%m%d%H%M%S') -  timedelta(hours=1)
    #         meteo['DOY'] = meteo['UTC_time'].dt.dayofyear + meteo['UTC_time'].dt.hour / 24 + meteo['UTC_time'].dt.minute / 1440 + meteo['UTC_time'].dt.second / 86400
            
    #         fig, axs = plt.subplots(figsize=(12, 6))
    #         fig.subplots_adjust(wspace=0.28, bottom = 0.225)
    #         axs.plot(df['DOY'], df['PAR'], c = 'k', label='LIF PAR (Corrected)')
    #         axs.plot(meteo['DOY'], meteo['PPFD_IN_1_1_1'], c = 'r', ls = '--', label='ICOS PPFD (20s)')
    #         # METEO = pd.concat([METEO, meteo])
    #         axs.set_title('Validation of PAR (LIF vs ICOS)')
    #         axs.set_xlabel('DOY (UTC time)')
    #         axs.set_ylabel('PAR' + unitpar)
    #         axs.legend()
    #         fig.savefig(savepath + '/' + lif_file.split('\\')[-1].replace('.csv', '.jpg'), dpi=300)
    #     else:
    #         print(f'No corresponding meteo file found for {date_str}, only plot LIF data.')
    #         continue
        
    #     print('Processing file {}/{}: {}'.format(i+1, len(lif_files), lif_file.split('\\')[-1]))

    # %% copy corrected data to L2 folder and save yearly file
    # path_raw = r'E:\Datahub\Barbeau\Data_LIF\A_LIF_PAR_Time_Cor\µLIDAR_situ_data_Barbeau\PROCESSED\2022\L0'
    # path_cor = r'E:\Datahub\Barbeau\Data_LIF\A_LIF_PAR_Time_Cor\µLIDAR_situ_data_Barbeau\PROCESSED\2022\L1\DataCorrected'
    # savepath = r'E:\Datahub\Barbeau\Data_LIF\A_LIF_PAR_Time_Cor\µLIDAR_situ_data_Barbeau\PROCESSED\2022\L2'
    # if not os.path.exists(savepath+'/Daily'):
    #     os.makedirs(savepath+'/Daily')
    # if not os.path.exists(savepath+'/Yearly'):
    #     os.makedirs(savepath+'/Yearly')
    
    # csvfiles_cor = glob.glob(os.path.join(path_cor, '*.csv'))
    # csvfiles_raw = glob.glob(os.path.join(path_raw, '*.csv'))
    # csvfiles_cor_names = [os.path.basename(f) for f in csvfiles_cor]
    # df_all = pd.DataFrame() # to store all data for yearly file
    # for i, csvfile in enumerate(csvfiles_raw):
    #     if os.path.getsize(csvfile) < 2000*1024: # 2000 kb
    #         continue
    #     filename = os.path.basename(csvfile)
    #     if filename not in csvfiles_cor_names:
    #         # copy csvfile to savepath
    #         shutil.copy(csvfile, os.path.join(savepath, 'Daily', filename))
    #         df_cor = pd.read_csv(csvfile)
    #         df_all = pd.concat([df_all, df_cor], ignore_index=True)
    #     else:
    #         shutil.copy(os.path.join(path_cor, filename), os.path.join(savepath, 'Daily', filename))
    #         df_cor = pd.read_csv(os.path.join(path_cor, filename))
    #         df_all = pd.concat([df_all, df_cor], ignore_index=True)
    # df_all.to_csv(os.path.join(savepath, 'Yearly', f'{yearstr}_LIF.csv'), index=False)

    # %% ==========================================================================================
    # ================================== Correction for data 2023 =================================
    # =============================================================================================
    # datapath = r'E:\Datahub\Barbeau\Data_LIF\A_LIF_PAR_Time_Cor\µLIDAR_situ_data_Barbeau\PROCESSED\2023\L1\Datachecked'
    # savepath = r'E:\Datahub\Barbeau\Data_LIF\A_LIF_PAR_Time_Cor\µLIDAR_situ_data_Barbeau\PROCESSED\2023\L1\DataCorrected'
    # # # 20231025 - 20231101
    # # # remove the second part of the data
    # dates = ['20231025', '20231026', '20231027', '20231028', '20231029', '20231030', \
    #          '20231031', '20231101']
    # for datestr in dates:
    #     df = pd.read_csv(os.path.join(datapath, f'{datestr}.csv'))
    #     cutpoint = find_cutpoint(df)
    #     if cutpoint is not None:
    #         df2 = df.iloc[:cutpoint, :]
    #         df2 = df2.reset_index(drop=True)
    #         cutpoint1 = find_cutpoint(df2)
    #         if cutpoint1 is not None:
    #             df2 = df2.iloc[cutpoint1:, :]
    #             df2 = df2.reset_index(drop=True)
    #         df2.to_csv(os.path.join(savepath, f'{datestr}.csv'), index=False)

    # the rest in Datachecked folder are discarded due to the unworthyness to do the correction, 
    # and we will not correct them for now.
    
    # # copy corrected data to L2 folder and save yearly file
    # path_raw = r'E:\Datahub\Barbeau\Data_LIF\A_LIF_PAR_Time_Cor\µLIDAR_situ_data_Barbeau\PROCESSED\2023\L0'
    # path_cor = r'E:\Datahub\Barbeau\Data_LIF\A_LIF_PAR_Time_Cor\µLIDAR_situ_data_Barbeau\PROCESSED\2023\L1\DataCorrected'
    # path_chk = r'E:\Datahub\Barbeau\Data_LIF\A_LIF_PAR_Time_Cor\µLIDAR_situ_data_Barbeau\PROCESSED\2023\L1\Datachecked'
    # savepath = r'E:\Datahub\Barbeau\Data_LIF\A_LIF_PAR_Time_Cor\µLIDAR_situ_data_Barbeau\PROCESSED\2023\L2'
    # if not os.path.exists(savepath+'/Daily'):
    #     os.makedirs(savepath+'/Daily')
    # if not os.path.exists(savepath+'/Yearly'):
    #     os.makedirs(savepath+'/Yearly')
    
    # csvfiles_cor = glob.glob(os.path.join(path_cor, '*.csv'))
    # csvfiles_raw = glob.glob(os.path.join(path_raw, '*.csv'))
    # csvfiles_cor_names = [os.path.basename(f) for f in csvfiles_cor]
    # csvfiles_chk_names = [os.path.basename(f) for f in glob.glob(os.path.join(path_chk, '*.csv'))]
    # df_all = pd.DataFrame() # to store all data for yearly file
    # for i, csvfile in enumerate(csvfiles_raw):
    #     month = int(os.path.basename(csvfile)[4:6])
    #     if (month>=3) and (month<=11):
    #         if (os.path.getsize(csvfile) < 2000*1024): # 2000 kb
    #             continue
    #         filename = os.path.basename(csvfile)
    #         if (filename not in csvfiles_cor_names) and (filename not in csvfiles_chk_names):
    #             # copy csvfile to savepath
    #             shutil.copy(csvfile, os.path.join(savepath, 'Daily', filename))
    #             df_cor = pd.read_csv(csvfile)
    #             df_all = pd.concat([df_all, df_cor], ignore_index=True)
    #         elif (filename in csvfiles_cor_names):
    #             shutil.copy(os.path.join(path_cor, filename), os.path.join(savepath, 'Daily', filename))
    #             df_cor = pd.read_csv(os.path.join(path_cor, filename))
    #             df_all = pd.concat([df_all, df_cor], ignore_index=True)
    # df_all.to_csv(os.path.join(savepath, 'Yearly', f'{yearstr}_LIF.csv'), index=False)
    # %% plot corrected data
    # yearstr = '2023'
    # savepath = os.path.join(root, 'PROCESSED', yearstr)

    # savepath_figs = os.path.join(savepath,'L1','figs_corrected')
    # if not os.path.exists(savepath_figs):
    #     os.makedirs(savepath_figs)
    # # flux data
    # df_flux = glob.glob(os.path.join(path_flux, 'Barbeau_' + yearstr + '*.xls'))[0]
    # df_flux = pd.read_excel(df_flux, sheet_name='data')
    # df_flux.loc[df_flux['hh'] == 0, 'jj'] -= 1
    # df_flux.loc[df_flux['hh'] == 0, 'hh'] = 24
    # df_flux['hh'] -= 0.5
    # df_flux['DOY'] = df_flux['jj'] + df_flux['hh'] / 24

    # csvfiles = glob.glob(os.path.join(savepath,'L1','DataCorrected','*.csv'))
    # for i, csvfile in enumerate(csvfiles):
    #     df = pd.read_csv(csvfile)
    #     datestr = os.path.basename(csvfile).split('.')[0]
    #     doy = datetime.strptime(datestr, '%Y%m%d').timetuple().tm_yday
    #     idx_flux = (df_flux['DOY'] >= doy) & (df_flux['DOY'] < doy + 1)
    #     if os.path.getsize(csvfile) < 2000*1024: # 2000 kb
    #         continue
    #     fig, ax = plt.subplots(2,1,figsize=(10,6), sharex=True)
    #     ax[0].plot(df['DOY'], df['PAR'], 'b-', marker='.',label='PAR (LIF)')
    #     ax[0].plot(df_flux.loc[idx_flux, 'DOY'], df_flux.loc[idx_flux, 'PAR (µmol/m2/s)'], 'k-', marker='.', label='PAR (flux)')
    #     ax[1].plot(df['DOY'], df['Fs'], 'b-', marker='.',label='Fs')
    #     ax[0].set_xlim([df_flux.loc[idx_flux, 'DOY'].min(), df_flux.loc[idx_flux, 'DOY'].max()])
    #     ax[1].set_xlim([df_flux.loc[idx_flux, 'DOY'].min(), df_flux.loc[idx_flux, 'DOY'].max()])
    #     # ax[0].set_xlabel('DOY')
    #     ax[0].set_ylabel('PAR (µmol m$^{-2}$ s$^{-1}$)')
    #     ax[0].legend()
    #     ax[1].set_xlabel('DOY')
    #     ax[1].set_ylabel('Fs (-)')
    #     ax[1].legend()

    #     fig.savefig(os.path.join(savepath_figs, f'{datestr}_PAR_Fs.png'), dpi=300)
    # %% ==========================================================================================
    # ================================== Correction for data 2024 =================================
    # =============================================================================================
    # datapath = r'E:\Datahub\Barbeau\Data_LIF\A_LIF_PAR_Time_Cor\µLIDAR_situ_data_Barbeau\PROCESSED\2024\L1\Datachecked'
    # savepath = r'E:\Datahub\Barbeau\Data_LIF\A_LIF_PAR_Time_Cor\µLIDAR_situ_data_Barbeau\PROCESSED\2024\L1\DataCorrected'
    # # # # 20240602 - 20240615
    # # # # remove the first part of the data
    # dates = ['20240602', '20240603', '20240604', '20240605', '20240606', '20240607', \
    #          '20240608', '20240609', '20240610', '20240611', '20240612', '20240613', \
    #             '20240614', '20240615']
    # for datestr in dates:
    #     df = pd.read_csv(os.path.join(datapath, f'{datestr}.csv'))
    #     cutpoint = find_cutpoint(df)
    #     if cutpoint is not None:
    #         df2 = df.iloc[cutpoint:, :]
    #         df2 = df2.reset_index(drop=True)
    #         cutpoint1 = find_cutpoint(df2)
    #         if cutpoint1 is not None:
    #             df2 = df2.iloc[:cutpoint1, :]
    #             df2 = df2.reset_index(drop=True)
    #         df2.to_csv(os.path.join(savepath, f'{datestr}.csv'), index=False)
    # # # # 20240821 - 20240912
    # # # remove the second part of the data
    # dates = ['20240821', '20240822', '20240823', '20240824', '20240825', \
    #          '20240826', '20240829', '20240830', \
    #             '20240901', '20240902', '20240903', '20240904', '20240905', '20240906', \
    #                 '20240907', '20240910', '20240911', '20241015','20241020','20241022', \
    #                     '20241023','20241024','20241025','20241026','20241105','20241129','20241130']
    # for datestr in dates:
    #     df = pd.read_csv(os.path.join(datapath, f'{datestr}.csv'))
    #     cutpoint = find_cutpoint(df)
    #     if cutpoint is not None:
    #         df2 = df.iloc[:cutpoint, :]
    #         df2 = df2.reset_index(drop=True)
    #         cutpoint1 = find_cutpoint(df2)
    #         if cutpoint1 is not None:
    #             df2 = df2.iloc[:cutpoint1, :]
    #             df2 = df2.reset_index(drop=True)
    #         df2.to_csv(os.path.join(savepath, f'{datestr}.csv'), index=False)
   
    # # keep middle part of the data for 20240827 - 20240912, and remove the first and second part of the data
    # dates = ['20240827', '20240828','20240831','20240908','20240909', '20240912', \
    #          '20241016','20241017','20241018','20241019','20241021','20241027', \
    #             '20241028','20241029']
    # for datestr in dates:
    #     df = pd.read_csv(os.path.join(datapath, f'{datestr}.csv'))
    #     cutpoint = find_cutpoint(df)
    #     if cutpoint is not None:
    #         df2 = df.iloc[cutpoint:, :]
    #         df2 = df2.reset_index(drop=True)
    #         cutpoint1 = find_cutpoint(df2)
    #         if cutpoint1 is not None:
    #             df2 = df2.iloc[:cutpoint1, :]
    #             df2 = df2.reset_index(drop=True)
    #         df2.to_csv(os.path.join(savepath, f'{datestr}.csv'), index=False)

    # the rest in Datachecked folder are discarded due to the unworthyness to do the correction, 
    # and we will not correct them for now.
    
    # # copy corrected data to L2 folder and save yearly file
    # path_raw = r'E:\Datahub\Barbeau\Data_LIF\A_LIF_PAR_Time_Cor\µLIDAR_situ_data_Barbeau\PROCESSED\2024\L0'
    # path_cor = r'E:\Datahub\Barbeau\Data_LIF\A_LIF_PAR_Time_Cor\µLIDAR_situ_data_Barbeau\PROCESSED\2024\L1\DataCorrected'
    # path_chk = r'E:\Datahub\Barbeau\Data_LIF\A_LIF_PAR_Time_Cor\µLIDAR_situ_data_Barbeau\PROCESSED\2024\L1\Datachecked'
    # savepath = r'E:\Datahub\Barbeau\Data_LIF\A_LIF_PAR_Time_Cor\µLIDAR_situ_data_Barbeau\PROCESSED\2024\L2'
    # if not os.path.exists(savepath+'/Daily'):
    #     os.makedirs(savepath+'/Daily')
    # if not os.path.exists(savepath+'/Yearly'):
    #     os.makedirs(savepath+'/Yearly')
    
    # csvfiles_cor = glob.glob(os.path.join(path_cor, '*.csv'))
    # csvfiles_raw = glob.glob(os.path.join(path_raw, '*.csv'))
    # csvfiles_cor_names = [os.path.basename(f) for f in csvfiles_cor]
    # csvfiles_chk_names = [os.path.basename(f) for f in glob.glob(os.path.join(path_chk, '*.csv'))]
    # df_all = pd.DataFrame() # to store all data for yearly file
    # for i, csvfile in enumerate(csvfiles_raw):
    #     month = int(os.path.basename(csvfile)[4:6])
    #     if (month>=3) and (month<=11):
    #         if (os.path.getsize(csvfile) < 2000*1024): # 2000 kb
    #             continue
    #         filename = os.path.basename(csvfile)
    #         if (filename not in csvfiles_cor_names) and (filename not in csvfiles_chk_names):
    #             # copy csvfile to savepath
    #             shutil.copy(csvfile, os.path.join(savepath, 'Daily', filename))
    #             df_cor = pd.read_csv(csvfile)
    #             df_all = pd.concat([df_all, df_cor], ignore_index=True)
    #         elif (filename in csvfiles_cor_names):
    #             shutil.copy(os.path.join(path_cor, filename), os.path.join(savepath, 'Daily', filename))
    #             df_cor = pd.read_csv(os.path.join(path_cor, filename))
    #             df_all = pd.concat([df_all, df_cor], ignore_index=True)
    # df_all.to_csv(os.path.join(savepath, 'Yearly', f'{yearstr}_LIF.csv'), index=False)

    # %% plot corrected data
    # yearstr = '2024'
    # savepath = os.path.join(root, 'PROCESSED', yearstr)

    # savepath_figs = os.path.join(savepath,'L1','figs_corrected')
    # if not os.path.exists(savepath_figs):
    #     os.makedirs(savepath_figs)
    # # flux data
    # df_flux = glob.glob(os.path.join(path_flux, 'Barbeau_' + yearstr + '*.xls'))[0]
    # df_flux = pd.read_excel(df_flux, sheet_name='data')
    # df_flux.loc[df_flux['hh'] == 0, 'jj'] -= 1
    # df_flux.loc[df_flux['hh'] == 0, 'hh'] = 24
    # df_flux['hh'] -= 0.5
    # df_flux['DOY'] = df_flux['jj'] + df_flux['hh'] / 24

    # csvfiles = glob.glob(os.path.join(savepath,'L1','DataCorrected','*.csv'))
    # for i, csvfile in enumerate(csvfiles):
    #     df = pd.read_csv(csvfile)
    #     datestr = os.path.basename(csvfile).split('.')[0]
    #     doy = datetime.strptime(datestr, '%Y%m%d').timetuple().tm_yday
    #     idx_flux = (df_flux['DOY'] >= doy) & (df_flux['DOY'] < doy + 1)
    #     if os.path.getsize(csvfile) < 2000*1024: # 2000 kb
    #         continue
    #     fig, ax = plt.subplots(2,1,figsize=(10,6), sharex=True)
    #     ax[0].plot(df['DOY'], df['PAR'], 'b-', marker='.',label='PAR (LIF)')
    #     ax[0].plot(df_flux.loc[idx_flux, 'DOY'], df_flux.loc[idx_flux, 'PAR (µmol/m2/s)'], 'k-', marker='.', label='PAR (flux)')
    #     ax[1].plot(df['DOY'], df['Fs'], 'b-', marker='.',label='Fs')
    #     ax[0].set_xlim([df_flux.loc[idx_flux, 'DOY'].min(), df_flux.loc[idx_flux, 'DOY'].max()])
    #     ax[1].set_xlim([df_flux.loc[idx_flux, 'DOY'].min(), df_flux.loc[idx_flux, 'DOY'].max()])
    #     # ax[0].set_xlabel('DOY')
    #     ax[0].set_ylabel('PAR (µmol m$^{-2}$ s$^{-1}$)')
    #     ax[0].legend()
    #     ax[1].set_xlabel('DOY')
    #     ax[1].set_ylabel('Fs (-)')
    #     ax[1].legend()

    #     fig.savefig(os.path.join(savepath_figs, f'{datestr}_PAR_Fs.png'), dpi=300)

    # %% ==========================================================================================
    # =======================================for data 2025 ========================================
    # =============================================================================================
    # 20250309
    # remove the second part of the data
    # 20250625
    # remove the first part of the data
    # 20250729
    # remove some outliers in the data

    # %% plot corrected data
    # yearstr = '2025'
    # savepath = os.path.join(root, 'PROCESSED', yearstr)

    # # savepath_figs = os.path.join(savepath,'L1','figs_corrected')
    # # if not os.path.exists(savepath_figs):
    # #     os.makedirs(savepath_figs)
    # # # flux data
    # # df_flux = glob.glob(os.path.join(path_flux, 'Barbeau_' + yearstr + '*.xls'))[0]
    # # df_flux = pd.read_excel(df_flux, sheet_name='data')
    # # df_flux.loc[df_flux['hh'] == 0, 'jj'] -= 1
    # # df_flux.loc[df_flux['hh'] == 0, 'hh'] = 24
    # # df_flux['hh'] -= 0.5
    # # df_flux['DOY'] = df_flux['jj'] + df_flux['hh'] / 24

    # # csvfiles = glob.glob(os.path.join(savepath,'L1','DataCorrected','*.csv'))
    # # for i, csvfile in enumerate(csvfiles):
    # #     df = pd.read_csv(csvfile)
    # #     datestr = os.path.basename(csvfile).split('.')[0]
    # #     doy = datetime.strptime(datestr, '%Y%m%d').timetuple().tm_yday
    # #     idx_flux = (df_flux['DOY'] >= doy) & (df_flux['DOY'] < doy + 1)
    # #     if os.path.getsize(csvfile) < 2000*1024: # 2000 kb
    # #         continue
    # #     fig, ax = plt.subplots(2,1,figsize=(10,6), sharex=True)
    # #     ax[0].plot(df['DOY'], df['PAR'], 'b-', marker='.',label='PAR (LIF)')
    # #     ax[0].plot(df_flux.loc[idx_flux, 'DOY'], df_flux.loc[idx_flux, 'PAR (µmol/m2/s)'], 'k-', marker='.', label='PAR (flux)')
    # #     ax[1].plot(df['DOY'], df['Fs'], 'b-', marker='.',label='Fs')
    # #     ax[0].set_xlim([df_flux.loc[idx_flux, 'DOY'].min(), df_flux.loc[idx_flux, 'DOY'].max()])
    # #     ax[1].set_xlim([df_flux.loc[idx_flux, 'DOY'].min(), df_flux.loc[idx_flux, 'DOY'].max()])
    # #     # ax[0].set_xlabel('DOY')
    # #     ax[0].set_ylabel('PAR (µmol m$^{-2}$ s$^{-1}$)')
    # #     ax[0].legend()
    # #     ax[1].set_xlabel('DOY')
    # #     ax[1].set_ylabel('Fs (-)')
    # #     ax[1].legend()

    # #     fig.savefig(os.path.join(savepath_figs, f'{datestr}_PAR_Fs.png'), dpi=300)

    # # %% copy corrected data to L2 folder and save yearly file
    # path_raw = r'E:\Datahub\Barbeau\Data_LIF\A_LIF_PAR_Time_Cor\µLIDAR_situ_data_Barbeau\PROCESSED\2025\L0'
    # path_cor = r'E:\Datahub\Barbeau\Data_LIF\A_LIF_PAR_Time_Cor\µLIDAR_situ_data_Barbeau\PROCESSED\2025\L1\DataCorrected'
    # savepath = r'E:\Datahub\Barbeau\Data_LIF\A_LIF_PAR_Time_Cor\µLIDAR_situ_data_Barbeau\PROCESSED\2025\L2'
    # if not os.path.exists(savepath+'/Daily'):
    #     os.makedirs(savepath+'/Daily')
    # if not os.path.exists(savepath+'/Yearly'):
    #     os.makedirs(savepath+'/Yearly')
    
    # csvfiles_cor = glob.glob(os.path.join(path_cor, '*.csv'))
    # csvfiles_raw = glob.glob(os.path.join(path_raw, '*.csv'))
    # csvfiles_cor_names = [os.path.basename(f) for f in csvfiles_cor]
    # df_all = pd.DataFrame() # to store all data for yearly file
    # for i, csvfile in enumerate(csvfiles_raw):
    #     if os.path.getsize(csvfile) < 2000*1024: # 2000 kb
    #         continue
    #     filename = os.path.basename(csvfile)
    #     if filename not in csvfiles_cor_names:
    #         # copy csvfile to savepath
    #         shutil.copy(csvfile, os.path.join(savepath, 'Daily', filename))
    #         df_cor = pd.read_csv(csvfile)
    #         df_all = pd.concat([df_all, df_cor], ignore_index=True)
    #     else:
    #         shutil.copy(os.path.join(path_cor, filename), os.path.join(savepath, 'Daily', filename))
    #         df_cor = pd.read_csv(os.path.join(path_cor, filename))
    #         df_all = pd.concat([df_all, df_cor], ignore_index=True)
    # df_all.to_csv(os.path.join(savepath, 'Yearly', f'{yearstr}_LIF.csv'), index=False)


    # %% ==========================================================================================
    # =======================================for data 2026 ========================================
    # =============================================================================================
    # # %% copy corrected data to L2 folder and save yearly file

    path_raw = r'E:\Datahub\Barbeau\Data_LIF\A_LIF_PAR_Time_Cor\µLIDAR_situ_data_Barbeau\PROCESSED\2026\L0'
    path_cor = r'E:\Datahub\Barbeau\Data_LIF\A_LIF_PAR_Time_Cor\µLIDAR_situ_data_Barbeau\PROCESSED\2026\L1\DataCorrected'
    path_chk = r'E:\Datahub\Barbeau\Data_LIF\A_LIF_PAR_Time_Cor\µLIDAR_situ_data_Barbeau\PROCESSED\2026\L1\Datachecked'
    savepath = r'E:\Datahub\Barbeau\Data_LIF\A_LIF_PAR_Time_Cor\µLIDAR_situ_data_Barbeau\PROCESSED\2026\L2'
    if not os.path.exists(savepath+'/Daily'):
        os.makedirs(savepath+'/Daily')
    if not os.path.exists(savepath+'/Yearly'):
        os.makedirs(savepath+'/Yearly')
    
    csvfiles_cor = glob.glob(os.path.join(path_cor, '*.csv'))
    csvfiles_raw = glob.glob(os.path.join(path_raw, '*.csv'))
    csvfiles_cor_names = [os.path.basename(f) for f in csvfiles_cor]
    csvfiles_chk_names = [os.path.basename(f) for f in glob.glob(os.path.join(path_chk, '*.csv'))]
    df_all = pd.DataFrame() # to store all data for yearly file
    for i, csvfile in enumerate(csvfiles_raw):
        month = int(os.path.basename(csvfile)[4:6])
        if (month>=3) and (month<=11):
            if (os.path.getsize(csvfile) < 2000*1024): # 2000 kb
                continue
            filename = os.path.basename(csvfile)
            if (filename not in csvfiles_cor_names) and (filename not in csvfiles_chk_names):
                # copy csvfile to savepath
                shutil.copy(csvfile, os.path.join(savepath, 'Daily', filename))
                df_cor = pd.read_csv(csvfile)
                df_all = pd.concat([df_all, df_cor], ignore_index=True)
            elif (filename in csvfiles_cor_names):
                shutil.copy(os.path.join(path_cor, filename), os.path.join(savepath, 'Daily', filename))
                df_cor = pd.read_csv(os.path.join(path_cor, filename))
                df_all = pd.concat([df_all, df_cor], ignore_index=True)
    df_all.to_csv(os.path.join(savepath, 'Yearly', f'{yearstr}_LIF.csv'), index=False)