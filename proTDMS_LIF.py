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
import pandas as pd
import numpy as np
import glob as glob
from io import StringIO
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from joblib import Parallel, delayed
import tkinter as tk
from tkinter import filedialog

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
        # 02
        # epoch_lv = datetime(1904, 1, 1)
        # df['UTC_time_2'] = df['time_01011904'].apply(lambda x: epoch_lv + timedelta(seconds=x))
        df.loc[idx, 'UTC_time'] = np.nan
        # df.loc[idx, 'UTC_time_2'] = np.nan
        # reset index
        df = df.reset_index(drop=True)
    return df



if __name__ == "__main__":
    # %% 设置路径
    root = r'E:\Datahub\Barbeau\Data_LIF\A_LIF_PAR_Time_Cor\µLIDAR_situ_data_Barbeau'
    lif_path = os.path.join(root, 'RAW')
    meteo_path = r'E:\Datahub\Barbeau\Data_LIF\A_LIF_PAR_Time_Cor\PAR_data\Radiations_20s'
    yearstr = '2025' # '2022'
    for subfolders in ['L0', 'L1', 'L2']:
        if not os.path.exists(os.path.join(root, 'PROCESSED', yearstr,subfolders)):
            os.makedirs(os.path.join(root, 'PROCESSED', yearstr,subfolders))
    # %% process .mfl files
    savepath = os.path.join(root, 'PROCESSED', yearstr, 'L0')
    # mfl_files = glob.glob(os.path.join(lif_path, yearstr,'*.mfl'))
    # dfs = []
    # for i, mfl_file in enumerate(mfl_files):
    #     df = read_mfl(mfl_file)
    #     dfs.append(df)
    #     df.to_csv(os.path.join(savepath, os.path.basename(mfl_file).replace('.mfl', '.csv')), index=False)
    #     print(f'Processing file {i+1}/{len(mfl_files)}: {os.path.basename(mfl_file)}')
    # df_all = pd.concat(dfs, ignore_index=True)
    # # df_all.to_csv(os.path.join(savepath, f'LIF_PAR_time_series_{yearstr}.csv'), index=False)

    # date_start = (df_all.loc[0, 'UTC_time']).date()
    # date_end = (df_all.loc[len(df_all)-1, 'UTC_time']).date()
    # for date in pd.date_range(date_start, date_end):
    #     df_date = df_all[(df_all['UTC_time'] >= date) & (df_all['UTC_time'] < date + pd.Timedelta(days=1))]
    #     if len(df_date) > 0:
    #         df_date.to_csv(os.path.join(savepath.replace('L0', 'L1'), f'{date.strftime("%Y%m%d")}.csv'), index=False)

    # %% remove replicated data
    
    print('a')