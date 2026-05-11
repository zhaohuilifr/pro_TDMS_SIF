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
        # 将时间戳转换为 CET 时间
        df['UTC_time'] = pd.to_datetime(df['time_01011904'], unit='s', origin=pd.Timestamp('1904-01-01'))
        df.loc[idx, 'UTC_time'] = np.nan
    return df

# 写一个可视化.mfl文件的GUI界面，可以选择.mfl文件，显示时间序列图，并且可以选择一个时间范围进行放大查看
def visualize_mfl():
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    mfl_file = filedialog.askopenfilename(title="Select .mfl file", filetypes=[("MFL files", "*.mfl")])
    if not mfl_file:
        print("No file selected.")
        return
    
    df = read_mfl(mfl_file)
    
    plt.figure(figsize=(12, 6))
    plt.plot(df['UTC_time'], df['Fs'], label='Original Time', color='blue')
    plt.xlabel('UTC Time')
    plt.ylabel('Time (s since 1904-01-01)')
    plt.title('Time Series from .mfl File')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    # %% 设置路径
    root = r'E:\Datahub\Barbeau\Data_LIF\A_LIF_PAR_Time_Cor\µLIDAR_situ_data_Barbeau'
    lif_path = os.path.join(root, 'RAW')
    meteo_path = r'E:\Datahub\Barbeau\Data_LIF\A_LIF_PAR_Time_Cor\PAR_data\Radiations_20s'
    for subfolders in ['Corrected', 'Tables']:
        if not os.path.exists(os.path.join(root, subfolders)):
            os.makedirs(os.path.join(root, subfolders))
    savepath = os.path.join(root, 'Corrected')
    savepath_tables = os.path.join(root, 'Tables')

    # %% 读取数据
    visualize_mfl()