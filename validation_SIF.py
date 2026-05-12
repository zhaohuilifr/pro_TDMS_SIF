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

sif_mat = r'E:\Datahub\Barbeau\Data_SIF\SIF3data\2022\PROCESSED_matlab\L1\SIF_2022.csv'
sif_py = r'E:\Datahub\Barbeau\Data_SIF\SIF3data\2022\PROCESSED\L2\Yearly\PROSIF_SIFresults_2022_Yearly.csv'

df_mat = pd.read_csv(sif_mat)
# df_mat['Time_start'] = df_mat['Time_start'].apply(matlab2datetime)
df_mat['SIF_3FLD_O2A'] = df_mat['SIF_3FLD_O2A'] * 1e3 # 转换为 mW/m^2/sr/nm

df_py = pd.read_csv(sif_py)
df_py['SIF_3FLD'] = df_py['SIF_3FLD']/np.pi
print(df_mat.loc[0, 'Time_start'], df_py.loc[152, 'Time_start'])

fig, ax = plt.subplots(figsize=(10, 6))
# idx_mat =
idx_py = df_py['prcsppix_sat'] == 0
ax.scatter(df_mat['Time_start'], df_mat['SIF_3FLD_O2A'], marker = '.', 
           color = 'b',label='MATLAB')
ax.scatter(df_py.loc[idx_py, 'Time_start'], df_py.loc[idx_py, 'SIF_3FLD'], 
           marker = '.', color= 'orange',label='Python', linestyle='--')
ax.set_ylim([0, 5])
ax.legend()
ax.set_xlabel('Time')
plt.show()