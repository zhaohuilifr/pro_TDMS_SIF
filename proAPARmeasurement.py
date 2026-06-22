# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import glob

def fluxdata_doy_calculation(fluxdata):
    # fluxdata uses CET time, which is UTC+1
    mask = fluxdata['hh'] == 0
    fluxdata.loc[mask, 'date'] = pd.to_datetime({'year': fluxdata.loc[mask, 'an'],
                                                  'month': fluxdata.loc[mask, 'mois'],
                                                  'day': fluxdata.loc[mask, 'jour']}) - pd.Timedelta(days=1)
    fluxdata.loc[~mask, 'date'] = pd.to_datetime({'year': fluxdata.loc[~mask, 'an'],
                                                   'month': fluxdata.loc[~mask, 'mois'],
                                                   'day': fluxdata.loc[~mask, 'jour']})
    fluxdata.loc[mask, 'hh'] = 24
    fluxdata.loc[mask, 'jj'] = fluxdata.loc[mask, 'jj'] - 1
    # 从日期提取年月日
    fluxdata['an'] = fluxdata['date'].dt.year
    fluxdata['mois'] = fluxdata['date'].dt.month
    fluxdata['jour'] = fluxdata['date'].dt.day
    fluxdata = fluxdata.drop('date', axis=1)
    fluxdata['hh'] -= 0.5

    fluxdata['hour'] = (fluxdata['hh'].astype(int))
    fluxdata['minute'] = (fluxdata['hh'] - fluxdata['hour']) * 60
    fluxdata['DOY'] = fluxdata['jj'] + (fluxdata['hour'] + fluxdata['minute'] / 60) / 24

    # !!!!!!!!!!!!!!!! CET to UTC conversion !!!!!!!!!!!!!!!!
    fluxdata['DOY_UTC'] = fluxdata['DOY'] - 1/24
    # fluxdata.loc[fluxdata['DOY_UTC'] < 0, 'DOY_UTC'] += 365 # for DOY < 1, add 365 to make it positive
    fluxdata['hour_UTC'] = fluxdata['hour'] - 1
    fluxdata.loc[fluxdata['hour_UTC'] < 0, 'hour_UTC'] += 24 # for hour < 0, add 24 to make it positive
    fluxdata['minute_UTC'] = fluxdata['minute']
    # remove rows with DOY_UTC < 0 or DOY_UTC > 365, as they are out of the valid range
    fluxdata = fluxdata.loc[(fluxdata['DOY_UTC'] >= 0) & (fluxdata['DOY_UTC'] <= 365)].reset_index(drop=True)

    return fluxdata


datapath = r'E:\Datahub\Barbeau\Data_flux\Daniel'
df = pd.read_csv(os.path.join(datapath, 'Extraction_PPFD_20220101-0030_to_20251231-2330.csv'), index_col=False)
df_values = df.iloc[1:, :]
df_values.columns = df.columns

df_values['date'] = pd.to_datetime(df_values['TIMESTAMP'], format='%m/%d/%Y %H:%M', errors='raise')
df_values['DOY'] = df_values['date'].dt.dayofyear + (df_values['date'].dt.hour + df_values['date'].dt.minute / 60) / 24

df_values['jj'] = df_values['date'].dt.dayofyear
df_values['hh'] = df_values['date'].dt.hour + df_values['date'].dt.minute / 60
mask = df_values['hh'] ==0
df_values.loc[mask, 'hh'] = 24
df_values.loc[mask, 'jj'] = df_values.loc[mask, 'jj'] - 1
# 从日期提取年月日
df_values['an'] = df_values['date'].dt.year
df_values['mois'] = df_values['date'].dt.month
df_values['jour'] = df_values['date'].dt.day

df_values['hh'] -= 0.5
df_values['hour'] = (df_values['hh'].astype(int))
df_values['minute'] = (df_values['hh'] - df_values['hour']) * 60
df_values['DOY'] = df_values['jj'] + (df_values['hour'] + df_values['minute'] / 60) / 24

# !!!!!!!!!!!!!!!! CET to UTC conversion !!!!!!!!!!!!!!!!
df_values['DOY_UTC'] = df_values['DOY'] - 1/24
# fluxdata.loc[fluxdata['DOY_UTC'] < 0, 'DOY_UTC'] += 365 # for DOY < 1, add 365 to make it positive
df_values['hour_UTC'] = df_values['hour'] - 1
df_values.loc[df_values['hour_UTC'] < 0, 'hour_UTC'] += 24 # for hour < 0, add 24 to make it positive
df_values['minute_UTC'] = df_values['minute']
# remove rows with DOY_UTC < 0 or DOY_UTC > 365, as they are out of the valid range
df_values = df_values.loc[(df_values['DOY_UTC'] >= 0) & (df_values['DOY_UTC'] <= 365)].reset_index(drop=True)

# remove nan values
df_values = df_values.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
df_values[df_values == 0.1] = np.nan
df_values[df_values == -9999] = np.nan

df_res = pd.DataFrame()
df_res['date'] = df_values['date']
df_res['an'] = df_values['an']
df_res['mois'] = df_values['mois']
df_res['jour'] = df_values['jour']
df_res['DOY'] = df_values['DOY']
df_res['hour'] = df_values['hour']
df_res['minute'] = df_values['minute']
df_res['DOY_UTC'] = df_values['DOY_UTC']
df_res['hour_UTC'] = df_values['hour_UTC']
df_res['minute_UTC'] = df_values['minute_UTC']

df_res['PARin(PQS1)'] = df_values['PPFD_IN_1_1_2']
df_res['PARout(PQS1)'] = df_values['PPFD_OUT_1_1_1']
parbc_cols = [f'PPFD_BC_IN_{i}_1_1' for i in range(1, 16)]
df_res['PARbc_1_15(PQS1)'] = df_values[parbc_cols].mean(axis=1, skipna=True)
parbc_cols = [f'PPFD_BC_IN_{i}_1_1' for i in range(16, 36)]
df_res['PARbc_16_35(PQS1)'] = df_values[parbc_cols].mean(axis=1, skipna=True)
parbc_cols = [f'PPFD_BC_IN_{i}_1_1' for i in range(1, 36)]
df_res['PARbc_1_35(PQS1)'] = df_values[parbc_cols].mean(axis=1, skipna=True)
df_res.loc[df_res['PARbc_1_15(PQS1)'] < 0 ,'PARbc_1_15(PQS1)'] = np.nan
df_res.loc[df_res['PARbc_16_35(PQS1)'] < 0 ,'PARbc_16_35(PQS1)'] = np.nan
df_res.loc[df_res['PARbc_1_35(PQS1)'] < 0 ,'PARbc_1_35(PQS1)'] = np.nan


df_res['APAR_1_15'] = df_res['PARin(PQS1)'] - df_res['PARout(PQS1)'] - df_res['PARbc_1_15(PQS1)']
df_res['APAR_16_35'] = df_res['PARin(PQS1)'] - df_res['PARout(PQS1)'] - df_res['PARbc_16_35(PQS1)']
df_res['APAR_1_35'] = df_res['PARin(PQS1)'] - df_res['PARout(PQS1)'] - df_res['PARbc_1_35(PQS1)']

df_res['PARin(BF5)'] = df_values['PPFD_IN_1_1_1']
df_res['PARdir(BF5)'] = df_values['PPFD_DIR_1_1_1']
df_res['PARdif(BF5)'] = df_values['PPFD_DIF_1_1_1']
df_res['frac_dif'] = df_res['PARdif(BF5)'] / df_res['PARin(BF5)']
df_res.loc[(df_res['frac_dif'] < 0) | (df_res['frac_dif'] > 1), 'frac_dif'] = np.nan

df_res.to_excel(os.path.join(datapath, 'Barbeau_APAR_20220101-0030_to_20251231-2330.xlsx'), index=False)

# print('a')