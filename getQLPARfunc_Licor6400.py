# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import glob
from scipy.optimize import curve_fit

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16

def func_ql_par(par, alpha_ql, hf_par):
    beta_ql = hf_par/np.log(2)
    ql = alpha_ql* np.exp(-1/beta_ql*par)
    return ql


datapath = r'E:\Datahub\Barbeau\Data_leaf\Licor_6400\20260726'
df = pd.read_excel(os.path.join(datapath, 'zhaohui-24072026-testql-oak2_.xlsx'), sheet_name='Sheet1')

initial_guess = [0.7, 1000]
bounds = ([0, 0], [1, np.inf])
popt, pcov = curve_fit(func_ql_par, df['PARi'], df['qL'], p0=initial_guess, bounds=bounds)
qL_fitted = func_ql_par(df['PARi'], *popt)
qL_liter = func_ql_par(df['PARi'], 0.67, 761.7)

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(df['PARi'], df['qL'], color='blue', label='Observed qL', alpha=0.7)
textstr = f'Fitted parameters:\nα = {popt[0]:.4f}\nHF_PAR = {popt[1]:.2f}'
props = dict(boxstyle='round', facecolor='white', alpha=0.5)

ax.plot(df['PARi'], qL_fitted, color='red', label='Fitted qL'+'\n'+textstr, linewidth=2)
ax.plot(df['PARi'], qL_liter, color='green', label='Literature qL \n (α=0.67, HF_PAR=761.7)', linewidth=2)
ax.set_xlabel('PAR (µmol m⁻² s⁻¹)')
ax.set_ylabel('qL')
ax.set_title('Fitting qL vs PAR')
ax.legend()
# plt.tight_layout()
plt.show()