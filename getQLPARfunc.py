# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import glob

yearstr = '2025'
path_match = r'E:\Datahub\Barbeau\Data_matched_new1'
savepath_figs = r'E:\Datahub\Barbeau\Data_matched_new1\figs\ql_vs_PAR_{year}'.format(year=yearstr)
os.makedirs(savepath_figs, exist_ok=True)
df_matched = pd.read_excel(os.path.join(path_match, f'Barbeau_{yearstr}_matched_CASTANEA.xlsx'))


df_matched.loc[df_matched['Fs'] < 0.15, 'Fs'] = np.nan
df_matched.loc[df_matched['SIFPSIILAI_yield_top'] < 0 , 'SIFPSIILAI_yield_top'] = np.nan
df_matched.loc[df_matched['SIFPSIILAI_yield_top'] > 0.03 , 'SIFPSIILAI_yield_top'] = np.nan
sifpsii_max = df_matched['SIFPSIILAI_yield_top'].max()
sifpsii_min = df_matched['SIFPSIILAI_yield_top'].min()
# scale Fs to the same range as SIFPSIILAI_yield_top
df_matched['Fs_scaled'] = (df_matched['Fs'] - df_matched['Fs'].min()) \
    / (df_matched['Fs'].max() - df_matched['Fs'].min()) * (sifpsii_max - sifpsii_min) + sifpsii_min

sifyield = df_matched['Fs_scaled'].values
apar = df_matched['PARdiraLAI'] + df_matched['PARdifaLAI']
sifpsii = sifyield * apar
phiPSIImax = 0.81
kDF = 10
ql = df_matched['JaLAI'] * (1 - phiPSIImax) / (phiPSIImax * (1 + kDF) * sifpsii)

idx = (df_matched['hour_UTC'] >= 6) & (df_matched['hour_UTC'] <= 18) & (df_matched['PAR (µmol/m2/s)'] > 0)
idx_day = idx & (df_matched['DOY_UTC'] >= 150) & (df_matched['DOY_UTC'] <= 250)

# %% 把每天的qL归一化到0.2 - 0.85的范围内
ql_day = ql[idx_day]


fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(df_matched.loc[idx_day, 'PAR (µmol/m2/s)'], ql[idx_day], 
           c = df_matched.loc[idx_day,'DOY_UTC'], cmap='viridis', s=20, edgecolor='none')
cbar = plt.colorbar(ax.collections[0], ax=ax)
cbar.set_label('Day of Year (UTC)')
ax.set_xlabel('PAR (µmol/m2/s)')
ax.set_ylabel('qL (unitless)')
fig.savefig(os.path.join(savepath_figs, f'qL_vs_PAR_DOY_150_250_DOYcolor.jpg'), dpi=300)

# fig, ax = plt.subplots(2,2,figsize=(8, 6))
# ax[0,0].scatter(df_matched.loc[idx_day, 'PAR (µmol/m2/s)'], sifpsii[idx_day], \
#            c = df_matched.loc[idx_day,'DOY_UTC'], 
#            cmap='viridis', s=20, edgecolor='none')
# cbar = plt.colorbar(ax[0,0].collections[0], ax=ax[0,0])
# ax[0,1].scatter(df_matched.loc[idx_day, 'SIF_3FLD'], sifpsii[idx_day], \
#            c = df_matched.loc[idx_day,'DOY_UTC'], 
#            cmap='viridis', s=20, edgecolor='none')
# cbar = plt.colorbar(ax[0,1].collections[0], ax=ax[0,1])
# ax[1,0].scatter(df_matched.loc[idx_day, 'PAR (µmol/m2/s)'], df_matched.loc[idx_day,'SIFPSIILAI'], \
#               c = df_matched.loc[idx_day,'DOY_UTC'], 
#               cmap='viridis', s=20, edgecolor='none')
# cbar = plt.colorbar(ax[1,0].collections[0], ax=ax[1,0])
# ax[1,1].scatter(df_matched.loc[idx_day, 'SIF_3FLD'], df_matched.loc[idx_day,'SIFPSIILAI'], \
#               c = df_matched.loc[idx_day,'DOY_UTC'], 
#               cmap='viridis', s=20, edgecolor='none')
# cbar = plt.colorbar(ax[1,1].collections[0], ax=ax[1,1])
# ax[0,0].set_xlabel('PAR (µmol/m2/s)')
# ax[0,0].set_ylabel('SIFsim (FS)', color='blue')
# ax[1,0].set_xlabel('PAR (µmol/m2/s)')
# ax[1,0].set_ylabel('SIFsim (Gu)', color='orange')
# ax[0,1].set_xlabel('SIF_3FLD')
# ax[0,1].set_ylabel('SIFsim (FS)', color='blue')
# ax[1,1].set_xlabel('SIF_3FLD')
# ax[1,1].set_ylabel('SIFsim (Gu)', color='orange')
# fig.savefig(os.path.join(savepath_figs, f'SIFsim_vs_PAR_DOY_150_250_DOYcolor.jpg'), dpi=300)

for i in range(150, 251):
    idx_day_i = idx & (df_matched['DOY_UTC'] >= i) & (df_matched['DOY_UTC'] < i+1)
    if idx_day_i.sum() == 0:
        continue
    fig, ax = plt.subplots(2,1, figsize=(8, 6))
    ax[0].scatter(df_matched.loc[idx_day_i, 'PAR (µmol/m2/s)'], ql[idx_day_i], \
               c = df_matched.loc[idx_day_i,'hour_UTC']+df_matched.loc[idx_day_i,'minute_UTC']/30.0, 
               cmap='viridis', s=20, edgecolor='none')
    cbar = plt.colorbar(ax[0].collections[0], ax=ax[0])
    cbar.set_label('Hour of Day (UTC)')
    ax[0].set_xlabel('PAR (µmol/m2/s)')
    ax[0].set_ylabel('qL (unitless)')
    ax[0].set_ylim(0, 1)
    ax[0].set_xlim(0, 2050)
    ax[1].plot(df_matched.loc[idx_day_i,'DOY_UTC'], sifpsii[idx_day_i], marker = 'o', color = 'blue', label='SIFsim (FS)')
    ax1 = ax[1].twinx()
    ax1.plot(df_matched.loc[idx_day_i,'DOY_UTC'], df_matched.loc[idx_day_i,'PAR (µmol/m2/s)'], marker = 'o', color='orange', label='SIFsim (Gu)')
    # # 再添加一个y轴来绘制SIFcanopy_760nm
    # ax2 = ax[1].twinx()
    # # 调整第三个y轴的位置，避免与第二个y轴重叠
    # ax2.spines['right'].set_position(('outward', 60))  # 将第三个y轴向右移动60像素
    # ax2.plot(df_matched.loc[idx_day_i,'DOY_UTC'], df_matched.loc[idx_day_i,'SIFcanopy_760nm'], marker = 'o', color='green', label='SIFcanopy_760nm (Gu)')

    ax[1].set_xlabel('Hour of Day (UTC)')
    ax[1].set_ylabel('SIFPSIILAI')
    ax[1].legend(loc='upper left')
    ax1.set_ylabel('SIFsim (FS)', color='blue')
    ax1.legend(loc='upper right')
    # ax2.set_ylabel('SIFcanopy_760nm (Gu)', color='green')
    # ax2.legend(loc='center right')
    
    fig.savefig(os.path.join(savepath_figs, f'qL_vs_PAR_DOY_{i}.jpg'), dpi=300)
    plt.close(fig)
# plt.show()
