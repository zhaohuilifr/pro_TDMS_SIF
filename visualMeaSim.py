# -*- coding: utf-8 -*-

import glob as glob
import os
import matplotlib.pyplot as plt
# import imageio
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.cm as cm
from scipy import stats
'''
比较LIF和用455+-25 nm激发的SIF的相关性
'''

# plot fonts
plt.rcParams.update({'font.size': 14, 'font.family': 'Arial'})
unitf = '\n'+'(mW m$^{-2}$ nm$^{-1}$ sr$^{-1}$)'
unitgpp = '\n'+'(μmol CO$_{2}$ m$^{-2}$ s$^{-1}$)'
unitpar = '\n'+'(μmol photon m$^{-2}$ s$^{-1}$)'
unit_phiF = ' '+'(-)'

# %% functions
def get_r2(x, y):
    temp = pd.concat([x, y], axis=1)
    temp = temp.dropna()
    x = temp[x.name]
    y = temp[y.name]
    x_reshaped = x.values.reshape(-1, 1)
    if len(x_reshaped) < 2:
        return np.nan, np.nan
    model = LinearRegression().fit(x_reshaped, y)
    y_pred = model.predict(x_reshaped)
    # r2 = r2_score(y, y_pred)
    # # 计算p值
    # p_value = np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
    r_value, p_value = stats.pearsonr(x, y)[0], stats.pearsonr(x, y)[1]
    

    return r_value**2, p_value

def read_csv_with_fallback(path, **kwargs):
    # Some source files are saved in Windows legacy encodings (e.g., cp1252).
    for enc in ('utf-8', 'cp1252', 'latin-1'):
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, **kwargs)

def plot_x_y_dense(ax, df, xname, yname, xlabel, ylabel, xmax, ymax, c):
    # 保存alpha_ql, half saturation PAR, r2, rmse
    res = []
    # 绘制SIF模拟与观测的散点图
    if c.name == 'DOY_mea':
        norm = plt.Normalize(150, 250)
    elif c.name == 'VPD':
        norm = plt.Normalize(0, 5)
    elif c.name == 'PARdiffuse_fraction_top':
        norm = plt.Normalize(0, 1)
    elif c.name == 'JaLAI':
        norm = plt.Normalize(0, 200)
    cb = ax.scatter(df[xname], df[yname], c=c, cmap='jet', norm=norm)  # , alpha=0.5
    # if xmax == ymax:
    #     ax.plot([0, ymax], [0, ymax], color='k', linestyle='--', label='1:1 line')
    # 绘制线性拟合线
    if len(df) > 2:
        model = LinearRegression()
        X = df[xname].values.reshape(-1, 1)
        y = df[yname].values
        # 去掉nan值
        valid_idx = ~np.isnan(X.flatten()) & ~np.isnan(y)
        X = X[valid_idx].reshape(-1, 1)
        y = y[valid_idx]
        model.fit(X, y)
        y_pred = model.predict(X)
        # 计算rmse
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))
        # linelabel = f'R²={r2_score(y, y_pred):.2f}'+'\n'+f'RMSE={rmse:.2f}'
        linelabel = f'R²={r2_score(y, y_pred):.2f}'
        ax.plot(X.flatten(), y_pred, color='r', linestyle='--', label=linelabel)
        ax.legend(loc=4)
        res.append([r2_score(y, y_pred), rmse])

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    # if xname == 'Fs':
    #     ax.set_xlim([0.0018,xmax])
    #     # ax.set_ylim([0.0005,ymax])
    # else:
    #     ax.set_xlim([0,xmax])
    #     # ax.set_ylim([0,ymax])
    ax.grid(True)
    # fig.suptitle(f'Simulation vs Measurement (alpha_qL={int(alpha_ql)/100.0}, {tname})', fontsize=24, fontweight='bold')
    # fig.savefig(os.path.join(path_save_figs, f'Sim_vs_mea_alphaql{alpha_ql}_sensitivity_{tname}.png'), dpi=300)
    return res, cb

def scatter_seasonal_trend(ax, df, xname, yname1, yname2, ma, ymax1, ymax2, tname, ylabel1, ylabel2, c1, c2):
    if xname == 'DOY_mea':
        ax.scatter(df[xname], df[yname1], marker = ma, c=c1, alpha=0.5)
        axs1 = ax.twinx()
        axs1.scatter(df[xname], df[yname2], marker = ma, c=c2, alpha=0.5)
    elif xname == 'DOY_int':
        ax.plot(df[xname], df[yname1], marker = ma, c=c1, alpha=0.5)
        axs1 = ax.twinx()
        axs1.plot(df[xname], df[yname2], marker = ma, c=c2, alpha=0.5)
    ax.set_ylim([0,ymax1])
    axs1.set_ylim([0,ymax2])
    ax.set_title(tname)
    ax.set_ylabel(ylabel1, color=c1)
    axs1.set_ylabel(ylabel2, color=c2)
    ax.grid(True)
    ax.set_xlabel('DOY')

# %% visualization analysis
# %% GPP模拟与观测和Christophe结果的对比
# datapath = r'E:\Datahub\Barbeau\Data_matched'
# savepath_figs = datapath + r'\figs'
# if not os.path.exists(savepath_figs):
#     os.makedirs(savepath_figs)
# data = read_csv_with_fallback(
#     os.path.join(datapath, 'SIF_GPP_matching_Res_67_729_new.csv'),
#     index_col=False
# )
# # idx_gpp_filter = (data['qc_co2'] == 2) | (data['GPP'] <= 0)
# idx_gpp_filter = (data['GPP'] <= 0)
# data.loc[idx_gpp_filter, 'GPP'] = np.nan

# data['DOY_int'] = data['DOY'].astype(int)
# data = data.loc[:, ['hh','DOY','DOY_int', 'GPP', 'PBhLAI', 'SIF_3FLD','SIF_SFM_lin_a','SIFcanopy_760nm','SIFtotal_760nm','Fs']]
# data_daily = data.groupby('DOY_int').mean().reset_index()

# fig= plt.figure(figsize=(24, 12))
# # fig.subplots_adjust(top=0.925, wspace=0.3, hspace = 0.3)
# gs = fig.add_gridspec(3, 2)
# ax0 = fig.add_subplot(gs[0, 0])
# ax1 = fig.add_subplot(gs[0, 1])
# ax_mid = fig.add_subplot(gs[1, :])
# ax_bottom = fig.add_subplot(gs[2, :])  

# ax0.scatter(data['GPP'], data['PBhLAI'], color='b', marker='.')
# ax0.set_xlabel('Measured GPP'+ unitgpp)
# ax0.set_ylabel('Simulated GPP'+ unitgpp)
# ax0.set_xlim([0, 50])
# ax0.set_ylim([0, 50])
# ax0.plot([0, 50], [0, 50], color='k', linestyle='--', label='1:1 line')
# r2, p_value = get_r2(data['GPP'], data['PBhLAI'])
# ax0.text(0.05, 0.95, f'R²={r2:.2f}\np={p_value:.4f}', transform=ax0.transAxes, fontsize=12, verticalalignment='top')

# ax1.scatter(data_daily['GPP'], data_daily['PBhLAI'], color='b', marker='o')
# ax1.set_xlabel('Measured GPP'+ unitgpp)
# ax1.set_ylabel('Simulated GPP'+ unitgpp)
# ax1.set_xlim([0, 50])
# ax1.set_ylim([0, 50])
# ax1.plot([0, 50], [0, 50], color='k', linestyle='--', label='1:1 line')
# r2, p_value = get_r2(data_daily['GPP'], data_daily['PBhLAI'])
# ax1.text(0.05, 0.95, f'R²={r2:.2f}\np={p_value:.4f}', transform=ax1.transAxes, fontsize=12, verticalalignment='top')


# ax_mid.plot(data['DOY'], data['GPP'], color='b', marker='.', label='Measured GPP(Half-hourly)')
# ax_mid.plot(data['DOY'], data['PBhLAI'], color='g', marker='.', label='Simulated GPP(Half-hourly)')
# ax_mid.set_xlim([140, 260])
# ax_mid.set_xlabel('DOY')
# ax_mid.set_ylabel('GPP (μmol CO$_{2}$ m$^{-2}$ s$^{-1}$)')
# ax_mid.legend()

# ax_bottom.plot(data_daily['DOY_int']+0.5, data_daily['GPP'], color='b', marker='o', label='Measured GPP(Daily)')
# ax_bottom.plot(data_daily['DOY_int']+0.5, data_daily['PBhLAI'], color='g', marker='o', label='Simulated GPP(Daily)')
# ax_bottom.set_xlim([140, 260])
# ax_bottom.set_xlabel('DOY')
# ax_bottom.set_ylabel('GPP (μmol CO$_{2}$ m$^{-2}$ s$^{-1}$)')
# ax_bottom.legend()

# # %% SIFplot
# fig, ax = plt.subplots(2, 4, figsize=(24, 12))
# fig.subplots_adjust(left = 0.05, wspace=0.4, hspace=0.35)
# idx_sif_filter = (data['hh'] >= 7.5) & (data['hh'] <= 18.5)
# data = data.loc[idx_sif_filter, :]

# ax[0,0].scatter(data['GPP'], data['SIF_3FLD'], color='b', marker='.')
# ax[0,0].set_xlabel('Measured GPP'+ unitgpp)
# ax[0,0].set_ylabel('Measured SIF'+ unitf)
# r2, p_value = get_r2(data['GPP'], data['SIF_3FLD'])
# ax[0,0].text(0.05, 0.95, f'R²={r2:.2f}\np={p_value:.4f}', transform=ax[0,0].transAxes, fontsize=12, verticalalignment='top')
# ax[0,1].scatter(data['GPP'], data['SIF_SFM_lin_a'], color='b', marker='.')
# ax[0,1].set_xlabel('Measured GPP'+ unitgpp)
# ax[0,1].set_ylabel('Measured SIF'+ unitf)
# r2, p_value = get_r2(data['GPP'], data['SIF_SFM_lin_a'])
# ax[0,1].text(0.05, 0.95, f'R²={r2:.2f}\np={p_value:.4f}', transform=ax[0,1].transAxes, fontsize=12, verticalalignment='top')
# ax[0,2].scatter(data['SIF_3FLD'], data['SIFcanopy_760nm'], color='b', marker='.')
# ax[0,2].set_xlabel('Measured SIF'+ unitf)
# ax[0,2].set_ylabel('Simulated SIF(canopy)'+ unitf)
# r2, p_value = get_r2(data['SIF_3FLD'], data['SIFcanopy_760nm'])
# ax[0,2].text(0.05, 0.95, f'R²={r2:.2f}\np={p_value:.4f}', transform=ax[0,2].transAxes, fontsize=12, verticalalignment='top')
# ax[0,3].scatter(data['SIF_SFM_lin_a'], data['SIFcanopy_760nm'], color='b', marker='.')
# ax[0,3].set_xlabel('Measured SIF'+ unitf)
# ax[0,3].set_ylabel('Simulated SIF(total)'+ unitf)
# r2, p_value = get_r2(data['SIF_SFM_lin_a'], data['SIFcanopy_760nm'])
# ax[0,3].text(0.05, 0.95, f'R²={r2:.2f}\np={p_value:.4f}', transform=ax[0,3].transAxes, fontsize=12, verticalalignment='top')

# ax[1,0].scatter(data_daily['GPP'], data_daily['SIF_3FLD'], color='b', marker='o')
# ax[1,0].set_xlabel('Measured GPP'+ unitgpp)
# ax[1,0].set_ylabel('Measured SIF'+ unitf)
# r2, p_value = get_r2(data_daily['GPP'], data_daily['SIF_3FLD'])
# ax[1,0].text(0.05, 0.95, f'R²={r2:.2f}\np={p_value:.4f}', transform=ax[1,0].transAxes, fontsize=12, verticalalignment='top')
# ax[1,1].scatter(data_daily['GPP'], data_daily['SIF_SFM_lin_a'], color='b', marker='o')
# ax[1,1].set_xlabel('Measured GPP'+ unitgpp)
# ax[1,1].set_ylabel('Measured SIF'+ unitf)
# r2, p_value = get_r2(data_daily['GPP'], data_daily['SIF_SFM_lin_a'])
# ax[1,1].text(0.05, 0.95, f'R²={r2:.2f}\np={p_value:.4f}', transform=ax[1,1].transAxes, fontsize=12, verticalalignment='top')
# ax[1,2].scatter(data_daily['SIF_3FLD'], data_daily['SIFcanopy_760nm'], color='b', marker='o')
# ax[1,2].set_xlabel('Measured SIF'+ unitf)
# ax[1,2].set_ylabel('Simulated SIF(canopy)'+ unitf)
# r2, p_value = get_r2(data_daily['SIF_3FLD'], data_daily['SIFcanopy_760nm'])
# ax[1,2].text(0.05, 0.95, f'R²={r2:.2f}\np={p_value:.4f}', transform=ax[1,2].transAxes, fontsize=12, verticalalignment='top')
# ax[1,3].scatter(data_daily['SIF_SFM_lin_a'], data_daily['SIFtotal_760nm'], color='b', marker='o')
# ax[1,3].set_xlabel('Measured SIF'+ unitf)
# ax[1,3].set_ylabel('Simulated SIF(total)'+ unitf)
# r2, p_value = get_r2(data_daily['SIF_SFM_lin_a'], data_daily['SIFtotal_760nm'])
# ax[1,3].text(0.05, 0.95, f'R²={r2:.2f}\np={p_value:.4f}', transform=ax[1,3].transAxes, fontsize=12, verticalalignment='top')
# # fig.savefig(os.path.join(savepath_figs, f'SIF_GPP_comparison.png'), dpi=300)


# fig, ax = plt.subplots(3, 1, figsize=(24, 12), sharex=True)
# ax[0].plot(data['DOY'], data['GPP'], color='b', marker='.', label='Measured GPP(Half-hourly)')
# ax[0].set_xlim([1, 365])
# ax[0].legend()
# ax[1].plot(data['DOY'], data['SIF_3FLD'], color='b', marker='.', label='Measured SIF(Half-hourly)')
# ax[1].set_xlim([1, 365])
# ax[1].legend()
# ax[2].plot(data['DOY'], data['Fs'], color='b', marker='.', label='Measured Fs(Half-hourly)')
# ax[2].set_xlim([1, 365])
# ax[2].legend()


datapath = r'E:\Datahub\Barbeau\Data_matched'
savepath_figs = datapath + r'\figs'
if not os.path.exists(savepath_figs):
    os.makedirs(savepath_figs)
data = pd.read_excel(
    os.path.join(datapath, 'Barbeau_2025_matched.xlsx'),
    index_col=False
)
# idx_gpp_filter = (data['qc_co2'] == 2) | (data['GPP'] <= 0)
idx_gpp_filter = (data['GPP'] <= 0)
data.loc[idx_gpp_filter, 'GPP'] = np.nan

data['DOY_int'] = data['DOY'].astype(int)
data = data.loc[:, ['hh','DOY','DOY_int', 'GPP', 'SIF_3FLD','SIF_SFM_lin_a','Fs']]
data_daily = data.groupby('DOY_int').mean().reset_index()


# %% SIFplot
fig, ax = plt.subplots(2, 4, figsize=(24, 12))
fig.subplots_adjust(left = 0.05, wspace=0.4, hspace=0.35)
idx_sif_filter = (data['hh'] >= 7.5) & (data['hh'] <= 18.5)
data = data.loc[idx_sif_filter, :]

ax[0,0].scatter(data['GPP'], data['SIF_3FLD'], color='b', marker='.')
ax[0,0].set_xlabel('Measured GPP'+ unitgpp)
ax[0,0].set_ylabel('Measured SIF'+ unitf)
r2, p_value = get_r2(data['GPP'], data['SIF_3FLD'])
ax[0,0].text(0.05, 0.95, f'R²={r2:.2f}\np={p_value:.4f}', transform=ax[0,0].transAxes, fontsize=12, verticalalignment='top')
ax[0,1].scatter(data['GPP'], data['SIF_SFM_lin_a'], color='b', marker='.')
ax[0,1].set_xlabel('Measured GPP'+ unitgpp)
ax[0,1].set_ylabel('Measured SIF'+ unitf)
r2, p_value = get_r2(data['GPP'], data['SIF_SFM_lin_a'])
ax[0,1].text(0.05, 0.95, f'R²={r2:.2f}\np={p_value:.4f}', transform=ax[0,1].transAxes, fontsize=12, verticalalignment='top')


ax[1,0].scatter(data_daily['GPP'], data_daily['SIF_3FLD'], color='b', marker='o')
ax[1,0].set_xlabel('Measured GPP'+ unitgpp)
ax[1,0].set_ylabel('Measured SIF'+ unitf)
r2, p_value = get_r2(data_daily['GPP'], data_daily['SIF_3FLD'])
ax[1,0].text(0.05, 0.95, f'R²={r2:.2f}\np={p_value:.4f}', transform=ax[1,0].transAxes, fontsize=12, verticalalignment='top')
ax[1,1].scatter(data_daily['GPP'], data_daily['SIF_SFM_lin_a'], color='b', marker='o')
ax[1,1].set_xlabel('Measured GPP'+ unitgpp)
ax[1,1].set_ylabel('Measured SIF'+ unitf)
r2, p_value = get_r2(data_daily['GPP'], data_daily['SIF_SFM_lin_a'])
ax[1,1].text(0.05, 0.95, f'R²={r2:.2f}\np={p_value:.4f}', transform=ax[1,1].transAxes, fontsize=12, verticalalignment='top')



fig, ax = plt.subplots(3, 1, figsize=(24, 12), sharex=True)
ax[0].plot(data['DOY'], data['GPP'], color='b', marker='.', label='Measured GPP(Half-hourly)')
ax[0].set_xlim([1, 365])
ax[0].legend()
ax[1].plot(data['DOY'], data['SIF_3FLD'], color='b', marker='.', label='Measured SIF(Half-hourly)')
ax[1].set_xlim([1, 365])
ax[1].legend()
ax[2].plot(data['DOY'], data['Fs'], color='b', marker='.', label='Measured Fs(Half-hourly)')
ax[2].set_xlim([1, 365])
ax[2].legend()

plt.show()