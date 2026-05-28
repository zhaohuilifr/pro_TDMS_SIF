# -*- coding: utf-8 -*-

import os
import glob as glob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.cm as cm
import statsmodels.api as sm

# %% plot fonts
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['axes.labelsize'] = 6.5   # 坐标轴标签字号 (X/Y Label)
plt.rcParams['xtick.labelsize'] = 5.5  # X轴刻度字号
plt.rcParams['ytick.labelsize'] = 5.5  # Y轴刻度字号
plt.rcParams['legend.fontsize'] = 5   # 图例字号
plt.rcParams['axes.titlesize'] = 7   # 子图标题字号 (如果需要)
plt.rcParams['axes.linewidth'] = 0.7  # 边框粗细
plt.rcParams['xtick.direction'] = 'in' # 刻度线朝内，学术图表经典风格
plt.rcParams['ytick.direction'] = 'in'
font_subfigs_index = {'fontsize': 10, 'fontweight': 'bold'}

unitf = '\n'+'(mW m$^{-2}$ nm$^{-1}$ sr$^{-1}$)'
unitgpp = '\n'+'(μmol CO$_{2}$ m$^{-2}$ s$^{-1}$)'
unitpar = '\n'+'(μmol photon m$^{-2}$ s$^{-1}$)'
unit_phiF = ' '+'(-)'
unit_vis = ' '+'(-)'


# %% functions
def create_scaled_figure(nrows, ncols, base_ax_size=(3, 1.5), dpi=300):
    """根据子图行列数，自动计算画布大小并等比例缩放字号"""
    # 1. 根据每个子图的理想大小，自动计算总画布的宽高
    fig_width = ncols * base_ax_size[0]
    fig_height = nrows * base_ax_size[1]

    # 2. 核心算法：以单子图画布周长（17.6英寸）为基准，计算当前的放大系数
    # 这样可以确保无论你画 1x1, 2x2 还是 4x4，字体的相对视觉大小完全恒定
    base_perimeter = 2 * (base_ax_size[0] + base_ax_size[1])
    current_perimeter = 2 * (fig_width + fig_height)

    # 引入非线性微调，防止多子图时字号无限膨胀
    scale_factor = (current_perimeter / base_perimeter) ** 0.2

    # 3. 基于基准字号，等比例映射出当前画布的最佳字号
    font_sizes = {
        'axes.labelsize': int(6.5 * scale_factor),
        'xtick.labelsize': int(5.5 * scale_factor),
        'ytick.labelsize': int(5.5 * scale_factor),
        'legend.fontsize': int(5 * scale_factor),
        'axes.titlesize': int(7 * scale_factor),
    }

    # 4. 使用全局参数临时更新并创建画布
    plt.rcParams.update(font_sizes)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(fig_width, fig_height), dpi=dpi
    )
    return fig, axes

def get_r2_pvalue(x, y):
    temp = pd.concat([x, y], axis=1)
    temp = temp.dropna()
    if len(temp) < 2:
        return np.nan, np.nan
    x_clean = temp[x.name]
    y_clean = temp[y.name]
    # statsmodels 默认不包含截距（intercept），必须手动加上常数项
    X = sm.add_constant(x_clean)
    # 拟合普通的最小二乘回归 (OLS)
    model = sm.OLS(y_clean, X).fit()
    # 提取 R2
    r2 = model.rsquared
    # model.pvalues 会返回一个 Series，包含 [截距的p值, 斜率的p值]
    # 我们论文中通常关心的“回归 p 值”是指自变量（斜率）的 p 值，即索引为 1 的值
    p_value = model.pvalues.iloc[1]

    return r2, p_value

# %% some global settings
root = r'E:\Datahub\Barbeau\Data_matched'
savepath_figs = os.path.join(root, 'figs/Figures')
if not os.path.exists(savepath_figs):
    os.makedirs(savepath_figs)

# %% Figure 1. Measured APAR/GPP and simulated APAR/GPP from CASTANEA
# filename_2022 = 'Barbeau_2022_matched_CASTANEA_validation.xlsx'
filename_2022 = 'Barbeau_2022_matched_CASTANEA_fracdiff_validation.xlsx'
filename_2025 = 'Barbeau_2025_matched_CASTANEA.xlsx'
df_2022 = pd.read_excel(os.path.join(root, filename_2022))
df_2025 = pd.read_excel(os.path.join(root, filename_2025))
df_2022['APARmeas'] = df_2022['PAR (µmol/m2/s)'] * df_2022['fPAR']
df_2022['APARsimu'] = df_2022['PARdiraLAI'] + df_2022['PARdifaLAI']

df_2022 = df_2022.rename(columns={
    'GPP': 'GPPmeas',
    'PBhLAI': 'GPPsimu'
})
# df_2025['APARmeas'] = df_2025['PAR (µmol/m2/s)'] * df_2025['fPAR']
# df_2025['APARsimu'] = df_2025['PARdiraLAI'] + df_2025['PARdifaLAI']
df_2025 = df_2025.rename(columns={
    'GPP': 'GPPmeas',
    'PBhLAI': 'GPPsimu'
})
# remove outliers
df_2022.loc[df_2022['GPPmeas'] < 0, 'GPPmeas'] = np.nan
df_2025.loc[df_2025['GPPmeas'] < 0, 'GPPmeas'] = np.nan
df_2022.loc[df_2022['SIF_3FLD'] < 0, 'SIF_3FLD'] = np.nan
df_2025.loc[df_2025['SIF_3FLD'] < 0, 'SIF_3FLD'] = np.nan
df_2022.loc[df_2022['Fs'] < 0.15, 'Fs'] = np.nan
df_2025.loc[df_2025['Fs'] < 0.15, 'Fs'] = np.nan

idx_2022 = df_2022['hour_UTC'].between(8, 17)
idx_2025 = df_2025['hour_UTC'].between(8, 17)

fig, ax = plt.subplots(2, 2, figsize=(16, 10))
fig, ax = create_scaled_figure(nrows=2, ncols=2, base_ax_size=(3, 2), dpi=300)
fig.subplots_adjust(wspace = 0.3, hspace = 0.33)
ax[0, 0].scatter(df_2022.loc[idx_2022, 'APARmeas'], df_2022.loc[idx_2022, 'APARsimu'], s=6, edgecolor='none', facecolor='k', alpha=0.2, zorder=2)
r2, p_value = get_r2_pvalue(df_2022.loc[idx_2022, 'APARmeas'], df_2022.loc[idx_2022, 'APARsimu'])
ax[0, 0].set_xlabel('Measured APAR' + unitpar)
ax[0, 0].set_ylabel('Simulated APAR' + unitpar)
ax[0, 0].plot([0, 2100], [0, 2100], color='r', linestyle='--', lw=1, label = f'\nR²={r2:.2f}, p={p_value:.2f}')
ax[0, 0].set_xlim(0, 2100)
ax[0, 0].set_ylim(0, 2100)
ax[0, 0].legend(frameon=False)
ax[0, 1].scatter(df_2022.loc[idx_2022, 'GPPmeas'], df_2022.loc[idx_2022, 'GPPsimu'], s=6, edgecolor='none', facecolor='k', alpha=0.2, zorder=2)
r2, p_value = get_r2_pvalue(df_2022.loc[idx_2022, 'GPPmeas'], df_2022.loc[idx_2022, 'GPPsimu'])
ax[0, 1].set_xlabel('Measured GPP' + unitgpp)
ax[0, 1].set_ylabel('Simulated GPP' + unitgpp)
ax[0, 1].plot([0, 60], [0, 60], color='r', linestyle='--', lw=1, label = f'\nR²={r2:.2f}, p={p_value:.2f}')
ax[0, 1].set_xlim(0, 60)
ax[0, 1].set_ylim(0, 60)
ax[0, 1].legend(frameon=False)


ax[1, 1].scatter(df_2025.loc[idx_2025, 'GPPmeas'], df_2025.loc[idx_2025, 'GPPsimu'], s=6, edgecolor='none', facecolor='k', alpha=0.2, zorder=2)
r2, p_value = get_r2_pvalue(df_2025.loc[idx_2025, 'GPPmeas'], df_2025.loc[idx_2025, 'GPPsimu'])
ax[1, 1].set_xlabel('Measured GPP' + unitgpp)
ax[1, 1].set_ylabel('Simulated GPP' + unitgpp)
ax[1, 1].plot([0, 60], [0, 60], color='r', linestyle='--', lw=1, label = f'\nR²={r2:.2f}, p={p_value:.2f}')
ax[1, 1].set_xlim(0, 60)
ax[1, 1].set_ylim(0, 60)
ax[1, 1].legend(frameon=False)

# add (a), (b), (c), (d) labels
ax[0, 0].text(-0.15, 1.05, '(a)', transform=ax[0, 0].transAxes, **font_subfigs_index)
ax[0, 1].text(-0.15, 1.05, '(b)', transform=ax[0, 1].transAxes, **font_subfigs_index)
ax[1, 0].text(-0.15, 1.05, '(c)', transform=ax[1, 0].transAxes, **font_subfigs_index)
ax[1, 1].text(-0.15, 1.05, '(d)', transform=ax[1, 1].transAxes, **font_subfigs_index)

fig.savefig(os.path.join(savepath_figs, 'Fig1_APAR_GPP_validation.pdf'), dpi=500, bbox_inches='tight')



fig, ax = create_scaled_figure(nrows=2, ncols=2, base_ax_size=(3, 2), dpi=300)
ax[0,0].scatter(df_2022.loc[idx_2022, 'SIF_3FLD'], df_2022.loc[idx_2022, 'SIFcanopy_760nm']*8, s=6, edgecolor='none', facecolor='k', alpha=0.2, zorder=2)
r2, p_value = get_r2_pvalue(df_2022.loc[idx_2022, 'SIF_3FLD'], df_2022.loc[idx_2022, 'SIFcanopy_760nm']*8)
ax[0,0].set_xlabel('Measured SIF@760nm' + unitf)
ax[0,0].set_ylabel('Simulated SIF@760nm' + unitf)
ax[0,0].plot([0, 1.2], [0, 1.2], color='r', linestyle='--', lw=1, label =  f'\nR²={r2:.2f}, p={p_value:.2f}')
ax[0,0].set_xlim(0, 1.2)
ax[0,0].set_ylim(0, 1.2)
ax[0,0].legend(frameon=False)
ax[0,1].scatter(df_2022.loc[idx_2022, 'Fs'], df_2022.loc[idx_2022, 'SIFPSIILAI_yield_top'], s=6, edgecolor='none', facecolor='k', alpha=0.2, zorder=2)
r2, p_value = get_r2_pvalue(df_2022.loc[idx_2022, 'Fs'], df_2022.loc[idx_2022, 'SIFPSIILAI_yield_top'])
ax[0,1].set_xlabel('Measured $\phi_F$ (Fs)' + unit_phiF)
ax[0,1].set_ylabel('Simulated $\phi_F$' + unit_phiF)
ax[0,1].plot([0, 0.5], [0, 0.05], color='r', linestyle='--', lw=1, label =f'\nR²={r2:.2f}, p={p_value:.2f}')
ax[0,1].set_xlim(0, 0.5)
ax[0,1].set_ylim(0, 0.05)
ax[0,1].legend(frameon=False)

ax[1,0].scatter(df_2025.loc[idx_2025, 'SIF_3FLD'], df_2025.loc[idx_2025, 'SIFcanopy_760nm']*8, s=6, edgecolor='none', facecolor='k', alpha=0.2, zorder=2)
r2, p_value = get_r2_pvalue(df_2025.loc[idx_2025, 'SIF_3FLD'], df_2025.loc[idx_2025, 'SIFcanopy_760nm']*8)
ax[1,0].set_xlabel('Measured SIF@760nm' + unitf)
ax[1,0].set_ylabel('Simulated SIF@760nm' + unitf)
ax[1,0].plot([0, 1.2], [0, 1.2], color='r', linestyle='--', lw=1, label =f'\nR²={r2:.2f}, p={p_value:.2f}')
ax[1,0].set_xlim(0, 1.2)
ax[1,0].set_ylim(0, 1.2)
ax[1,0].legend(frameon=False)
ax[1,1].scatter(df_2025.loc[idx_2025, 'Fs'], df_2025.loc[idx_2025, 'SIFPSIILAI_yield_top'], s=6, edgecolor='none', facecolor='k', alpha=0.2, zorder=2)
r2, p_value = get_r2_pvalue(df_2025.loc[idx_2025, 'Fs'], df_2025.loc[idx_2025, 'SIFPSIILAI_yield_top'])
ax[1,1].set_xlabel('Measured $\phi_F$ (Fs)' + unit_phiF)
ax[1,1].set_ylabel('Simulated $\phi_F$' + unit_phiF)
ax[1,1].plot([0, 0.5], [0, 0.015], color='r', linestyle='--', lw=1, label =f'\nR²={r2:.2f}, p={p_value:.2f}')
ax[1,1].set_xlim(0, 0.5)
ax[1,1].set_ylim(0, 0.015)
ax[1,1].legend(frameon=False)
# add (a), (b), (c), (d) labels
ax[0, 0].text(-0.15, 1.05, '(a)', transform=ax[0, 0].transAxes, **font_subfigs_index)
ax[0, 1].text(-0.15, 1.05, '(b)', transform=ax[0, 1].transAxes, **font_subfigs_index)
ax[1, 0].text(-0.15, 1.05, '(c)', transform=ax[1, 0].transAxes, **font_subfigs_index)
ax[1, 1].text(-0.15, 1.05, '(d)', transform=ax[1, 1].transAxes, **font_subfigs_index)
fig.savefig(os.path.join(savepath_figs, 'Fig2_SIF_Fs_validation.pdf'), dpi=500, bbox_inches='tight')

# plt.show()