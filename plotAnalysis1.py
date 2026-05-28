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


# plot fonts
plt.rcParams.update({'font.size': 14, 'font.family': 'Arial'})
unitf = '\n'+'(mW m$^{-2}$ nm$^{-1}$ sr$^{-1}$)'
unitgpp = '\n'+'(μmol CO$_{2}$ m$^{-2}$ s$^{-1}$)'
unitpar = '\n'+'(μmol photon m$^{-2}$ s$^{-1}$)'
unit_phiF = ' '+'(-)'

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


if __name__ == "__main__":
    # folders = ['Res_67_729_AEF_m1','Res_67_729_AEF_m2','Res_67_729_Allband_AEF_m1','Res_67_729_Allband_AEF_m2'] #'Res_67_729_150-250',
    folders = ['Res_67_300','Res_67_729_Vcmax_change','Res_67_729_Vcmax_fixed','Res_67_729_qL_fixed','Res_67_1500'] #['Res_67_300', 'Res_67_400', 'Res_67_500', 'Res_67_600'] #Res_67_1000_AEF_m2
    for folder in folders:
        path_root = r'D:\Projet ifx Castanea\result_Barbeau2024\fluorescence\Res_LIF_analysis' + os.sep + folder
        path_sif_layers = os.path.join(path_root, 'SIF_layers')
        path_sif_canopy = os.path.join(path_root, 'SIF_canopy')
        path_cas_output = os.path.join(path_root, 'Cas_Outputs')
        path_save_figs = os.path.join(path_root, 'Figs')
        path_save_file = os.path.join(path_root, 'Analysis')
        if not os.path.exists(path_save_figs):
            os.makedirs(path_save_figs)
        if not os.path.exists(path_save_file):
            os.makedirs(path_save_file)
        print(f'Paths set for {folder}.')

        # %% SIFyield_top vs Fs at different PARdiffuse_fraction
        data = pd.read_csv(os.path.join(path_save_file, f'SIF_GPP_matching_{path_root.split(os.sep)[-1]}.csv'), index_col=False)
        data['SIFcanopy_760nm'] = data['SIFcanopy_760nm']/0.2/0.1 # convert to mW m-2 nm-1 sr-1
        data['APARtotal'] = data['PARdiraLAI'] + data['PARdifaLAI']
        data['hour'] = (data['DOY_mea'] - data['DOY_mea'].astype(int))
        # data = data[(data['hour'] >= (9.0/24.0)) & (data['hour'] <= (16.0/24.0))].reset_index(drop=True)
        data.loc[data['SIFPSIILAI_yield']<0,'SIFPSIILAI_yield'] = np.nan # outlier removal
        data.loc[data['SIFPSIILAI_yield_top']<0,'SIFPSIILAI_yield_top'] = np.nan # outlier removal
        data.loc[data['SIFPSIILAI_yield_top5']<0,'SIFPSIILAI_yield_top5'] = np.nan # outlier removal
        data['DOY_int'] = data['DOY_mea'].astype(int)
        data_daily = data.groupby('DOY_int').mean().reset_index()
        data_daily['DOY_mea'] = data_daily['DOY_int']

        # %% VPD> 3的数据
        # data = data.loc[(data['VPD'] > 3.0)].reset_index(drop=True)
        data = data.loc[(data['VPD'] <= 3.0)].reset_index(drop=True)
        # data_daily = data_daily.loc[(data_daily['VPD'] > 3.0)].reset_index(drop=True)
        fignames_prefix = ['Half-hourly_', 'Daily_']
        for k, df in enumerate([data]): #, data_daily
            fig, axs = plt.subplots(1, 3, figsize=(14,6))
            fig.subplots_adjust(top=0.925, wspace = 0.45, hspace = 0.3, bottom=0.17,right = 0.86)

            res, cb = plot_x_y_dense(ax = axs[0], df = df, xname='PBhLAI', yname='GPP', xlabel='Simulated GPP'+ unitgpp, 
                                        ylabel='Measured GPP'+ unitgpp, xmax = 2.5, ymax = 50, c = df['VPD'])
            res, cb = plot_x_y_dense(ax = axs[1], df = df, xname='SIFcanopy_760nm', yname='SIF_mea', xlabel='Simulated SIF@760nm'+ unitf, 
                                    ylabel='Measured SIF@760nm'+ unitf, xmax = 2.5, ymax = 2.5, c = df['VPD'])
            plot_x_y_dense(ax = axs[2], df = df, xname='SIFPSIILAI_yield_top', yname='Fs', xlabel='$\phi_F (top)$' + unit_phiF+ '(simulated)', 
                                ylabel='Fs'+ unit_phiF, xmax=0.0005, ymax=0.45, c = df['VPD'])
            # 添加colorbar
            cbar = fig.colorbar(cb, ax=axs, orientation='vertical', fraction=0.02, pad=0.01)
            cbar.set_label('VPD', rotation=90, labelpad=15)
            cbar.set_ticklabels([0, 1, 2, 3, 4, 5])
        # plt.show()
            # fig.tight_layout()
            # fig.savefig(os.path.join(path_save_figs, fignames_prefix[k] + f'SIF_GPP_comparison_{path_root.split(os.sep)[-1]}_VPDcolor_highVPD.png'), dpi=300)
            fig.savefig(os.path.join(path_save_figs, fignames_prefix[k] + f'SIF_GPP_comparison_{path_root.split(os.sep)[-1]}_VPDcolor_lowVPD.png'), dpi=300)
        # %% 找到每天7：00- 18：00的半小时数据中 VPD>3 比例大于 0.5 的日期
        # # data = data[(data['hour'] >= (7.0/24.0)) & (data['hour'] <= (18.0/24.0))].reset_index(drop=True)
        # vpd_threshold = 3.0
        # vpd_proportion_threshold = 0.4
        # valid_dates = []
        # for doy in data_daily['DOY_int'].unique():
        #     day_data = data[data['DOY_int'] == doy]
        #     vpd_proportion = (day_data['VPD'] > vpd_threshold).mean()
        #     if vpd_proportion > vpd_proportion_threshold:
        #         # print((day_data['VPD'] > vpd_threshold))
        #         # print(vpd_proportion)
        #         valid_dates.append(doy.astype(int).item())
        # print(valid_dates)
        # %% 根据上述条件选择日期
        # valid_dates = [194, 199, 215, 222, 223, 224, 225]
        # # # 选择满足条件的日期的数据
        # data_valid = data[data['DOY_int'].isin(valid_dates)].reset_index(drop=True)
        # # # 绘制
        # fig, ax = plt.subplots(7, 7, figsize=(14, 12))
        # # fig.subplots_adjust(hspace=0.4, wspace=0.1)
        # for idoy, doy in enumerate(valid_dates):
        #     day_data = data_valid[data_valid['DOY_int'] == doy]
        #     ax[0, idoy].plot(day_data['hour']*24, day_data['APARtotal'], marker='o', linestyle='-', color='darkred')
        #     ax[1, idoy].plot(day_data['hour']*24, day_data['PBhLAI'], marker='o', linestyle='-', color='darkgreen')
        #     ax[2, idoy].plot(day_data['hour']*24, day_data['VPD'], marker='o', linestyle='-', color='tab:purple')
        #     ax[3, idoy].plot(day_data['hour']*24, day_data['SIFcanopy_760nm']/0.2/0.1, marker='o', linestyle='-', color='tab:blue')
        #     ax[4, idoy].plot(day_data['hour']*24, day_data['NPQLAI']/28, marker='o', linestyle='-', color='tab:orange')
        #     ax[5, idoy].plot(day_data['hour']*24, day_data['Fs'], marker='o', linestyle='-', color='tab:green')
        #     ax[6, idoy].plot(day_data['hour']*24, day_data['SIFPSIILAI_yield_top'], marker='o', linestyle='-', color='tab:red')

        #     ax[0, idoy].set_title(f'DOY {doy}')
        #     if idoy == 0:
        #         ax[0, idoy].set_ylabel('APAR' + unitpar)
        #         ax[1, idoy].set_ylabel('GPP' + unitgpp)
        #         ax[2, idoy].set_ylabel('VPD' + ' (kPa)')
        #         ax[3, idoy].set_ylabel('SIFcanopy@760nm' + unitf)
        #         ax[4, idoy].set_ylabel('NPQ' + unit_phiF)
        #         ax[5, idoy].set_ylabel('Fs' + unit_phiF)
        #         ax[6, idoy].set_ylabel('$\phi_F (top)$' + unit_phiF)
        #     else:
        #         ax[0, idoy].set_yticklabels([])
        #         ax[1, idoy].set_yticklabels([])
        #         ax[2, idoy].set_yticklabels([])
        #         ax[3, idoy].set_yticklabels([])
        #         ax[4, idoy].set_yticklabels([])
        #         ax[5, idoy].set_yticklabels([])
        #         ax[6, idoy].set_yticklabels([])
        #     for i in range(6):
        #         ax[i, idoy].set_xticklabels([])
        #     ax[6, idoy].set_xlabel('Hour of Day')
        #     ax[0, idoy].set_ylim([0, 2000])
        #     ax[1, idoy].set_ylim([0, 45])
        #     ax[2, idoy].set_ylim([0,5])
        #     # ax[3, idoy].set_ylim([0, 0.2])
        #     ax[3, idoy].set_ylim([0, 2])
        #     ax[4, idoy].set_ylim([0, 5])
        #     ax[5, idoy].set_ylim([0.1,0.45])
        #     ax[6, idoy].set_ylim([0, 0.012])
        #     for i in range(7):
        #         ax[i, idoy].grid()
        # fig.savefig(os.path.join(path_save_figs, f'VPD_SIF_NPQ_Fs_SIFyield_top_DOY_{doy}.png'), dpi=300, bbox_inches='tight')
        # # plt.show()



        # fig, ax = plt.subplots(2, 7, figsize=(14, 8))
        # fig.subplots_adjust(hspace=0.4, wspace=0.1)
        # cmap = cm.get_cmap('viridis')
        # for idoy, doy in enumerate(valid_dates):
        #     day_data = data_valid[data_valid['DOY_int'] == doy]
        #     # project day_data['hour']*24 to a color range for plotting
        #     color = day_data['hour']*24
        #     sc1 = ax[0, idoy].scatter(day_data['SIFcanopy_760nm']/0.2/0.1, day_data['SIF_mea'], c=color, cmap=cmap, edgecolor='k')
        #     sc2 = ax[1, idoy].scatter(day_data['SIFPSIILAI_yield_top'], day_data['Fs'], c=color, cmap=cmap, edgecolor='k')

        #     if idoy == 0:
        #         ax[0, idoy].set_ylabel('Measured SIF@760nm' + unitf)
        #         ax[1, idoy].set_ylabel('Fs' + unit_phiF)
        #     else:
        #         ax[0, idoy].set_yticklabels([])
        #         ax[1, idoy].set_yticklabels([])
        #     ax[0, idoy].set_title(f'DOY {doy}')
        #     # ax[0, idoy].set_xlabel('Simulated SIF@760nm' + unitf)
        #     # ax[1, idoy].set_xlabel('$\phi_F (top)$' + unit_phiF + '(simulated)')
        #     ax[0, idoy].set_xlim([0, 2])
        #     ax[1, idoy].set_xlim([0.003, 0.012])
        #     ax[0, idoy].set_ylim([0, 2.5])
        #     ax[1, idoy].set_ylim([0.1, 0.35])
        # # 所有子图共用一个colorbar
        # cbar = fig.colorbar(sc1, ax=ax, orientation='vertical', fraction=0.02, pad=0.01)
        # cbar.set_label('Hour of Day', rotation=90, labelpad=15)
        # # cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
        # # cbar.set_ticklabels(['0', '6', '12', '18', '24'])

        # # 所有子图公用一个xlabel和ylabel
        # fig.text(0.5, 0.48, 'Simulated SIF@760nm' + '(mW m$^{-2}$ nm$^{-1}$ sr$^{-1}$)', ha='center', fontsize=16)
        # fig.text(0.5, 0.02, '$\phi_F (top)$' + unit_phiF + '(simulated)', ha='center', fontsize=16)

        # # plt.xlabel('Simulated SIF@760nm' + unitf)
        # # plt.ylabel('Measured SIF@760nm' + unitf)
        # plt.savefig(os.path.join(path_save_figs, f'SIFyield_top_vs_Fs_VPD_DOY_all.png'), dpi=300)
        # # # plt.show()

        # %% 根据上述条件选择日期
        # valid_dates = [171, 202, 203, 217, 218, 230, 231]
        # # # 选择满足条件的日期的数据
        # data_valid = data[data['DOY_int'].isin(valid_dates)].reset_index(drop=True)
        # # # 绘制
        # fig, ax = plt.subplots(7, 7, figsize=(14, 12))
        # # fig.subplots_adjust(hspace=0.4, wspace=0.1)
        # for idoy, doy in enumerate(valid_dates):
        #     day_data = data_valid[data_valid['DOY_int'] == doy]
        #     ax[0, idoy].plot(day_data['hour']*24, day_data['APARtotal'], marker='o', linestyle='-', color='darkred')
        #     ax[1, idoy].plot(day_data['hour']*24, day_data['PBhLAI'], marker='o', linestyle='-', color='darkgreen')
        #     ax[2, idoy].plot(day_data['hour']*24, day_data['VPD'], marker='o', linestyle='-', color='tab:purple')
        #     ax[3, idoy].plot(day_data['hour']*24, day_data['SIFcanopy_760nm'], marker='o', linestyle='-', color='tab:blue')
        #     ax[4, idoy].plot(day_data['hour']*24, day_data['NPQLAI']/28, marker='o', linestyle='-', color='tab:orange')
        #     ax[5, idoy].plot(day_data['hour']*24, day_data['Fs'], marker='o', linestyle='-', color='tab:green')
        #     ax[6, idoy].plot(day_data['hour']*24, day_data['SIFPSIILAI_yield_top'], marker='o', linestyle='-', color='tab:red')

        #     ax[0, idoy].set_title(f'DOY {doy}')
        #     if idoy == 0:
        #         ax[0, idoy].set_ylabel('APAR' + unitpar)
        #         ax[1, idoy].set_ylabel('GPP' + unitgpp)
        #         ax[2, idoy].set_ylabel('VPD' + ' (kPa)')
        #         ax[3, idoy].set_ylabel('SIFcanopy@760nm' + unitf)
        #         ax[4, idoy].set_ylabel('NPQ' + unit_phiF)
        #         ax[5, idoy].set_ylabel('Fs' + unit_phiF)
        #         ax[6, idoy].set_ylabel('$\phi_F (top)$' + unit_phiF)
        #     else:
        #         ax[0, idoy].set_yticklabels([])
        #         ax[1, idoy].set_yticklabels([])
        #         ax[2, idoy].set_yticklabels([])
        #         ax[3, idoy].set_yticklabels([])
        #         ax[4, idoy].set_yticklabels([])
        #         ax[5, idoy].set_yticklabels([])
        #         ax[6, idoy].set_yticklabels([])
        #     for i in range(6):
        #         ax[i, idoy].set_xticklabels([])
        #     ax[6, idoy].set_xlabel('Hour of Day')
        #     ax[0, idoy].set_ylim([0, 2000])
        #     ax[1, idoy].set_ylim([0, 45])
        #     ax[2, idoy].set_ylim([0,5])
        #     # ax[3, idoy].set_ylim([0, 0.2])
        #     ax[3, idoy].set_ylim([0, 2])
        #     ax[4, idoy].set_ylim([0, 5])
        #     ax[5, idoy].set_ylim([0.1,0.45])
        #     ax[6, idoy].set_ylim([0, 0.012])
        #     for i in range(7):
        #         ax[i, idoy].grid()
        # fig.savefig(os.path.join(path_save_figs, f'VPD_SIF_NPQ_Fs_SIFyield_top_DOY_{doy}_lowVPD.png'), dpi=300, bbox_inches='tight')
        # # plt.show()



        # fig, ax = plt.subplots(2, 7, figsize=(14, 8))
        # fig.subplots_adjust(hspace=0.4, wspace=0.1)
        # cmap = cm.get_cmap('viridis')
        # for idoy, doy in enumerate(valid_dates):
        #     day_data = data_valid[data_valid['DOY_int'] == doy]
        #     # project day_data['hour']*24 to a color range for plotting
        #     color = day_data['hour']*24
        #     sc1 = ax[0, idoy].scatter(day_data['SIFcanopy_760nm'], day_data['SIF_mea'], c=color, cmap=cmap, edgecolor='k')
        #     sc2 = ax[1, idoy].scatter(day_data['SIFPSIILAI_yield_top'], day_data['Fs'], c=color, cmap=cmap, edgecolor='k')

        #     if idoy == 0:
        #         ax[0, idoy].set_ylabel('Measured SIF@760nm' + unitf)
        #         ax[1, idoy].set_ylabel('Fs' + unit_phiF)
        #     else:
        #         ax[0, idoy].set_yticklabels([])
        #         ax[1, idoy].set_yticklabels([])
        #     ax[0, idoy].set_title(f'DOY {doy}')
        #     # ax[0, idoy].set_xlabel('Simulated SIF@760nm' + unitf)
        #     # ax[1, idoy].set_xlabel('$\phi_F (top)$' + unit_phiF + '(simulated)')
        #     ax[0, idoy].set_xlim([0, 2])
        #     ax[1, idoy].set_xlim([0.003, 0.012])
        #     ax[0, idoy].set_ylim([0, 2.5])
        #     ax[1, idoy].set_ylim([0.1, 0.35])
        # # 所有子图共用一个colorbar
        # cbar = fig.colorbar(sc1, ax=ax, orientation='vertical', fraction=0.02, pad=0.01)
        # cbar.set_label('Hour of Day', rotation=90, labelpad=15)
        # # cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
        # # cbar.set_ticklabels(['0', '6', '12', '18', '24'])

        # # 所有子图公用一个xlabel和ylabel
        # fig.text(0.5, 0.48, 'Simulated SIF@760nm' + '(mW m$^{-2}$ nm$^{-1}$ sr$^{-1}$)', ha='center', fontsize=16)
        # fig.text(0.5, 0.02, '$\phi_F (top)$' + unit_phiF + '(simulated)', ha='center', fontsize=16)
        # plt.savefig(os.path.join(path_save_figs, f'SIFyield_top_vs_Fs_VPD_DOY_all_lowVPD.png'), dpi=300)
        # # plt.show()
        # %% SIFyield_top vs Fs at different PARdiffuse_fraction
    #     doy = 199 # 223
    #     data = pd.read_csv(os.path.join(path_save_file, f'SIF_GPP_matching_{path_root.split(os.sep)[-1]}.csv'), index_col=False)
    #     data['hour'] = ((data['DOY_mea'] - data['DOY_mea'].astype(int))*24)
    #     idx = (data['DOY_mea']>= doy) & (data['DOY_mea'] < (doy+1))
    #     # for idoy, doy in enumerate(valid_dates):
    #     layer_files = glob.glob(os.path.join(path_sif_layers, f'BAR_2006_2022_RU390_V5_2_no_stress_ok_SIF_2022_{doy}_*ba.csv'))
    #     dt = []
    #     for j, layer_file in enumerate(layer_files):
    #         layer_data = pd.read_csv(layer_file)
    #         mint_j = layer_file.split(os.sep)[-1].split('_')[-5][-2:]
    #         if mint_j == '04':
    #             mint_j = '00'
    #         elif mint_j == '54':
    #             mint_j = '30'
    #         hour_j = layer_file.split(os.sep)[-1].split('_')[-5][:-2]
    #         # doy_f = int(doy_j) + int(hour_j)/24 + int(mint_j)/(24*60) 
    #         doy_f =int(hour_j)/24.0 + int(mint_j)/(24.0*60.0)
    #         dt.append(doy_f)
    #     # 按照doy_f排序    layer_files    
    #     dt_sorted = np.array(dt)
    #     idx1 = (dt_sorted >= (7.0/24.0)) & (dt_sorted <= (18.0/24.0))
    #     layerfiles = np.array(layer_files)
    #     layer_files = layerfiles[idx1].tolist()
    #     dt_sorted = dt_sorted[idx1].tolist()
    #     layer_files_sorted = [x for _, x in sorted(zip(dt_sorted, layer_files))]
    #     dt_sorted = np.array(sorted(dt_sorted))
    #     ncolumns1 = ['APARtotal','PBhLAI','SIFPSIILAI','NPQLAI']
    #     cbnames1 = ['APAR','GPP','$SIF_{PSII}$','NPQ']
    #     units1 = [unitpar, unitgpp, '(mW m$^{-2}$)', unit_phiF]
    #     ncolumns2 = ['qL', 'phi_P', 'SIFPSIILAI_yield' ]
    #     cbnames2 = ['$q_L$', '$\phi_P$', '$\phi_F$' ]
    #     units2 = [unit_phiF, unit_phiF, unit_phiF]
    #     vmins, vmins1 = [0,0,0,0], [0,0,0]
    #     vmaxs, vmaxs1 = [200,2,0.5,5], [0.7, 0.04, 0.01]
    #     fig, ax = plt.subplots(4, 2,figsize=(18, 12))
    #     fig.subplots_adjust(hspace=0.6)
    #     for icol, col in enumerate(ncolumns1):

    #         df = pd.DataFrame(np.zeros((28, len(dt_sorted))), columns=dt_sorted)
    #         df1 = df.copy()
    #         for j, layer_file in enumerate(layer_files_sorted):
    #             layer_data = pd.read_csv(layer_file, index_col=False)
    #             layer_data['APARtotal'] = (layer_data['PARdiraLAI'] + layer_data['PARdifaLAI'])
    #             layer_data['phi_P'] = layer_data['PBhLAI'] / layer_data['APARtotal']
    #             df.iloc[:,j] = layer_data.loc[:, col]
    #             if icol < len(ncolumns2):
    #                 df1.iloc[:,j] = layer_data.loc[:, ncolumns2[icol]]
    #         cb = ax[icol,0].imshow(df, aspect='auto', cmap='viridis', 
    #                              vmin=vmins[icol], vmax=vmaxs[icol])
    #         ax[icol,0].set_title(f'{cbnames1[icol]} across layers and time')
    #         ax[icol,0].set_ylabel('Layer')
    #         ax[icol,0].set_yticks([0, 22])
    #         ax[icol,0].set_yticklabels(['Tp', 'Bt'])
    #         ax[icol,0].set_xticks(range(0, len(dt_sorted), 5))
    #         ax[icol,0].set_xticklabels([round(x*24, 2) for x in dt_sorted][::5])
    #         cbar = fig.colorbar(cb, ax=ax[icol,0], orientation='vertical')
    #         cbar.set_label(cbnames1[icol] + units1[icol])

    #         if icol < len(ncolumns2):
    #             cb1 = ax[icol,1].imshow(df1, aspect='auto', cmap='viridis', 
    #                                 vmin=vmins1[icol], vmax=vmaxs1[icol])
    #             ax[icol,1].set_title(f'{cbnames2[icol]} across layers and time')
    #             ax[icol,1].set_ylabel('Layer')
    #             ax[icol,1].set_yticks([0, 22])
    #             ax[icol,1].set_yticklabels(['Tp', 'Bt'])
    #             ax[icol,1].set_xticks(range(0, len(dt_sorted), 5))
    #             ax[icol,1].set_xticklabels([round(x*24, 2) for x in dt_sorted][::5])
    #             cbar1 = fig.colorbar(cb1, ax=ax[icol,1], orientation='vertical')
    #             cbar1.set_label(cbnames2[icol] + units2[icol])
        
    #     ax[3,1].bar(range(0, len(data.loc[idx, 'hour'])), data.loc[idx, 'VPD'], color='tab:orange', edgecolor='white')
    #     ax[3,1].set_title('VPD across time')
    #     ax[3,1].set_xlabel('Hour of Day')
    #     ax[3,1].set_ylabel('VPD (kPa)')
    #     # ax[3,1].set_xlim([6, 19])
    #     ax[3,1].set_xticks(range(0, len(data.loc[idx, 'hour']), 5))
    #     ax[3,1].set_xticklabels([round(x, 2) for x in data.loc[idx, 'hour']][::5])
    #     fig.suptitle(f'{folder} - DOY {doy}')

    #     # add text (a), b), c) in the top left corner outside each subplot)
    #     labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']
    #     for i in range(4):
    #         for j in range(2):
    #             ax[i,j].text(-0.1, 1.3, labels[i*2+j], transform=ax[i,j].transAxes, fontsize=22, fontweight='bold', va='top', ha='left')
    #     # fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    #     fig.savefig(os.path.join(path_save_figs, f'Layer_time_{folder}_DOY_{doy}.png'), dpi=300)
    #     # fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    # # plt.show()


    