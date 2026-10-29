# -*- coding: utf-8 -*-
"""
比较不同年份 rad_760 在不同 PAR 区间（0-2000，10 组，每组 200 μmol m-2 s-1）下是否存在显著差异。

方法：
  1. 每个 PAR 区间内，对各年份 rad_760 做整体检验：
       - Kruskal-Wallis H 检验（非参数，主结果，rad_760 一般不服从正态分布）
       - 单因素 ANOVA（参数，作参考）
       - 效应量 ε²（Kruskal-Wallis）和 η²（ANOVA）
  2. 整体检验显著时，做年份两两比较：Mann-Whitney U 检验 + Holm 校正
  3. 输出汇总表（CSV）和箱线图

注意：每组样本量很大（上万个点），p 值几乎一定显著，请同时看效应量和中位数差异。
     另外 2 分钟一次的观测存在自相关，若要更稳健，可将 AGGREGATE 设为 'daily'，
     先在每天每个 PAR 区间内取中位数，再按“天”做检验。
"""

from itertools import combinations
from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit

# ------------------------- 参数设置 -------------------------
BASE_DIR = Path(r"E:\Datahub\Barbeau\Data_SIF\SIF3data")
YEARS = [2022, 2023, 2024, 2025]
REL_PATH = Path("PROCESSED/New folder/L2/Yearly/PROSIF_rad_irrad_760.csv")
OUT_DIR = BASE_DIR / "rad760_year_comparison"

PAR_EDGES = np.arange(0, 2001, 200)    # 0,200,...,2000 -> 10 组
ALPHA = 0.05
AGGREGATE = None                       # None: 用所有观测点；'daily': 每天每个区间取中位数
MONTHS = [6,7,8,9]                          # 例如 [6, 7, 8] 只分析生长季；None 为全年
MIN_N = 10                             # 某年某区间样本数少于该值时不参与检验


def matlab_datenum_to_datetime(yearstr, datenum, offset_hours=2):
    """MATLAB datenum -> datetime（向量化，datenum 可为 Series）
    719529 为 1970-01-01 对应的 MATLAB datenum；2022 年时间偏移 5 h，2023-2025 年偏移 2 h
    """
    if yearstr == '2022':  # 2023 - 2025, offset_hours = 2
        offset_hours = 5
    dt = pd.to_datetime(datenum - 719529, unit='D') + pd.Timedelta(hours=offset_hours)
    hour = dt.dt.hour + dt.dt.minute / 60 + dt.dt.second / 3600
    doy = dt.dt.dayofyear + hour / 24
    return dt, doy, hour


def load_year(year):
    f = BASE_DIR / str(year) / REL_PATH
    df = pd.read_csv(f, usecols=["Time_mid", "rad_760", "irrad_760", 'rad_760_dc', 'rad_760_dn',"Integration_time","PAR_micromoles_m_2s_1"],
                     na_values=["NaN", "nan", ""])
    df.loc[(df['rad_760'] < 0) | (df['rad_760'] > 22000), 'rad_760'] = np.nan
    df.loc[(df['irrad_760'] < 0) | (df['irrad_760'] > 22000), 'irrad_760'] = np.nan
    df = df.rename(columns={"PAR_micromoles_m_2s_1": "PAR"})
    df = df.dropna(subset=["Time_mid", "rad_760", "irrad_760", "PAR"])
    df["time"], df["doy"], df["hour"] = matlab_datenum_to_datetime(str(year), df["Time_mid"])
    df["date"] = df["time"].dt.date
    df["year"] = year
    # 只取生长季（6-9月）和白天（9-16点）的数据
    if MONTHS is not None:
        df = df[df["time"].dt.month.isin(MONTHS)]
    if "hour" in df.columns:
        df = df[(df["hour"] >= 5) & (df["hour"] <= 19)] #13
    return df


def holm(pvals):
    """Holm-Bonferroni 校正"""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(running, 1.0)
    return adj


def main(colname="rad_760"):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data = pd.concat([load_year(y) for y in YEARS], ignore_index=True)
    if MONTHS is not None:
        data = data[data["time"].dt.month.isin(MONTHS)]

    labels = [f"{a}-{b}" for a, b in zip(PAR_EDGES[:-1], PAR_EDGES[1:])]
    data["PAR_bin"] = pd.cut(data["PAR"], bins=PAR_EDGES, labels=labels,
                             right=False, include_lowest=True)
    data = data.dropna(subset=["PAR_bin"])

    if AGGREGATE == "daily":
        data = (data.groupby(["year", "date", "PAR_bin"], observed=True)[colname]
                    .median().reset_index())

    overall_rows, pair_rows = [], []

    for lab in labels:
        sub = data[data["PAR_bin"] == lab]
        groups = {y: sub.loc[sub["year"] == y, colname].to_numpy() for y in YEARS}
        groups = {y: g for y, g in groups.items() if len(g) >= MIN_N}

        row = {"PAR_bin": lab}
        for y in YEARS:
            g = sub.loc[sub["year"] == y, colname]
            row[f"n_{y}"] = len(g)
            row[f"median_{y}"] = g.median() if len(g) else np.nan
            row[f"mean_{y}"] = g.mean() if len(g) else np.nan

        if len(groups) < 2:
            row.update(KW_H=np.nan, KW_p=np.nan, eps2=np.nan,
                       ANOVA_F=np.nan, ANOVA_p=np.nan, eta2=np.nan, significant=False)
            overall_rows.append(row)
            continue

        arrs = list(groups.values())
        N, k = sum(len(a) for a in arrs), len(arrs)

        H, p_kw = stats.kruskal(*arrs)
        eps2 = (H - k + 1) / (N - k)                       # Kruskal-Wallis 效应量

        F, p_an = stats.f_oneway(*arrs)
        allv = np.concatenate(arrs)
        ss_between = sum(len(a) * (a.mean() - allv.mean()) ** 2 for a in arrs)
        ss_total = ((allv - allv.mean()) ** 2).sum()
        eta2 = ss_between / ss_total

        row.update(KW_H=H, KW_p=p_kw, eps2=eps2,
                   ANOVA_F=F, ANOVA_p=p_an, eta2=eta2, significant=p_kw < ALPHA)
        overall_rows.append(row)

        # 两两比较
        if p_kw < ALPHA:
            pairs = list(combinations(groups.keys(), 2))
            tmp = []
            for y1, y2 in pairs:
                a, b = groups[y1], groups[y2]
                U, p = stats.mannwhitneyu(a, b, alternative="two-sided")
                r_rb = 1 - 2 * U / (len(a) * len(b))       # rank-biserial 相关（效应量）
                tmp.append(dict(PAR_bin=lab, year1=y1, year2=y2,
                                median1=np.median(a), median2=np.median(b),
                                median_diff=np.median(a) - np.median(b),
                                U=U, p_raw=p, rank_biserial=r_rb))
            p_adj = holm([t["p_raw"] for t in tmp])
            for t, pa in zip(tmp, p_adj):
                t["p_holm"] = pa
                t["significant"] = pa < ALPHA
            pair_rows.extend(tmp)

    overall = pd.DataFrame(overall_rows)
    pairwise = pd.DataFrame(pair_rows)
    suffix = f"_{AGGREGATE}" if AGGREGATE else ""
    overall.to_csv(OUT_DIR / f"overall_test{suffix}.csv", index=False)
    pairwise.to_csv(OUT_DIR / f"pairwise_test{suffix}.csv", index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print("=== 各 PAR 区间的整体检验 ===")
    print(overall[["PAR_bin"] + [f"n_{y}" for y in YEARS] + [f"median_{y}" for y in YEARS]
                  + ["KW_H", "KW_p", "eps2", "ANOVA_p", "eta2", "significant"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    if not pairwise.empty:
        print("\n=== 两两比较 (Mann-Whitney U + Holm) ===")
        print(pairwise[["PAR_bin", "year1", "year2", "median_diff", "p_holm",
                        "rank_biserial", "significant"]]
              .to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    # ------------------------- 作图 -------------------------
    fig, ax = plt.subplots(figsize=(14, 6))
    n_y = len(YEARS)
    width = 0.8 / n_y
    colors = plt.cm.tab10.colors
    for j, y in enumerate(YEARS):
        vals = [data.loc[(data["PAR_bin"] == lab) & (data["year"] == y), colname].to_numpy()
                for lab in labels]
        pos = np.arange(len(labels)) + (j - (n_y - 1) / 2) * width
        bp = ax.boxplot([v if len(v) else [np.nan] for v in vals], positions=pos,
                        widths=width * 0.9, patch_artist=True, showfliers=False)
        for box in bp["boxes"]:
            box.set_facecolor(colors[j]); box.set_alpha(0.7)
        ax.plot([], [], color=colors[j], lw=8, alpha=0.7, label=str(y))

    # 在显著区间上方标星号，显示2022是否和2023，2024，2025等年份有显著差异
    ymax = ax.get_ylim()[1]
    # for lab in labels:
    #     sub = pairwise[pairwise["PAR_bin"] == lab]
    #     if sub.empty:
    #         continue
    #     for _, r in sub.iterrows():
    #         if r["year1"] == 2022 and r["significant"]:
    #             star = "***" if r["p_holm"] < 0.001 else "**" if r["p_holm"] < 0.01 else "*" if r["p_holm"] < 0.05 else "ns"
    #             x = labels.index(lab)
    #             y = ymax * 0.97
    #             ax.text(x, y, star, ha="center", va="top", fontsize=10)
    for i, r in overall.iterrows():
        p = r["KW_p"]
        if pd.notna(p):
            star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            ax.text(i, ymax * 0.97, star, ha="center", va="top", fontsize=10)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30)
    ax.set_xlabel("PAR (μmol m$^{-2}$ s$^{-1}$)")
    # ax.set_ylabel(f"{colname}")
    ax.set_ylabel("(DN-DC)/inttime" if (colname == "rad_760") or (colname == "irrad_760") else "inttime")
    ax.set_title(f"{colname} by PAR bin and year (Kruskal-Wallis)")
    ax.legend(title="Year", ncol=n_y)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{colname}_by_PAR_year{suffix}.png", dpi=300)
    plt.show()

    print(f"\n结果已保存到: {OUT_DIR}")


# 
def load_year_sif(year):
    yearpath = os.path.join(BASE_DIR, year,'PROCESSED\\L2\\Yearly')
    df_sif = pd.read_csv(os.path.join(yearpath, 'PROSIF_SIFresults_' + year + '_Yearly.csv'), index_col=False)
    df_sif = df_sif.rename(columns={"PAR_micromoles_m_2s_1": "PAR"})
    # time_spend
    df_sif['time_spend'] = (df_sif['Time_end'] - df_sif['Time_start']) * 24 * 60 * 60 # seconds
    # idx_time = (df_sif['time_spend'] > 1) & (df_sif['time_spend'] <5.5)
    # idx_value = (df_sif['SIF_3FLD'] > 0) & (df_sif['SIF_3FLD'] < 10)
    # df_sif = df_sif[idx_time & idx_value]

    idx_time = (df_sif['time_spend'] >= 0) & (df_sif['time_spend'] <5.5)
    idx_value = (df_sif['SIF_3FLD'] > 0) & (df_sif['SIF_3FLD'] < 10)
    df_sif = df_sif[idx_value & idx_time]

    # df_sif['DOY_sif'] = df_sif['Time_mid'].apply(lambda x: matlab_datenum_to_datetime(year, x)[1])
    # df_sif['Hour'] = df_sif['Time_mid'].apply(lambda x: matlab_datenum_to_datetime(year, x)[2])
    df_sif["time"], df_sif["doy"], df_sif["hour"] = matlab_datenum_to_datetime(str(year), df_sif["Time_mid"])
    df_sif["date"] = df_sif["time"].dt.date
    df_sif["year"] = year
    # 只取生长季（6-9月）和白天（9-16点）的数据
    if MONTHS is not None:
        df_sif = df_sif[df_sif["time"].dt.month.isin(MONTHS)]
    if "hour" in df_sif.columns:
        df_sif = df_sif[(df_sif["hour"] >= 5) & (df_sif["hour"] <= 19)] #13
    return df_sif

def load_year_flux(year):
    path = r'E:\Datahub\Barbeau\Data_matched_new4'
    df = pd.read_excel(os.path.join(path, 'Barbeau_' + year + '_matched_CASTANEA_full_APAR.xlsx'), sheet_name='Sheet1')
    if MONTHS is not None:
        df = df[df['mois'].isin(MONTHS)]
    if "hour_UTC" in df.columns:
        df = df[(df["hour_UTC"] >= 5) & (df["hour_UTC"] <= 19)] #13

    return df



if __name__ == "__main__":
    # main(colname="rad_760")
    # main(colname="irrad_760")
    # main(colname="Integration_time")

    # %% daily comparison
    # data2022 = r'E:\Datahub\Barbeau\Data_SIF\SIF3data\2022\PROCESSED\L2\Daily\20220812_HR_SIF_Daily.csv'
    # data2023 = r'E:\Datahub\Barbeau\Data_SIF\SIF3data\2023\PROCESSED\L2\Daily\20230812_HR_SIF_Daily.csv'
    # data2024 = r'E:\Datahub\Barbeau\Data_SIF\SIF3data\2024\PROCESSED\L2\Daily\20240812_HR_SIF_Daily.csv'
    # data2025 = r'E:\Datahub\Barbeau\Data_SIF\SIF3data\2025\PROCESSED\L2\Daily\20250812_HR_SIF_Daily.csv'

    # df_2022 = pd.read_csv(data2022)
    # df_2023 = pd.read_csv(data2023)
    # df_2024 = pd.read_csv(data2024)
    # df_2025 = pd.read_csv(data2025)

    df_2022 = load_year(2022)
    df_2023 = load_year(2023)
    df_2024 = load_year(2024)
    df_2025 = load_year(2025)

    choose_doy_2022 = 179 # 164
    choose_doy_2023 = 176 # 164
    choose_doy_2024 = 176 # 164
    choose_doy_2025 = 179 # 164
    # choose doy = 194

    df_20221 = df_2022[(df_2022['doy'] >= choose_doy_2022) & (df_2022['doy'] <= choose_doy_2022+1)]
    df_20231 = df_2023[(df_2023['doy'] >= choose_doy_2023) & (df_2023['doy'] <= choose_doy_2023+1)]
    df_20241 = df_2024[(df_2024['doy'] >= choose_doy_2024) & (df_2024['doy'] <= choose_doy_2024+1)]
    df_20251 = df_2025[(df_2025['doy'] >= choose_doy_2025) & (df_2025['doy'] <= choose_doy_2025+1)]

    # %% plot rad_760 vs hour for each year
    # fig, ax = plt.subplots(4,4, figsize=(18, 10))
    # fig.subplots_adjust(left=0.1, right=0.95, wspace=0.2, hspace=0.33)
    # ax[0,0].plot(df_20221['hour'], df_20221['PAR'], 'k.-', label='2022')
    # ax[0,1].plot(df_20221['hour'], df_20221['rad_760'], 'k.-', label='2022')
    # ax[0,2].plot(df_20221['hour'], df_20221['rad_760_dn']/df_20221['Integration_time'], 'k.-', label='2022')
    # ax[0,3].plot(df_20221['hour'], df_20221['rad_760_dc']/df_20221['Integration_time'], 'k.-', label='2022')

    # ax[1,0].plot(df_20231['hour'], df_20231['PAR'], 'k.-', label='2023')
    # ax[1,1].plot(df_20231['hour'], df_20231['rad_760'], 'k.-', label='2023')
    # ax[1,2].plot(df_20231['hour'], df_20231['rad_760_dn']/df_20231['Integration_time'], 'k.-', label='2023')
    # ax[1,3].plot(df_20231['hour'], df_20231['rad_760_dc']/df_20231['Integration_time'], 'k.-', label='2023')

    # ax[2,0].plot(df_20241['hour'], df_20241['PAR'], 'k.-', label='2024')
    # ax[2,1].plot(df_20241['hour'], df_20241['rad_760'], 'k.-', label='2024')
    # ax[2,2].plot(df_20241['hour'], df_20241['rad_760_dn']/df_20241['Integration_time'], 'k.-', label='2024')
    # ax[2,3].plot(df_20241['hour'], df_20241['rad_760_dc']/df_20241['Integration_time'], 'k.-', label='2024')

    # ax[3,0].plot(df_20251['hour'], df_20251['PAR'], 'k.-', label='2025')
    # ax[3,1].plot(df_20251['hour'], df_20251['rad_760'], 'k.-', label='2025')
    # ax[3,2].plot(df_20251['hour'], df_20251['rad_760_dn']/df_20251['Integration_time'], 'k.-', label='2025')
    # ax[3,3].plot(df_20251['hour'], df_20251['rad_760_dc']/df_20251['Integration_time'], 'k.-', label='2025')
    
    # for i in range(4):
    #     ax[0,i].set_title(f'Year {YEARS[0]} - DOY {choose_doy_2022}')
    #     ax[1,i].set_title(f'Year {YEARS[1]} - DOY {choose_doy_2023}')
    #     ax[2,i].set_title(f'Year {YEARS[2]} - DOY {choose_doy_2024}')
    #     ax[3,i].set_title(f'Year {YEARS[3]} - DOY {choose_doy_2025}')
    #     ax[3,i].set_xlabel('Hour of Day')
    # for i in range(4):
    #     ax[i,0].set_ylabel('PAR (μmol m$^{-2}$ s$^{-1}$)')
    #     ax[i,0].set_ylim(0, 2000)
    #     for j in range(4):
    #         # ax[i,0].legend()
    #         ax[i,j].grid()
    #     ax[i,1].set_ylabel('(DN-DC)/inttime')
    #     ax[i,2].set_ylabel('DN/inttime')
    #     ax[i,3].set_ylabel('DC/inttime')
    #     # ax[i,1].legend()
    #     ax[i,1].set_ylim(0, 22000)
    #     ax[i,2].set_ylim(0, 22000)
    #     ax[i,3].set_ylim(0, 6500)

    # fig.tight_layout()
    # fig.savefig(OUT_DIR / f"rad_760_PAR_hour_comparison.png", dpi=300)
    
    # %% plot rad_760_dn vs rad_760_dc for each year
    # fig, ax = plt.subplots(1,4, figsize=(16, 5))
    # fig.subplots_adjust(left=0.1, right=0.95, wspace=0.2, hspace=0.33)
    # ax[0].scatter(df_20221['rad_760_dn']/df_20221['Integration_time'], df_20221['rad_760_dc']/df_20221['Integration_time'], s=10, c=df_20221['hour'], cmap='viridis', vmin=5, vmax=19, label='2022')
    # ax[1].scatter(df_20231['rad_760_dn']/df_20231['Integration_time'], df_20231['rad_760_dc']/df_20231['Integration_time'], s=10, c=df_20231['hour'], cmap='viridis', vmin=5, vmax=19, label='2023')
    # ax[2].scatter(df_20241['rad_760_dn']/df_20241['Integration_time'], df_20241['rad_760_dc']/df_20241['Integration_time'], s=10, c=df_20241['hour'], cmap='viridis', vmin=5, vmax=19, label='2024')
    # ax[3].scatter(df_20251['rad_760_dn']/df_20251['Integration_time'], df_20251['rad_760_dc']/df_20251['Integration_time'], s=10, c=df_20251['hour'], cmap='viridis', vmin=5, vmax=19, label='2025')
    # # calculate linear regression for each year
    # for i, df in enumerate([df_20221, df_20231, df_20241, df_20251]):
    #     # calculate linear regression for each year with intercept as 0
    #     xd = (df['rad_760_dn']/df['Integration_time']).to_numpy()
    #     yd = (df['rad_760_dc']/df['Integration_time']).to_numpy()
    #     valid = np.isfinite(xd) & np.isfinite(yd)
    #     xd, yd = xd[valid], yd[valid]

    #     # slope = np.sum(xd * yd) / np.sum(xd ** 2)  # least squares through origin
    #     # r2 = 1 - np.sum((yd - slope * xd) ** 2) / np.sum((yd - yd.mean()) ** 2)
    #     # x = np.array([0, 20000])
    #     # y = slope * x
    #     # ax[i].plot(x, y, 'r--', label=f'y={slope:.4f}x, R²={r2:.4f}')
    #     # power-law fit through origin: y = a * x^b

    #     pos = (xd > 0) & (yd > 0)
    #     b0, loga0 = np.polyfit(np.log(xd[pos]), np.log(yd[pos]), 1)  # initial guess from log-log fit
    #     (a, b), _ = curve_fit(lambda x, a, b: a * x ** b, xd[pos], yd[pos], p0=[np.exp(loga0), b0])
    #     r2 = 1 - np.sum((yd[pos] - a * xd[pos] ** b) ** 2) / np.sum((yd[pos] - yd[pos].mean()) ** 2)
    #     x = np.linspace(0, 20000, 200)
    #     y = a * x ** b
    #     if i == 0:
    #         ax[i].plot(x, y, 'r--', label=f'y={a:.2f}x^{b:.2f}, R²={r2:.2f}')
    #         for j in range(1, 4):
    #             ax[j].plot(x, y, 'r--', label=f'2022 fit: y={a:.2f}x^{b:.2f}, R²={r2:.2f}')
    #     else:
    #         ax[i].plot(x, y, 'k--', label=f'y={a:.2f}x^{b:.2f}, R²={r2:.2f}')
            
    #     # add one colorbar for all subplots
    #     if i == 3:
    #         cbar = plt.colorbar(ax[i].collections[0], ax=ax, orientation='vertical', fraction=0.02, pad=0.04)
    #         cbar.set_label('Hour of Day')
    #     ax[i].set_title(f'Year {YEARS[i]}')
    #     ax[i].set_xlabel('DN/inttime')
    #     ax[i].set_ylabel('DC/inttime')
    #     ax[i].legend()
    #     ax[i].grid()
    #     ax[i].set_xlim(0, 22000)
    #     ax[i].set_ylim(0, 6500)
    #     if i>0:
    #         ax[i].set_yticklabels([])
    # # fig.tight_layout()
    # # fig.savefig(OUT_DIR / f"rad_760_dn_dc_comparison_5-19.png", dpi=300)
    # fig.savefig(OUT_DIR / f"rad_760_dn_dc_comparison_13-19.png", dpi=300)

    # %% plot 
    df_2022_flux = load_year_flux('2022')
    df_2023_flux = load_year_flux('2023')
    df_2024_flux = load_year_flux('2024')
    df_2025_flux = load_year_flux('2025')

    df_20221_flux = df_2022_flux[(df_2022_flux['DOY_UTC'] >= choose_doy_2022) & (df_2022_flux['DOY_UTC'] <= choose_doy_2022+1)]
    df_20231_flux = df_2023_flux[(df_2023_flux['DOY_UTC'] >= choose_doy_2023) & (df_2023_flux['DOY_UTC'] <= choose_doy_2023+1)]
    df_20241_flux = df_2024_flux[(df_2024_flux['DOY_UTC'] >= choose_doy_2024) & (df_2024_flux['DOY_UTC'] <= choose_doy_2024+1)]
    df_20251_flux = df_2025_flux[(df_2025_flux['DOY_UTC'] >= choose_doy_2025) & (df_2025_flux['DOY_UTC'] <= choose_doy_2025+1)]
    fig, ax = plt.subplots(4,1, figsize=(10, 10))
    fig.subplots_adjust(left=0.1, right=0.95, wspace=0.2, hspace=0.33)
    ax[0].plot(df_2022_flux['DOY_UTC'], df_2022_flux['Ta (°C)'], 'k.-', label='2022')
    ax[1].plot(df_2023_flux['DOY_UTC'], df_2023_flux['Ta (°C)'], 'k.-', label='2023')
    ax[2].plot(df_2024_flux['DOY_UTC'], df_2024_flux['Ta (°C)'], 'k.-', label='2024')
    ax[3].plot(df_2025_flux['DOY_UTC'], df_2025_flux['Ta (°C)'], 'k.-', label='2025')
    for i in range(4):
        ax[i].set_title(f'Year {YEARS[i]}')
        if i == 3:
            ax[i].set_xlabel('DOY')
        ax[i].set_ylabel('Tair (°C)')
        ax[i].grid()
        ax[i].set_ylim(0, 45)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"seasonal_Tair.png", dpi=300)


    df_2022 = load_year_sif('2022')
    df_2023 = load_year_sif('2023')
    df_2024 = load_year_sif('2024')
    df_2025 = load_year_sif('2025')

    choose_doy_2022 = 179 # 164
    choose_doy_2023 = 176 # 164
    choose_doy_2024 = 176 # 164
    choose_doy_2025 = 179 # 164
    df_20221 = df_2022[(df_2022['doy'] >= choose_doy_2022) & (df_2022['doy'] <= choose_doy_2022+1)]
    df_20231 = df_2023[(df_2023['doy'] >= choose_doy_2023) & (df_2023['doy'] <= choose_doy_2023+1)]
    df_20241 = df_2024[(df_2024['doy'] >= choose_doy_2024) & (df_2024['doy'] <= choose_doy_2024+1)]
    df_20251 = df_2025[(df_2025['doy'] >= choose_doy_2025) & (df_2025['doy'] <= choose_doy_2025+1)]

    # Plot the data
    fig, ax = plt.subplots(4,4, figsize=(18, 10))
    fig.subplots_adjust(left=0.1, right=0.95, wspace=0.2, hspace=0.33)
    ax[0,0].plot(df_20221['hour'], df_20221['PAR'], 'k.-', label='2022')
    ax[0,1].plot(df_20221['hour'], df_20221['BOX_Temperature_celsius'], 'k.-', label='2022 Tbox')
    ax0 = ax[0,1].twinx()
    ax0.plot((df_20221_flux['DOY_UTC']-choose_doy_2022)*24, df_20221_flux['Ta (°C)'], 'r.-', label='2022 Tair')
    ax[0,2].plot(df_20221['hour'], df_20221['rad_760'], 'k.-', label='2022')
    ax[0,3].plot(df_20221['hour'], df_20221['SIF_3FLD'], 'k.-', label='2022')

    ax[1,0].plot(df_20231['hour'], df_20231['PAR'], 'k.-', label='2023')
    ax[1,1].plot(df_20231['hour'], df_20231['BOX_Temperature_celsius'], 'k.-', label='2023 Tbox')
    ax1 = ax[1,1].twinx()
    ax1.plot((df_20231_flux['DOY_UTC']-choose_doy_2023)*24, df_20231_flux['Ta (°C)'], 'r.-', label='2023 Tair')
    ax[1,2].plot(df_20231['hour'], df_20231['rad_760'], 'k.-', label='2023')
    ax[1,3].plot(df_20231['hour'], df_20231['SIF_3FLD'], 'k.-', label='2023')

    ax[2,0].plot(df_20241['hour'], df_20241['PAR'], 'k.-', label='2024')
    ax[2,1].plot(df_20241['hour'], df_20241['BOX_Temperature_celsius'], 'k.-', label='2024 Tbox')
    ax2 = ax[2,1].twinx()
    ax2.plot((df_20241_flux['DOY_UTC']-choose_doy_2024)*24, df_20241_flux['Ta (°C)'], 'r.-', label='2024 Tair')
    ax[2,2].plot(df_20241['hour'], df_20241['rad_760'], 'k.-', label='2024')
    ax[2,3].plot(df_20241['hour'], df_20241['SIF_3FLD'], 'k.-', label='2024')

    ax[3,0].plot(df_20251['hour'], df_20251['PAR'], 'k.-', label='2025')
    ax[3,1].plot(df_20251['hour'], df_20251['BOX_Temperature_celsius'], 'k.-', label='2025 Tbox')
    ax3 = ax[3,1].twinx()
    ax3.plot((df_20251_flux['DOY_UTC']-choose_doy_2025)*24, df_20251_flux['Ta (°C)'], 'r.-', label='2025 Tair')
    ax[3,2].plot(df_20251['hour'], df_20251['rad_760'], 'k.-', label='2025')
    ax[3,3].plot(df_20251['hour'], df_20251['SIF_3FLD'], 'k.-', label='2025')
    
    for i in range(4):
        ax[0,i].set_title(f'Year {YEARS[0]} - DOY {choose_doy_2022}')
        ax[1,i].set_title(f'Year {YEARS[1]} - DOY {choose_doy_2023}')
        ax[2,i].set_title(f'Year {YEARS[2]} - DOY {choose_doy_2024}')
        ax[3,i].set_title(f'Year {YEARS[3]} - DOY {choose_doy_2025}')
        ax[3,i].set_xlabel('Hour of Day')
    for i in range(4):
        ax[i,0].set_ylabel('PAR (μmol m$^{-2}$ s$^{-1}$)')
        
        for j in range(4):
            # ax[i,0].legend()
            ax[i,j].grid()
        ax[i,1].set_ylabel('Box Temperature (°C)')
        ax[i,2].set_ylabel('Radiance@760nm'+'\n'+'(mW m$^{-2}$ sr$^{-1}$ nm$^{-1}$)')
        ax[i,3].set_ylabel('SIF@760nm'+'\n'+'(mW m$^{-2}$ sr$^{-1}$ nm$^{-1}$)')
        # ax[i,1].legend()
        ax[i,0].set_ylim(0, 2000)
        ax[i,1].set_ylim(15, 25)
        ax[i,2].set_ylim(0, 60)
        ax[i,3].set_ylim(0, 2.5)
    for ax_i in [ax0, ax1, ax2, ax3]:
        ax_i.set_ylabel('Tair (°C)', color='r')
        ax_i.tick_params(axis='y', labelcolor='r')
        ax_i.set_ylim(0, 45)

    fig.tight_layout()
    fig.savefig(OUT_DIR / f"par_Tbox_Tair_radiance_sif.png", dpi=300)

    # %% seasonal Tbox
    fig, ax = plt.subplots(4,1, figsize=(10, 10))
    fig.subplots_adjust(left=0.1, right=0.95, wspace=0.2, hspace=0.33)
    ax[0].plot(df_2022['doy'], df_2022['BOX_Temperature_celsius'], 'k.-', label='2022')
    ax[1].plot(df_2023['doy'], df_2023['BOX_Temperature_celsius'], 'k.-', label='2023')
    ax[2].plot(df_2024['doy'], df_2024['BOX_Temperature_celsius'], 'k.-', label='2024')
    ax[3].plot(df_2025['doy'], df_2025['BOX_Temperature_celsius'], 'k.-', label='2025')
    for i in range(4):
        ax[i].set_title(f'Year {YEARS[i]}')
        if i == 3:
            ax[i].set_xlabel('DOY')
        ax[i].set_ylabel('BOX Temperature (°C)')
        ax[i].grid()
        ax[i].set_ylim(15, 25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"seasonal_Tbox.png", dpi=300)

    # %% seasonal Tair

    plt.show()