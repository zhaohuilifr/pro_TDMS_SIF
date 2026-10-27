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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# ------------------------- 参数设置 -------------------------
BASE_DIR = Path(r"E:\Datahub\Barbeau\Data_SIF\SIF3data")
YEARS = [2022, 2023, 2024, 2025]
REL_PATH = Path("PROCESSED/L2/Yearly/PROSIF_rad_irrad_760.csv")
OUT_DIR = BASE_DIR / "rad760_year_comparison"

PAR_EDGES = np.arange(0, 2001, 200)    # 0,200,...,2000 -> 10 组
ALPHA = 0.05
AGGREGATE = None                       # None: 用所有观测点；'daily': 每天每个区间取中位数
MONTHS = None                          # 例如 [6, 7, 8] 只分析生长季；None 为全年
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
    df = pd.read_csv(f, usecols=["Time_mid", "rad_760", "irrad_760", "Integration_time","PAR_micromoles_m_2s_1"],
                     na_values=["NaN", "nan", ""])
    df.loc[(df['rad_760'] < 0) | (df['rad_760'] > 22000), 'rad_760'] = np.nan
    df.loc[(df['irrad_760'] < 0) | (df['irrad_760'] > 22000), 'irrad_760'] = np.nan
    df = df.rename(columns={"PAR_micromoles_m_2s_1": "PAR"})
    df = df.dropna(subset=["Time_mid", "rad_760", "irrad_760", "PAR"])
    df["time"], df["doy"], df["hour"] = matlab_datenum_to_datetime(str(year), df["Time_mid"])
    df["date"] = df["time"].dt.date
    df["year"] = year
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


if __name__ == "__main__":
    main(colname="rad_760")
    main(colname="irrad_760")
    # main(colname="Integration_time")
