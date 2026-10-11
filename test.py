"""
Диагностика московских рядов перед построением модели.
Считает то, что в работе нужно предъявить до любой регрессии:
лаг между заболеваемостью и смертностью (CCF), стационарность (ADF/KPSS),
автокорреляцию, overdispersion смертей и темп роста r_t -> R.

Запуск:
    python3 diagnostics.py "/mnt/project-files/Master Diploma"

Требует: pandas, numpy, statsmodels.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf, adfuller, ccf, kpss

BASE = Path(sys.argv[1] if len(sys.argv) > 1 else ".")

# --- загрузка -----------------------------------------------------------------
# sep=';', decimal=',' — иначе числа прочитаются как строки.
off = pd.read_csv(BASE / "Moscow_data_official.csv", sep=";", decimal=",")
off = off.loc[:, ~off.columns.duplicated()]          # последний столбец cases дублирует второй
off = off.dropna(subset=["cases"])
off["t"] = pd.to_datetime(dict(year=off.year, month=off.month, day=1))

exc = pd.read_csv(BASE / "moscow_data.csv")
exc["t"] = pd.to_datetime(dict(year=exc.year, month=exc.month, day=1))

df = off.merge(exc[["t", "death_per_100k_V2"]], on="t", how="inner").sort_values("t")
print(f"наблюдений: {len(df)}  период: {df.t.min():%Y-%m} .. {df.t.max():%Y-%m}\n")

ln_cases = np.log(df["cases_per_100k"].to_numpy())
excess = df["death_per_100k_V2"].to_numpy()
official = df["officcial_death_per_100k"].to_numpy()

# --- 1. лаг по кросс-корреляции ----------------------------------------------
# ccf(x, y)[k] = corr(x_t, y_{t+k}); ищем лаг, на котором связь максимальна.
print("CCF  ln(cases) -> цель   (лаг в месяцах: corr)")
for name, y in (("избыточная V2", excess), ("официальная", official)):
    c = ccf(ln_cases - ln_cases.mean(), y - y.mean(), adjusted=False)[:5]
    best = int(np.argmax(np.abs(c)))
    row = "  ".join(f"L{k}:{v:+.2f}" for k, v in enumerate(c))
    print(f"  {name:>14}: {row}   -> максимум на L={best}")
print()

# --- 2. стационарность: ADF и KPSS проверяют противоположные гипотезы --------
# ADF:  H0 = единичный корень (нестационарен). p < 0.05 -> стационарен.
# KPSS: H0 = стационарен.                      p < 0.05 -> нестационарен.
print("стационарность (ADF p / KPSS p)")
for name, y in (("ln(cases)", ln_cases), ("избыточная V2", excess), ("официальная", official)):
    adf_p = adfuller(y, autolag="AIC")[1]
    kpss_p = kpss(y, regression="c", nlags="auto")[1]
    verdict = "согласованно стационарен" if (adf_p < 0.05 and kpss_p > 0.05) else "спорно — смотри разности"
    print(f"  {name:>14}: ADF {adf_p:.3f} / KPSS {kpss_p:.3f}   {verdict}")
print()

# --- 3. автокорреляция самих рядов -------------------------------------------
print("ACF (лаги 1-4)")
for name, y in (("ln(cases)", ln_cases), ("избыточная V2", excess)):
    a = acf(y, nlags=4, fft=False)[1:]
    print(f"  {name:>14}: " + "  ".join(f"r{k+1}={v:+.2f}" for k, v in enumerate(a)))
print("  (|r1| > 2/sqrt(n) ≈ %.2f -> HAC SE обязательны)\n" % (2 / np.sqrt(len(df))))

# --- 4. overdispersion: Пуассон или отрицательная биномиальная ---------------
# При Пуассоне Var = E. Сравниваем наблюдаемый разброс смертей с пуассоновским.
deaths = df["death"].to_numpy(dtype=float)
cases = df["cases"].to_numpy(dtype=float)
cfr_hat = deaths.sum() / cases.sum()
mu = cases * cfr_hat                       # ожидание при постоянном CFR
pearson_chi2 = (((deaths - mu) ** 2) / mu).sum()
ratio = pearson_chi2 / (len(df) - 1)
print(f"средний CFR по периоду: {cfr_hat*100:.3f} %")
print(f"Pearson chi2/df = {ratio:.1f}  -> "
      + ("Пуассон отвергается, нужна NegBin" if ratio > 1.5 else "Пуассон допустим"))
print()

# --- 5. темп роста и R по Wallinga & Lipsitch --------------------------------
# r = d ln(cases)/dt ; R = (1 + r*s^2/m)^(m^2/s^2) для гамма-интервала генерации.
def r_to_R(r, m, s):
    return (1 + r * s**2 / m) ** (m**2 / s**2)

days = df["t"].diff().dt.days.to_numpy(dtype=float)
r = np.concatenate([[np.nan], np.diff(ln_cases)]) / days
print("темп роста r (1/сутки) и R при двух интервалах генерации")
print("  дата      r        R(m=5.2,s=1.7)  R(m=3.0,s=1.5)")
for t, ri in zip(df["t"], r):
    if np.isnan(ri):
        continue
    pre = r_to_R(ri, 5.2, 1.7)
    omi = r_to_R(ri, 3.0, 1.5)
    print(f"  {t:%Y-%m}  {ri:+.4f}   {pre:6.2f}          {omi:6.2f}")
print("\n  Разница между двумя столбцами — не биология, а выбор интервала генерации."
      "\n  Для Омикрона брать m≈3.0, иначе получится артефакт.")