"""GrowSari demand forecasting engine.

This module is extracted from the Colab notebook and wrapped for Streamlit / CLI usage.
It supports reading demand history from local files, uploaded file-like objects, HTTP(S),
and S3 URLs using boto3 credentials.
"""
from __future__ import annotations

import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


ALL_MODELS = [
    "Croston", "ETS", "Ensemble", "LightGBM", "MovingAvg", "Naive", "SBA",
    "SeasonalNaive", "SeasonalRollAvg", "TSB", "XGBoost",
]

# ===================================================================== #
#  PHILIPPINE HOLIDAY CALENDAR  (2024-2026 verified + auto-extend)       #
# ===================================================================== #
# Movable holidays (Holy Week, Chinese New Year, Eid, Black Saturday) — verified per
# Proclamations 368/727/1006. Eid dates are approximate until separately proclaimed.
_MOVABLE = {
    2024: {"2024-02-10": "Chinese New Year", "2024-03-28": "Maundy Thursday",
           "2024-03-29": "Good Friday", "2024-03-30": "Black Saturday",
           "2024-04-10": "Eidul Fitr", "2024-06-17": "Eidul Adha"},
    2025: {"2025-01-29": "Chinese New Year", "2025-04-17": "Maundy Thursday",
           "2025-04-18": "Good Friday", "2025-04-19": "Black Saturday",
           "2025-03-31": "Eidul Fitr", "2025-06-06": "Eidul Adha",
           "2025-07-27": "INC Founding"},
    2026: {"2026-02-17": "Chinese New Year", "2026-04-02": "Maundy Thursday",
           "2026-04-03": "Good Friday", "2026-04-04": "Black Saturday",
           "2026-03-20": "Eidul Fitr", "2026-05-27": "Eidul Adha",
           "2026-02-25": "EDSA (special working)"},
}
# regular holidays that move slightly handled above; the rest are fixed-date
_FIXED_REGULAR = {"01-01": "New Year's Day", "04-09": "Araw ng Kagitingan",
                  "05-01": "Labor Day", "06-12": "Independence Day",
                  "11-30": "Bonifacio Day", "12-25": "Christmas Day", "12-30": "Rizal Day"}
_FIXED_SPECIAL = {"08-21": "Ninoy Aquino Day", "11-01": "All Saints' Day",
                  "11-02": "All Souls' Day", "12-08": "Immaculate Conception",
                  "12-24": "Christmas Eve", "12-31": "Last Day of Year"}


def _last_monday_august(year):
    d = pd.Timestamp(year, 8, 31)
    return d - pd.Timedelta(days=(d.dayofweek - 0) % 7)


def build_ph_holidays(year_min, year_max):
    """Return dict {Timestamp: (name, 'regular'|'special')} spanning the year range."""
    out = {}
    for y in range(year_min, year_max + 1):
        for mmdd, nm in _FIXED_REGULAR.items():
            out[pd.Timestamp(f"{y}-{mmdd}")] = (nm, "regular")
        for mmdd, nm in _FIXED_SPECIAL.items():
            out[pd.Timestamp(f"{y}-{mmdd}")] = (nm, "special")
        out[_last_monday_august(y)] = ("National Heroes Day", "regular")
        for ds, nm in _MOVABLE.get(y, {}).items():
            kind = "special" if ("special" in nm or "Black" in nm or "CNY" in nm
                                 or "New Year" in nm or "INC" in nm or "EDSA" in nm) else "regular"
            # Holy Week Thu/Fri & Eid are regular; CNY/Black Saturday/INC/EDSA special
            if nm in ("Maundy Thursday", "Good Friday", "Eidul Fitr", "Eidul Adha"):
                kind = "regular"
            else:
                kind = "special"
            out[pd.Timestamp(ds)] = (nm, kind)
    return out


# ===================================================================== #
#  TRADING CALENDAR  (data-driven: closed weekdays auto-detected)        #
# ===================================================================== #
# NOTE (fix): v3 hard-coded Sunday as closed. This data has real Sunday demand,
# so closed_dows now defaults to () = open every day and is auto-detected from
# the data after load (see detect_closed_dows in the next engine cell).
def trading_days_between(start, end, closed_dows=()):
    """Count trading days in [start, end] excluding closed weekdays (default: none)."""
    rng = pd.date_range(start, end, freq="D")
    return int(sum(d.dayofweek not in closed_dows for d in rng))


# ===================================================================== #
#  ADAPTIVE DEMAND CLASSIFICATION  (Syntetos-Boylan + ABC + lifecycle)  #
# ===================================================================== #
DEFAULT_PARAMS = dict(
    abc_a=0.80, abc_b=0.95,
    adi_cut=1.32, cv2_cut=0.49,
    new_history_days=60,        # auto-scaled if history is long
    dormant_recency_days=28,    # auto-scaled
    closed_dows=(),             # auto-detected after load (was (6,) = Sunday closed)
    seasonality_min_cycles=2,   # need >=2 cycles to assess seasonality
)


def _series_metrics(daily_y, dates, ref_date, params):
    """Compute SB + lifecycle metrics for one series given its daily demand vector.
    History is the series' OWN active window (first demand -> ref_date) in trading days."""
    demand_days = int((daily_y > 0).sum())
    nonzero = daily_y[daily_y > 0]
    total = float(daily_y.sum())
    first_demand = dates[daily_y > 0].min() if demand_days > 0 else dates.min()
    history_td = trading_days_between(first_demand, ref_date, params["closed_dows"])
    # ADI = avg interval between demands (in trading-day units), over the series' active window
    adi = history_td / demand_days if demand_days > 0 else np.inf
    cv2 = float((nonzero.std(ddof=0) / nonzero.mean()) ** 2) if demand_days > 1 and nonzero.mean() > 0 else 0.0
    last_demand = dates[daily_y > 0].max() if demand_days > 0 else None
    recency_td = trading_days_between(last_demand, ref_date, params["closed_dows"]) - 1 if last_demand is not None else 9999
    coverage = demand_days / history_td if history_td > 0 else 0.0
    return dict(total_units=total, demand_days=demand_days, history_td=history_td,
                recency_td=max(recency_td, 0), coverage=coverage, adi=adi, cv2=cv2)


def _seasonality_trend(y, period):
    """Strength of seasonality & trend via STL (returns 0..1 each). Needs >=2 periods."""
    if period < 2 or len(y) < 2 * period + 1 or (y > 0).sum() < period:
        return np.nan, np.nan
    try:
        from statsmodels.tsa.seasonal import STL
        res = STL(np.where(y <= 0, 1e-6, y), period=period, robust=True).fit()
        var_resid = np.var(res.resid)
        s_strength = max(0.0, 1 - var_resid / max(np.var(res.resid + res.seasonal), 1e-9))
        t_strength = max(0.0, 1 - var_resid / max(np.var(res.resid + res.trend), 1e-9))
        return round(float(s_strength), 3), round(float(t_strength), 3)
    except Exception:
        return np.nan, np.nan


def _pattern(adi, cv2, p):
    if adi <= p["adi_cut"] and cv2 <= p["cv2_cut"]: return "Smooth"
    if adi <= p["adi_cut"] and cv2 >  p["cv2_cut"]: return "Erratic"
    if adi >  p["adi_cut"] and cv2 <= p["cv2_cut"]: return "Intermittent"
    return "Lumpy"


def _recommend_method(seg, has_yoy, s_strength):
    seasonal = (has_yoy and (s_strength or 0) >= 0.3)
    table = {
        "Smooth":       ("ETS/LightGBM + seasonal features" if seasonal else "ETS/LightGBM + day-of-week", "daily"),
        "Erratic":      ("Quantile GBM / ETS damped on weekly buckets", "weekly"),
        "Intermittent": ("Croston / SBA / TSB", "weekly"),
        "Lumpy":        ("TSB / empirical resampling", "weekly/monthly"),
        "Dormant":      ("Exclude from active forecasting", "—"),
        "New":          ("Category-proxy disaggregation", "daily proxy"),
    }
    return table.get(seg, ("ETS", "daily"))


def classify_level(demand, group_keys, ref_date, params=None, name_col="name",
                   category_col="category", seasonal_period_daily=None):
    """
    Classify all series at one aggregation level.
    demand: long df with date, warehouse_id, channel, sku_id, category, name, y
    group_keys: subset of [warehouse_id, sku_id, category, channel] defining the grain.
    Returns a tidy classification DataFrame.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    ref_date = pd.Timestamp(ref_date)
    span_start, span_end = demand["date"].min(), ref_date
    history_td = trading_days_between(span_start, span_end, p["closed_dows"])

    # ---- adapt thresholds to available history ----
    hist_days = (span_end - span_start).days
    if hist_days >= 540:            # ~1.5+ yr -> longer windows are meaningful
        p["new_history_days"] = max(p["new_history_days"], 90)
        p["dormant_recency_days"] = max(p["dormant_recency_days"], 42)
    has_yoy = hist_days >= 700      # ~2 years -> YoY seasonal models unlockable
    sp = seasonal_period_daily or (7 - len(p["closed_dows"]))   # 6 for Mon-Sat

    g = demand.groupby(group_keys + ["date"])["y"].sum().reset_index()
    rows = []
    for key, sub in g.groupby(group_keys):
        key = key if isinstance(key, tuple) else (key,)
        s = sub.set_index("date")["y"].sort_index()
        dates = s.index
        m = _series_metrics(s.values, dates, ref_date, p)
        # lifecycle
        if m["demand_days"] == 0 or m["recency_td"] > p["dormant_recency_days"]:
            lifecycle = "Dormant"
        elif m["history_td"] < p["new_history_days"]:
            lifecycle = "New"
        else:
            lifecycle = "Active"
        pattern = _pattern(m["adi"], m["cv2"], p) if lifecycle == "Active" else None
        segment = lifecycle if lifecycle != "Active" else pattern
        s_str, t_str = (_seasonality_trend(s.reindex(
            pd.date_range(dates.min(), dates.max())).fillna(0).values, sp)
            if lifecycle == "Active" else (np.nan, np.nan))
        rec_method, rec_grain = _recommend_method(segment, has_yoy, s_str)
        rowd = dict(zip(group_keys, key))
        rowd.update(m)
        rowd.update(dict(lifecycle=lifecycle, pattern=pattern, segment=segment,
                         seasonality_strength=s_str, trend_strength=t_str,
                         has_yoy=has_yoy, rec_method=rec_method, rec_grain=rec_grain))
        rows.append(rowd)
    out = pd.DataFrame(rows)

    # ---- ABC within level ----
    out = out.sort_values("total_units", ascending=False).reset_index(drop=True)
    grand = out["total_units"].sum()
    out["units_share"] = out["total_units"] / grand if grand > 0 else 0
    out["cum_share"] = out["units_share"].cumsum()
    prev = out["cum_share"] - out["units_share"]
    out["ABC"] = np.where(prev < p["abc_a"], "A", np.where(prev < p["abc_b"], "B", "C"))
    out["coverage_pct"] = (out["coverage"] * 100).round(1)
    out["adi"] = out["adi"].round(3); out["cv2"] = out["cv2"].round(4)
    return out


LEVEL_KEYS = {
    "WH_SKU":              ["warehouse_id", "sku_id"],
    "WH_Category":         ["warehouse_id", "category"],
    "WH_SKU_Channel":      ["warehouse_id", "sku_id", "channel"],
    "WH_Category_Channel": ["warehouse_id", "category", "channel"],
}


def classify_all(demand, ref_date, params=None, levels=None):
    levels = levels or list(LEVEL_KEYS)
    return {lv: classify_level(demand, LEVEL_KEYS[lv], ref_date, params) for lv in levels}

GRAIN_ORDER = {"daily": 0, "weekly": 1, "monthly": 2}

# --- Trading calendar is DATA-DRIVEN (fix) ------------------------------ #
# v3 hard-coded CLOSED_DOWS=(6,) ("Sunday closed"). build_daily_series,
# _future_periods and seasonal_period all key off TRADING_DOW, so every Sunday
# was dropped from the series grid, the May test window (-> 26 instead of 31
# points) and the forecast horizon, and the backtest actual total came out
# LOWER than the true historical demand. This dataset has genuine Sunday sales,
# so we default to open-every-day and auto-detect any truly-closed weekday.
CLOSED_DOWS = ()                          # reassigned by detect_closed_dows() after load
TRADING_DOW = tuple(d for d in range(7) if d not in CLOSED_DOWS)

def detect_closed_dows(demand_df, rel_threshold=0.02):
    """A weekday (0=Mon..6=Sun) is CLOSED only if its TOTAL demand across the whole
    dataset is <= rel_threshold x the median open-weekday total. Returns () for
    7-day data, (6,) for genuine Mon-Sat data, etc."""
    by = demand_df.groupby(demand_df["date"].dt.dayofweek)["y"].sum().reindex(range(7), fill_value=0.0)
    open_med = by[by > 0].median() if (by > 0).any() else 0.0
    return tuple(int(d) for d in range(7) if by[d] <= rel_threshold * open_med)
MIN_OBS_ML = {"daily": 24, "weekly": 12, "monthly": 18}   # below this -> skip ML

# ===================================================================== #
#  SERIES BUILD + TEMPORAL RESAMPLE                                      #
# ===================================================================== #
def build_daily_series(sub):
    """sub: rows for ONE series with date,y[,availability]. Dense Mon-Sat grid, 0-fill."""
    s = sub.groupby("date").agg(y=("y", "sum"),
                                availability=("availability", "mean")).reset_index()
    full = pd.date_range(s["date"].min(), s["date"].max(), freq="D")
    full = full[[d.dayofweek in TRADING_DOW for d in full]]
    out = pd.DataFrame({"ds": full}).merge(s.rename(columns={"date": "ds"}), on="ds", how="left")
    out["y"] = out["y"].fillna(0.0)
    out["availability"] = out["availability"].fillna(1.0)
    return out


def resample_grain(daily, grain, holidays):
    """Aggregate a daily series to daily/weekly/monthly with calendar counts."""
    d = daily.copy()
    if grain == "daily":
        d["n_trading"] = 1
        d["n_holiday"] = d["ds"].isin(holidays).astype(int)
        return d
    if grain == "weekly":
        d["bucket"] = d["ds"] - pd.to_timedelta(d["ds"].dt.dayofweek, unit="D")
    else:  # monthly
        d["bucket"] = d["ds"].values.astype("datetime64[M]")
    agg = (d.groupby("bucket")
           .agg(y=("y", "sum"), availability=("availability", "mean"),
                n_trading=("y", "size"),
                n_holiday=("ds", lambda x: sum(t in holidays for t in x)))
           .reset_index().rename(columns={"bucket": "ds"}).sort_values("ds"))
    return agg


def grain_to_bucket(ds_series, to_grain):
    if to_grain == "daily":   return ds_series
    if to_grain == "weekly":  return ds_series - pd.to_timedelta(ds_series.dt.dayofweek, unit="D")
    return ds_series.values.astype("datetime64[M]")


def seasonal_period(grain, history_len):
    base = {"daily": len(TRADING_DOW), "weekly": 52, "monthly": 12}[grain]
    return base if history_len >= 2 * base + 1 else None


# ===================================================================== #
#  FEATURE ENGINEERING (grain-aware)                                    #
# ===================================================================== #
def add_features(df, grain, holidays, reg_dates, spc_dates):
    d = df.copy().reset_index(drop=True)
    ds = d["ds"]
    if grain == "daily":
        d["dow"] = ds.dt.dayofweek; d["day"] = ds.dt.day; d["month"] = ds.dt.month
        d["weekofmonth"] = (ds.dt.day - 1)//7 + 1
        d["is_payday"] = ds.dt.day.isin(list(range(13,17))+list(range(29,32))+[1,2]).astype(int)
        d["is_holiday"] = ds.isin(holidays).astype(int)
        d["is_reg_hol"] = ds.isin(reg_dates).astype(int)
        d["is_spc_hol"] = ds.isin(spc_dates).astype(int)
        hs = sorted(holidays)
        d["pre_hol"]  = [1 if any(0 < (h-t).days <= 3 for h in hs) else 0 for t in ds]
        d["post_hol"] = [1 if any(0 < (t-h).days <= 2 for h in hs) else 0 for t in ds]
        lags, rolls = (1,2,3,6,12,18), (3,6,12,24)
        seas = "dow"
    elif grain == "weekly":
        d["weekofyear"] = ds.dt.isocalendar().week.astype(int)
        d["month"] = ds.dt.month; d["weekofmonth"] = (ds.dt.day-1)//7 + 1
        lags, rolls = (1,2,3,4,52), (2,3,4,8)
        seas = "weekofyear"
    else:
        d["month"] = ds.dt.month; d["quarter"] = ds.dt.quarter
        lags, rolls = (1,2,3,12), (2,3,6)
        seas = "month"
    for L in lags:
        d[f"lag_{L}"] = d["y"].shift(L)
    for R in rolls:
        d[f"rmean_{R}"] = d["y"].shift(1).rolling(R, min_periods=1).mean()
        d[f"rstd_{R}"]  = d["y"].shift(1).rolling(R, min_periods=2).std()
    d["ewm"] = d["y"].shift(1).ewm(span=max(rolls[1], 3), min_periods=1).mean()
    d["seas_expmean"] = (d.groupby(seas)["y"]
                         .apply(lambda s: s.shift(1).expanding(min_periods=1).mean())
                         .reset_index(level=0, drop=True))
    d["t_index"] = np.arange(len(d))
    feat_cols = [c for c in d.columns if c not in ("ds", "y", "availability")]
    return d, feat_cols


# ===================================================================== #
#  MODEL LIBRARY  (uniform: return preds for the future rows)           #
# ===================================================================== #
def _inv(x, use_log):  return np.expm1(x) if use_log else x
def _fwd(x, use_log):  return np.log1p(np.clip(x, 0, None)) if use_log else x

def m_naive(train, n, **k):
    return np.repeat(max(0.0, train["y"].iloc[-1]), n)

def m_seasonal_naive(train, n, sp=None, fdates=None, **k):
    y = train["y"].values
    if sp and len(y) >= sp:
        base = y[-sp:]
        return np.array([max(0.0, base[i % sp]) for i in range(n)])
    return np.repeat(max(0.0, np.mean(y[-3:])), n)

def m_moving_avg(train, n, window=4, **k):
    return np.repeat(max(0.0, train["y"].tail(window).mean()), n)

def m_seasonal_rollavg(train, n, grain="daily", fdates=None, **k):
    if grain == "daily" and fdates is not None:
        dm = train.groupby(train["ds"].dt.dayofweek)["y"].apply(lambda s: s.tail(4).mean())
        recent = train["y"].tail(12).mean()
        return np.array([max(0.0, dm.get(dt.dayofweek, recent)*0.7 + recent*0.3) for dt in fdates])
    return np.repeat(max(0.0, train["y"].tail(4).mean()), n)

def m_ets(train, n, sp=None, use_log=False, **k):
    y = train["y"].astype(float).values
    yt = _fwd(y, use_log)
    if len(y) < 6 or (y > 0).sum() < 3:
        return np.repeat(max(0.0, np.mean(y[-3:]) if len(y) else 0), n)
    try:
        seasonal = "add" if (sp and len(y) >= 2*sp+1 and (y > 0).sum() >= sp) else None
        fit = ExponentialSmoothing(np.where(yt <= 0, 1e-6, yt) if not use_log else yt,
                                   trend="add", damped_trend=True,
                                   seasonal=seasonal, seasonal_periods=sp if seasonal else None,
                                   initialization_method="estimated").fit()
        fc = _inv(fit.forecast(n), use_log)
        return np.clip(np.nan_to_num(fc, nan=np.median(y)), 0, None)
    except Exception:
        return np.repeat(max(0.0, np.mean(y[-3:])), n)

# ---- intermittent demand ----
def _croston_core(y, n, alpha=0.1, variant="classic"):
    y = np.asarray(y, float)
    nz = np.where(y > 0)[0]
    if len(nz) == 0: return np.zeros(n)
    z = y[nz[0]]; p = 1.0; q = 1
    for i in range(nz[0]+1, len(y)):
        if y[i] > 0:
            z += alpha*(y[i]-z); p += alpha*(q-p); q = 1
        else:
            q += 1
    if variant == "sba": f = (1 - alpha/2) * z/p
    elif variant == "tsb":
        prob = 0.0; lvl = z
        for i in range(len(y)):
            occ = 1.0 if y[i] > 0 else 0.0
            prob += alpha*(occ-prob)
            if y[i] > 0: lvl += alpha*(y[i]-lvl)
        f = prob*lvl
    else: f = z/p
    return np.repeat(max(0.0, f), n)

def m_croston(train, n, **k): return _croston_core(train["y"].values, n, variant="classic")
def m_sba(train, n, **k):     return _croston_core(train["y"].values, n, variant="sba")
def m_tsb(train, n, **k):     return _croston_core(train["y"].values, n, variant="tsb")

# ---- ML recursive ----
def _new_ml(kind):
    if kind == "xgb":
        return XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.9,
                            colsample_bytree=0.9, min_child_weight=3, random_state=42,
                            n_jobs=2, verbosity=0)
    return LGBMRegressor(n_estimators=300, max_depth=6, num_leaves=31, learning_rate=0.05,
                         subsample=0.9, colsample_bytree=0.9, min_child_samples=5,
                         random_state=42, n_jobs=2, verbose=-1)

def m_ml(train_hist, fut_index, kind, grain, holidays, reg, spc, use_log=False,
         avail_future=1.0):
    """Recursive one-step ML. train_hist & fut_index carry 'ds' (+ n_trading/n_holiday for w/m)."""
    work = train_hist[["ds", "y", "availability"]].copy()
    df, fcols = add_features(work, grain, holidays, reg, spc)
    tr = df.dropna(subset=[c for c in fcols if c.startswith("lag_")][-1:])
    if len(tr) < 15: tr = df.fillna(0)
    Xy = tr.copy(); Xy["y"] = _fwd(Xy["y"].values, use_log)
    mdl = _new_ml(kind); mdl.fit(Xy[fcols].fillna(0), Xy["y"])
    preds, cur = [], work.copy()
    for _, r in fut_index.iterrows():
        cur = pd.concat([cur, pd.DataFrame({"ds":[r["ds"]], "y":[np.nan],
                                            "availability":[avail_future]})], ignore_index=True)
        f, _ = add_features(cur, grain, holidays, reg, spc)
        p = _inv(float(mdl.predict(f.iloc[[-1]][fcols].fillna(0))[0]), use_log)
        p = max(0.0, p); preds.append(p)
        cur.iloc[-1, cur.columns.get_loc("y")] = p
    return np.array(preds)

STAT_MODELS = {"Naive": m_naive, "SeasonalNaive": m_seasonal_naive, "MovingAvg": m_moving_avg,
               "SeasonalRollAvg": m_seasonal_rollavg, "ETS": m_ets,
               "Croston": m_croston, "SBA": m_sba, "TSB": m_tsb}
ML_MODELS = {"XGBoost": "xgb", "LightGBM": "lgb"}


# ===================================================================== #
#  METRICS                                                              #
# ===================================================================== #
def compute_metrics(a, p):
    a = np.asarray(a, float); p = np.asarray(p, float)
    err = p - a; abserr = np.abs(err)
    sum_a = a.sum()
    wape = abserr.sum()/sum_a if sum_a > 0 else np.nan
    nz = a > 0
    mape = np.mean(abserr[nz]/a[nz]) if nz.any() else np.nan
    smape = np.mean(2*abserr/(np.abs(a)+np.abs(p)+1e-9))
    mm = [min(x,y)/max(x,y) for x,y in zip(a,p) if max(x,y) > 0]
    agg_mm = (min(sum_a, p.sum())/max(sum_a, p.sum())) if max(sum_a, p.sum()) > 0 else np.nan
    return dict(weighted_accuracy=(1-wape) if wape==wape else np.nan,
                minmax_bucket=np.mean(mm) if mm else np.nan, minmax_agg=agg_mm,
                mape=mape, smape=smape, mae=abserr.mean(),
                rmse=np.sqrt((err**2).mean()), bias=err.mean(), test_volume=sum_a)

SELECT_DIR = {"weighted_accuracy": "max", "minmax_agg": "max", "minmax_bucket": "max",
              "mape": "min", "smape": "min", "mae": "min", "rmse": "min"}

def validate_grains(temporal, forecast):
    if GRAIN_ORDER[forecast] < GRAIN_ORDER[temporal]:
        raise ValueError(f"FORECAST_AGG='{forecast}' must be >= TEMPORAL_AGG='{temporal}' "
                         "(daily < weekly < monthly).")

def _future_periods(last_ds, grain, fc_end):
    """Modeling-grain bucket starts from day after last data through fc_end (inclusive).
    Full chain generated so recursive ML stays continuous; output sliced to window later."""
    start = last_ds + pd.Timedelta(days=1); fc_end = pd.Timestamp(fc_end)
    if grain == "daily":
        rng = pd.date_range(start, fc_end, freq="D")
        return pd.DatetimeIndex([d for d in rng if d.dayofweek in TRADING_DOW])
    if grain == "weekly":
        ws = start - pd.to_timedelta(start.dayofweek, unit="D")
        return pd.date_range(ws, fc_end, freq="W-MON")
    return pd.date_range(start.replace(day=1), fc_end, freq="MS")

def _bucket_end(period_start, fgrain):
    p = pd.Timestamp(period_start)
    if fgrain == "daily":   return p
    if fgrain == "weekly":  return p + pd.Timedelta(days=5)
    return (p + pd.DateOffset(months=1)) - pd.Timedelta(days=1)

def _in_window(period_start, fgrain, win_start, win_end):
    ps = pd.Timestamp(period_start); pe = _bucket_end(ps, fgrain)
    return (pe >= pd.Timestamp(win_start)) and (ps <= pd.Timestamp(win_end))

def _drop_trailing_partial(g, grain):
    if grain == "weekly" and len(g) and g["n_trading"].iloc[-1] < 6: g = g.iloc[:-1]
    if grain == "monthly" and len(g) and g["n_trading"].iloc[-1] < 20: g = g.iloc[:-1]
    return g.reset_index(drop=True)


def run_series(daily_df, label, meta, cfg, holidays, reg, spc):
    """Backtest on chosen TEST window + live forecast over chosen FORECAST window.
    cfg carries timestamps TEST_START, TEST_END, FC_START, FC_END (+ grains/options)."""
    grain, fgrain = cfg["TEMPORAL_AGG"], cfg["FORECAST_AGG"]
    use_log = cfg.get("PRE_TRANSFORM", "none") == "log1p"
    g = _drop_trailing_partial(resample_grain(daily_df, grain, holidays), grain)
    if g["y"].sum() == 0 or len(g) < 8: return None
    sp = seasonal_period(grain, len(g))
    test_start, test_end = pd.Timestamp(cfg["TEST_START"]), pd.Timestamp(cfg["TEST_END"])
    tr = g[g["ds"] < test_start].copy()
    te = g[(g["ds"] >= test_start) & (g["ds"] <= test_end)].copy()
    if len(te) < 1 or len(tr) < 6: return None

    models = cfg["RUN_MODELS"]; rows = []
    for nm in models:
        if nm in STAT_MODELS:
            p = STAT_MODELS[nm](tr, len(te), sp=sp, grain=grain, use_log=use_log, fdates=list(te["ds"]))
            for dt, pv, av in zip(te["ds"], p, te["y"]): rows.append((dt, nm, av, pv))
    ml_ok = len(tr) >= MIN_OBS_ML[grain]
    for nm, kind in ML_MODELS.items():
        if nm in models and ml_ok:
            p = m_ml(tr, te[["ds","availability"]].assign(availability=1.0), kind, grain,
                       holidays, reg, spc, use_log=use_log)
            for dt, pv, av in zip(te["ds"], p, te["y"]): rows.append((dt, nm, av, pv))
    if not rows: return None
    md = pd.DataFrame(rows, columns=["ds","model","actual","pred"])
    if "Ensemble" in models:
        ml_present = [m for m in ML_MODELS if m in md["model"].unique()]
        base = md[md.model.isin(ml_present)] if ml_present else md
        ens = base.groupby("ds").agg(actual=("actual","first"), pred=("pred","mean")).reset_index()
        ens["model"] = "Ensemble"
        md = pd.concat([md, ens[["ds","model","actual","pred"]]], ignore_index=True)

    md["fbucket"] = grain_to_bucket(md["ds"], fgrain)
    roll = md.groupby(["model","fbucket"]).agg(actual=("actual","sum"), pred=("pred","sum")).reset_index()
    lb_rows = []
    for nm, sgrp in roll.groupby("model"):
        met = compute_metrics(sgrp["actual"].values, sgrp["pred"].values)
        lb_rows.append({"series": label, **meta, "model": nm, **met})
    leaderboard = pd.DataFrame(lb_rows)
    metric = cfg["SELECTION_METRIC"]; direction = SELECT_DIR[metric]
    valid = leaderboard.dropna(subset=[metric]);  valid = valid if not valid.empty else leaderboard
    champ = valid.sort_values(metric, ascending=(direction=="min")).iloc[0]["model"]
    leaderboard["is_champion"] = leaderboard["model"] == champ
    leaderboard["champion_model"] = champ
    roll = roll.rename(columns={"fbucket":"period"}); roll.insert(0,"series",label)

    gfull = _drop_trailing_partial(resample_grain(daily_df, grain, holidays), grain)
    fc_start, fc_end = pd.Timestamp(cfg["FC_START"]), pd.Timestamp(cfg["FC_END"])
    fperiods = _future_periods(gfull["ds"].max(), grain, fc_end)
    fc = pd.DataFrame(columns=["series","model","period","forecast","is_champion"])
    if len(fperiods):
        fut_idx = pd.DataFrame({"ds": fperiods, "availability": 1.0})
        spf = seasonal_period(grain, len(gfull)); frows = []
        for nm in models:
            if nm in STAT_MODELS:
                p = STAT_MODELS[nm](gfull, len(fperiods), sp=spf, grain=grain, use_log=use_log, fdates=list(fperiods))
                for dt, pv in zip(fperiods, p): frows.append((dt, nm, pv))
        ml_ok2 = len(gfull) >= MIN_OBS_ML[grain]
        for nm, kind in ML_MODELS.items():
            if nm in models and ml_ok2:
                p = m_ml(gfull, fut_idx, kind, grain, holidays, reg, spc, use_log=use_log)
                for dt, pv in zip(fperiods, p): frows.append((dt, nm, pv))
        fcr = pd.DataFrame(frows, columns=["ds","model","forecast"])
        if "Ensemble" in models and not fcr.empty:
            ml_present = [m for m in ML_MODELS if m in fcr["model"].unique()]
            base = fcr[fcr.model.isin(ml_present)] if ml_present else fcr
            ensf = base.groupby("ds")["forecast"].mean().reset_index(); ensf["model"]="Ensemble"
            fcr = pd.concat([fcr, ensf[["ds","model","forecast"]]], ignore_index=True)
        fcr["period"] = grain_to_bucket(fcr["ds"], fgrain)
        fcr = fcr.groupby(["model","period"])["forecast"].sum().reset_index()
        keep = fcr["period"].apply(lambda ps: _in_window(ps, fgrain, fc_start, fc_end))
        fcr = fcr[keep].copy()
        fcr.insert(0,"series",label); fcr["is_champion"] = fcr["model"] == champ
        for kk,vv in meta.items(): fcr[kk]=vv
        fc = fcr
    return dict(backtest=roll, leaderboard=leaderboard, forecast=fc, champion=champ)


# ---- combination intake from uploaded Excel ----
COL_ALIASES = {
    "warehouse_id": ["warehouse_id","warehouse","wh","warehouse id"],
    "sku_id":       ["sku_id","sku","sku id","skuid"],
    "category":     ["category","cat","product category"],
    "channel":      ["channel","chan"],
}
def _norm_cols(df):
    out = df.copy(); lower = {c.lower().strip(): c for c in out.columns}; rename = {}
    for canon, alts in COL_ALIASES.items():
        for a in alts:
            if a in lower: rename[lower[a]] = canon; break
    return out.rename(columns=rename)

def load_combinations(combos_df, level, level_keys, classification_df=None):
    df = _norm_cols(combos_df)
    missing = [k for k in level_keys if k not in df.columns]
    if missing:
        raise ValueError(f"Uploaded file for level '{level}' is missing column(s): {missing}. "
                         f"Required columns: {level_keys}.")
    df = df[level_keys].dropna().drop_duplicates().reset_index(drop=True)
    if "sku_id" in df.columns:
        df["sku_id"] = pd.to_numeric(df["sku_id"], errors="coerce").astype("Int64")
        df = df.dropna(subset=["sku_id"]); df["sku_id"] = df["sku_id"].astype(int)
    if classification_df is not None:
        meta_cols = [c for c in ["segment","ABC","lifecycle","total_units"] if c in classification_df.columns]
        df = df.merge(classification_df[level_keys + meta_cols], on=level_keys, how="left")
        df["_found"] = df[meta_cols[0]].notna() if meta_cols else True
    return df

def combinations_template(level, level_keys, classification_df, top=None):
    cols = level_keys + [c for c in ["segment","ABC","lifecycle","total_units"]
                         if c in classification_df.columns]
    t = classification_df.sort_values("total_units", ascending=False)[cols].reset_index(drop=True)
    return t.head(top) if top else t

# ===================================================================== #
#  DATA LOADING + STREAMLIT-FRIENDLY WRAPPERS                            #
# ===================================================================== #

DATE_ALIASES = ["demand_date", "date", "ds", "order_date"]
DEMAND_ALIASES = ["demand_units", "demand", "actual_demand", "y", "qty", "quantity", "units"]
WAREHOUSE_ALIASES = ["warehouse_id", "warehouse", "dc_id", "dc", "wh", "depot", "depot_id"]
SKU_ALIASES = ["sku_id", "sku", "skuid", "item_id", "product_id"]
AVAILABILITY_ALIASES = ["availability_score", "availability", "available_score"]
CHANNEL_ALIASES = ["channel", "sales_channel", "order_channel"]
CATEGORY_ALIASES = ["category", "cat", "product_category"]
NAME_ALIASES = ["name", "sku_name", "product_name", "item_name"]


def _first_present(df: pd.DataFrame, aliases: List[str]) -> Optional[str]:
    lower = {str(c).strip().lower(): c for c in df.columns}
    for alias in aliases:
        if alias.lower() in lower:
            return lower[alias.lower()]
    return None


def _parse_s3_uri(uri: str) -> Tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path:
        raise ValueError(f"Invalid S3 URI: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def read_csv_source(
    source: Union[str, os.PathLike, io.BytesIO, io.StringIO, Any],
    *,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    aws_session_token: Optional[str] = None,
    region_name: Optional[str] = None,
    unsigned: bool = False,
    **read_csv_kwargs: Any,
) -> pd.DataFrame:
    """Read a CSV from local path, uploaded file-like object, HTTP(S), or s3://bucket/key.

    For private S3 buckets, provide AWS credentials via args, environment variables,
    Streamlit secrets, IAM role, or your default AWS profile.
    """
    if hasattr(source, "read"):
        return pd.read_csv(source, **read_csv_kwargs)

    source_str = str(source).strip()
    if source_str.startswith("s3://"):
        import boto3
        if unsigned:
            from botocore import UNSIGNED
            from botocore.config import Config
            client = boto3.client("s3", region_name=region_name, config=Config(signature_version=UNSIGNED))
        else:
            client = boto3.client(
                "s3",
                aws_access_key_id=aws_access_key_id or None,
                aws_secret_access_key=aws_secret_access_key or None,
                aws_session_token=aws_session_token or None,
                region_name=region_name or None,
            )
        bucket, key = _parse_s3_uri(source_str)
        obj = client.get_object(Bucket=bucket, Key=key)
        # StreamingBody keeps the app from needing a manual upload. Pandas will stream-read it.
        return pd.read_csv(obj["Body"], **read_csv_kwargs)

    return pd.read_csv(source_str, **read_csv_kwargs)


def standardize_demand_columns(raw: pd.DataFrame) -> pd.DataFrame:
    """Map common column variants to the engine contract."""
    df = raw.copy()
    rename: Dict[Any, str] = {}

    date_col = _first_present(df, DATE_ALIASES)
    y_col = _first_present(df, DEMAND_ALIASES)
    wh_col = _first_present(df, WAREHOUSE_ALIASES)
    sku_col = _first_present(df, SKU_ALIASES)
    availability_col = _first_present(df, AVAILABILITY_ALIASES)
    channel_col = _first_present(df, CHANNEL_ALIASES)
    category_col = _first_present(df, CATEGORY_ALIASES)
    name_col = _first_present(df, NAME_ALIASES)

    required = {
        "date/demand_date": date_col,
        "demand_units/demand/y": y_col,
        "warehouse_id": wh_col,
        "sku_id": sku_col,
    }
    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise ValueError(
            "Demand history is missing required columns: " + ", ".join(missing) +
            ". Expected at least date, demand quantity, warehouse_id, and sku_id."
        )

    rename[date_col] = "date"
    rename[y_col] = "y"
    rename[wh_col] = "warehouse_id"
    rename[sku_col] = "sku_id"
    if availability_col is not None:
        rename[availability_col] = "availability"
    if channel_col is not None:
        rename[channel_col] = "channel"
    if category_col is not None:
        rename[category_col] = "category"
    if name_col is not None:
        rename[name_col] = "name"

    df = df.rename(columns=rename)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce").fillna(0.0)
    df["sku_id"] = pd.to_numeric(df["sku_id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["date", "sku_id", "warehouse_id"])
    df["sku_id"] = df["sku_id"].astype(int)
    if "availability" not in df.columns:
        df["availability"] = 1.0
    else:
        df["availability"] = pd.to_numeric(df["availability"], errors="coerce").fillna(1.0)
    if "channel" not in df.columns:
        df["channel"] = "ALL"
    df["warehouse_id"] = df["warehouse_id"].astype(str)
    df["channel"] = df["channel"].astype(str)
    return df


def standardize_sku_master(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    sku_col = _first_present(df, SKU_ALIASES)
    category_col = _first_present(df, CATEGORY_ALIASES)
    name_col = _first_present(df, NAME_ALIASES)
    if sku_col is None:
        raise ValueError("SKU master must contain sku_id or an equivalent SKU column.")
    rename = {sku_col: "sku_id"}
    if category_col is not None:
        rename[category_col] = "category"
    if name_col is not None:
        rename[name_col] = "name"
    df = df.rename(columns=rename)
    keep = [c for c in ["sku_id", "name", "category"] if c in df.columns]
    df = df[keep].drop_duplicates(subset=["sku_id"])
    df["sku_id"] = pd.to_numeric(df["sku_id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["sku_id"])
    df["sku_id"] = df["sku_id"].astype(int)
    if "category" not in df.columns:
        df["category"] = "Unmapped"
    if "name" not in df.columns:
        df["name"] = df["sku_id"].astype(str)
    return df


def prepare_demand_data(demand_raw: pd.DataFrame, sku_master_raw: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Return demand in engine-ready format with category/name added from SKU master."""
    demand = standardize_demand_columns(demand_raw)
    if sku_master_raw is not None and not sku_master_raw.empty:
        skum = standardize_sku_master(sku_master_raw)
        # Prefer category/name from SKU master, but don't destroy already-present values unnecessarily.
        demand = demand.merge(skum, on="sku_id", how="left", suffixes=("", "_master"))
        if "category_master" in demand.columns:
            demand["category"] = demand.get("category", pd.Series(index=demand.index, dtype=object)).fillna(demand["category_master"])
            demand = demand.drop(columns=["category_master"])
        if "name_master" in demand.columns:
            demand["name"] = demand.get("name", pd.Series(index=demand.index, dtype=object)).fillna(demand["name_master"])
            demand = demand.drop(columns=["name_master"])
    if "category" not in demand.columns:
        demand["category"] = "Unmapped"
    if "name" not in demand.columns:
        demand["name"] = demand["sku_id"].astype(str)
    demand["category"] = demand["category"].fillna("Unmapped").astype(str)
    demand["name"] = demand["name"].fillna(demand["sku_id"].astype(str)).astype(str)
    return demand


def load_demand_from_sources(
    demand_source: Union[str, os.PathLike, io.BytesIO, io.StringIO, Any],
    sku_master_source: Optional[Union[str, os.PathLike, io.BytesIO, io.StringIO, Any]] = None,
    *,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    aws_session_token: Optional[str] = None,
    region_name: Optional[str] = None,
    unsigned_s3: bool = False,
) -> pd.DataFrame:
    demand_raw = read_csv_source(
        demand_source,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_session_token=aws_session_token,
        region_name=region_name,
        unsigned=unsigned_s3,
    )
    sku_raw = None
    if sku_master_source is not None:
        sku_raw = read_csv_source(
            sku_master_source,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            region_name=region_name,
            unsigned=unsigned_s3,
        )
    return prepare_demand_data(demand_raw, sku_raw)


@dataclass
class ForecastContext:
    demand: pd.DataFrame
    ref_date: pd.Timestamp
    holidays: Dict[pd.Timestamp, Tuple[str, str]]
    regular_holidays: set
    special_holidays: set
    classif: Dict[str, pd.DataFrame]
    closed_dows: Tuple[int, ...]
    trading_dow: Tuple[int, ...]


def build_context(demand: pd.DataFrame) -> ForecastContext:
    """Build classifications and holiday/trading-calendar context for one dataset."""
    global CLOSED_DOWS, TRADING_DOW
    if demand.empty:
        raise ValueError("Demand dataframe is empty after cleaning.")
    ref_date = demand["date"].max()
    closed_dows = detect_closed_dows(demand)
    trading_dow = tuple(d for d in range(7) if d not in closed_dows)
    CLOSED_DOWS = closed_dows
    TRADING_DOW = trading_dow
    DEFAULT_PARAMS["closed_dows"] = closed_dows
    hol = build_ph_holidays(demand["date"].min().year, ref_date.year + 1)
    reg = {d for d, v in hol.items() if v[1] == "regular"}
    spc = {d for d, v in hol.items() if v[1] == "special"}
    classif = classify_all(demand, ref_date)
    return ForecastContext(demand, ref_date, hol, reg, spc, classif, closed_dows, trading_dow)


def month_options(ctx: ForecastContext, future_months: int = 12) -> Tuple[List[str], List[str], str, str]:
    hist_months = sorted(ctx.demand["date"].dt.to_period("M").astype(str).unique())
    future = [(ctx.ref_date.to_period("M") + i).strftime("%Y-%m") for i in range(0, future_months + 1)]
    all_months = sorted(set(hist_months) | set(future))
    last_full = (ctx.ref_date.replace(day=1) - pd.Timedelta(days=1)).strftime("%Y-%m")
    next_month = (ctx.ref_date.to_period("M") + 1).strftime("%Y-%m")
    if last_full not in hist_months and hist_months:
        last_full = hist_months[-1]
    return hist_months, all_months, last_full, next_month


def build_cfg(
    temporal_agg: str,
    forecast_agg: str,
    test_months: Iterable[str],
    forecast_start_month: str,
    forecast_end_month: str,
    selection_metric: str = "weighted_accuracy",
    pre_transform: str = "log1p",
    run_models: Optional[List[str]] = None,
) -> Dict[str, Any]:
    validate_grains(temporal_agg, forecast_agg)
    test_months = sorted(list(test_months))
    if not test_months:
        raise ValueError("Select at least one test month.")
    test_start = pd.Timestamp(test_months[0] + "-01")
    test_end = pd.Timestamp(test_months[-1] + "-01") + pd.offsets.MonthEnd(0)
    fc_start = pd.Timestamp(forecast_start_month + "-01")
    fc_end = pd.Timestamp(forecast_end_month + "-01") + pd.offsets.MonthEnd(0)
    if fc_end < fc_start:
        raise ValueError("Forecast end month must be greater than or equal to forecast start month.")
    return {
        "TEMPORAL_AGG": temporal_agg,
        "FORECAST_AGG": forecast_agg,
        "TEST_START": test_start,
        "TEST_END": test_end,
        "FC_START": fc_start,
        "FC_END": fc_end,
        "SELECTION_METRIC": selection_metric,
        "PRE_TRANSFORM": pre_transform,
        "RUN_MODELS": run_models or ALL_MODELS,
    }


def _series_rows(demand: pd.DataFrame, keys: List[str], keyvals: List[Any]) -> pd.DataFrame:
    mask = np.ones(len(demand), dtype=bool)
    for k, v in zip(keys, keyvals):
        if k == "sku_id":
            try:
                v = int(v)
            except Exception:
                pass
        mask &= (demand[k] == v)
    return demand.loc[mask]


def run_forecasting_job(
    ctx: ForecastContext,
    *,
    level: str,
    combinations_df: Optional[pd.DataFrame] = None,
    temporal_agg: str = "daily",
    forecast_agg: str = "weekly",
    test_months: Optional[Iterable[str]] = None,
    forecast_start_month: Optional[str] = None,
    forecast_end_month: Optional[str] = None,
    selection_metric: str = "weighted_accuracy",
    pre_transform: str = "log1p",
    run_models: Optional[List[str]] = None,
    fallback_top_n: int = 10,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, pd.DataFrame]:
    if level not in LEVEL_KEYS:
        raise ValueError(f"Unsupported level '{level}'. Choose one of: {list(LEVEL_KEYS)}")
    hist_months, _all_months, last_full, next_month = month_options(ctx)
    test_months = list(test_months or [last_full])
    forecast_start_month = forecast_start_month or next_month
    forecast_end_month = forecast_end_month or forecast_start_month
    cfg = build_cfg(
        temporal_agg, forecast_agg, test_months, forecast_start_month, forecast_end_month,
        selection_metric, pre_transform, run_models,
    )
    keys = LEVEL_KEYS[level]
    if combinations_df is not None and not combinations_df.empty:
        combos = load_combinations(combinations_df, level, keys, ctx.classif[level])
        if "_found" in combos.columns:
            combos = combos[combos["_found"]].reset_index(drop=True)
    else:
        combos = combinations_template(level, keys, ctx.classif[level], top=fallback_top_n)

    bt, lb, fcast = [], [], []
    n = len(combos)
    for idx, (_, r) in enumerate(combos.iterrows(), 1):
        keyvals = [r[k] for k in keys]
        label = level + "|" + "|".join(str(v) for v in keyvals)
        if progress_callback:
            progress_callback(idx, n, label)
        sub = _series_rows(ctx.demand, keys, keyvals)
        if sub.empty:
            continue
        daily = build_daily_series(sub)
        meta = {"level": level, **{k: r[k] for k in keys}, "segment": r.get("segment"), "ABC": r.get("ABC")}
        res = run_series(daily, label, meta, cfg, ctx.holidays, ctx.regular_holidays, ctx.special_holidays)
        if res:
            bt.append(res["backtest"])
            lb.append(res["leaderboard"])
            if len(res["forecast"]):
                fcast.append(res["forecast"])

    backtest = pd.concat(bt, ignore_index=True) if bt else pd.DataFrame()
    leader = pd.concat(lb, ignore_index=True) if lb else pd.DataFrame()
    forecasts = pd.concat(fcast, ignore_index=True) if fcast else pd.DataFrame()
    champions = leader[leader.is_champion].copy().sort_values("test_volume", ascending=False) if not leader.empty else pd.DataFrame()
    forecast_champion = forecasts[forecasts.is_champion].copy() if not forecasts.empty else pd.DataFrame()

    run_cfg = pd.DataFrame({
        "parameter": ["level", "temporal_agg", "forecast_agg", "test_window", "forecast_window",
                      "select_metric", "pre_transform", "n_combinations", "data_range", "models"],
        "value": [level, temporal_agg, forecast_agg,
                  f"{cfg['TEST_START'].date()}..{cfg['TEST_END'].date()}",
                  f"{cfg['FC_START'].date()}..{cfg['FC_END'].date()}", selection_metric,
                  pre_transform, len(combos),
                  f"{ctx.demand.date.min().date()}..{ctx.ref_date.date()}", ", ".join(cfg["RUN_MODELS"])]
    })
    return {
        "Run_Config": run_cfg,
        "Combinations": combos,
        "Champions": champions,
        "Leaderboard": leader,
        "Forecast_Champion": forecast_champion,
        "Forecast_AllModels": forecasts,
        "Backtest_Detail": backtest,
        "Classification": ctx.classif[level],
    }


def export_results_excel(results: Dict[str, pd.DataFrame]) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as xl:
        for sheet, df in results.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                df.round(4).to_excel(xl, sheet_name=sheet[:31], index=False)
            elif sheet == "Run_Config" and isinstance(df, pd.DataFrame):
                df.to_excel(xl, sheet_name=sheet[:31], index=False)
    output.seek(0)
    return output.getvalue()


def summarize_context(ctx: ForecastContext) -> Dict[str, Any]:
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    return {
        "rows": len(ctx.demand),
        "date_min": ctx.demand["date"].min().date(),
        "date_max": ctx.ref_date.date(),
        "warehouses": ctx.demand["warehouse_id"].nunique(),
        "skus": ctx.demand["sku_id"].nunique(),
        "channels": ctx.demand["channel"].nunique() if "channel" in ctx.demand.columns else 0,
        "trading_weekdays": [day_names[d] for d in ctx.trading_dow],
        "closed_weekdays": [day_names[d] for d in ctx.closed_dows],
        "series_counts": {lv: len(df) for lv, df in ctx.classif.items()},
    }
