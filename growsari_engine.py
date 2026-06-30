"""
GrowSari Demand Forecasting — shared engine (v4)
=================================================
Used by BOTH the Colab notebook and the Streamlit app, so behaviour never
drifts between the two. Changes vs v3:

  1. Dynamic validation for any uploaded demand / sku-master / combinations
     file: required columns, dtypes, duplicate keys, date parsing, empty
     frames, unknown aggregation-level keys, etc. Every check raises a
     `ValidationError` with a human-readable message — nothing fails with a
     raw pandas KeyError/TypeError several cells later.
  2. S3 support: any of the three input files can be a local path/upload OR
     an `s3://bucket/key.csv` URL. Credentials are read from environment
     variables / Streamlit secrets — NEVER hardcoded here. Large files are
     streamed instead of loaded fully into memory upfront.
  3. Custom "week" calendar: each month is sliced into 6 buckets of (at most)
     5 days — day 1-5, 6-10, 11-15, 16-20, 21-25, 26-end-of-month — labelled
     by the bucket's LAST day (5, 10, 15, 20, 25, month-end), matching the
     client's requested output format, e.g. 2026-03-05 ... 2026-03-31,
     2026-04-05 ... 2026-04-30. This replaces the old ISO/W-MON weekly grain
     everywhere (resampling, backtest bucketing, forecast horizon, exports).
"""
from __future__ import annotations

import io
import os
import re
import warnings
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


# ===================================================================== #
#  ERRORS
# ===================================================================== #
class ValidationError(Exception):
    """Raised for any user-data problem we can explain in plain English."""


# ===================================================================== #
#  1. S3 / FILE LOADING  (works for local path, file-like upload, or s3://)
# ===================================================================== #
def is_s3_url(path) -> bool:
    return isinstance(path, str) and path.strip().lower().startswith("s3://")


def parse_s3_url(url: str):
    m = re.match(r"^s3://([^/]+)/(.+)$", url.strip(), re.IGNORECASE)
    if not m:
        raise ValidationError(f"'{url}' is not a valid s3:// URL. Expected format: "
                               f"s3://bucket-name/path/to/file.csv")
    return m.group(1), m.group(2)


def get_s3_client(access_key=None, secret_key=None, region=None):
    """
    Build a boto3 S3 client. Credentials are resolved in this priority order
    (never hardcode keys in code/notebooks):
      1. explicit args passed in (e.g. from a Streamlit secrets/sidebar field)
      2. environment variables AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_REGION
      3. the default boto3 credential chain (IAM role, ~/.aws/credentials, etc.)
    """
    try:
        import boto3
    except ImportError as e:
        raise ValidationError(
            "boto3 is not installed. Run: pip install boto3"
        ) from e

    access_key = access_key or os.environ.get("AWS_ACCESS_KEY_ID")
    secret_key = secret_key or os.environ.get("AWS_SECRET_ACCESS_KEY")
    region = region or os.environ.get("AWS_REGION", "ap-south-1")

    kwargs = {"region_name": region}
    if access_key and secret_key:
        kwargs["aws_access_key_id"] = access_key
        kwargs["aws_secret_access_key"] = secret_key
    # else: fall back to boto3's default credential discovery
    return boto3.client("s3", **kwargs)


def read_table_from_s3(url: str, access_key=None, secret_key=None, region=None,
                        nrows: Optional[int] = None) -> pd.DataFrame:
    """
    Stream a CSV/XLSX directly from S3 without downloading the whole object
    into memory first (important for "big dataset" files). Falls back to a
    full in-memory read only for Excel (openpyxl needs the full file).
    """
    bucket, key = parse_s3_url(url)
    s3 = get_s3_client(access_key, secret_key, region)
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
    except Exception as e:
        raise ValidationError(
            f"Could not read s3://{bucket}/{key} — check the path, bucket region "
            f"('{region}'), and that the credentials have s3:GetObject permission. "
            f"Underlying error: {e}"
        )
    body = obj["Body"]
    if key.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(body.read()))
    # CSV / TSV: stream in chunks so multi-GB files don't blow up memory
    chunks = []
    reader = pd.read_csv(body, chunksize=250_000, low_memory=False)
    total = 0
    for chunk in reader:
        chunks.append(chunk)
        total += len(chunk)
        if nrows and total >= nrows:
            break
    if not chunks:
        raise ValidationError(f"s3://{bucket}/{key} loaded but contained no rows.")
    df = pd.concat(chunks, ignore_index=True)
    return df.head(nrows) if nrows else df


def load_any_table(source, file_kind: str, access_key=None, secret_key=None,
                    region=None) -> pd.DataFrame:
    """
    Universal loader: `source` may be
      - an s3:// URL string                       -> streamed from S3
      - a local path string                        -> read from disk
      - a file-like / UploadedFile (Streamlit, ipywidgets bytes) -> read from memory
    `file_kind` is only used for error messages ("demand history", "SKU master",
    "combinations list").
    """
    if source is None:
        raise ValidationError(f"No {file_kind} file was provided.")

    try:
        if is_s3_url(source):
            return read_table_from_s3(source, access_key, secret_key, region)

        if isinstance(source, str):
            if not os.path.exists(source):
                raise ValidationError(f"Could not find the {file_kind} file at path: {source}")
            return pd.read_excel(source) if source.lower().endswith((".xlsx", ".xls")) \
                else pd.read_csv(source, low_memory=False)

        # file-like (Streamlit UploadedFile / BytesIO / ipywidgets bytes)
        name = getattr(source, "name", "") or ""
        data = source.read() if hasattr(source, "read") else source
        if isinstance(data, str):
            data = data.encode()
        if name.lower().endswith((".xlsx", ".xls")):
            return pd.read_excel(io.BytesIO(data))
        return pd.read_csv(io.BytesIO(data), low_memory=False)

    except ValidationError:
        raise
    except Exception as e:
        raise ValidationError(f"Failed to read the {file_kind} file: {e}")


# ===================================================================== #
#  2. DYNAMIC VALIDATION  (columns, dtypes, referencing)
# ===================================================================== #
# Common SKU header variants, reused everywhere so the three maps can't drift.
_SKU_ALIASES = [
    "sku_id", "sku", "skuid", "sku id", "sku code", "sku_code", "skucode",
    "code", "item", "item_id", "item id", "item code", "item_code",
    "material", "material_code", "material code", "product_code",
    "product code", "product_id", "product id", "article", "article_code",
]
REQUIRED_DEMAND_COLS_ALIASES = {
    "warehouse_id": ["warehouse_id", "warehouse", "wh", "warehouse id",
                     "depot", "depot_id", "depot id", "dc", "dc_id", "location"],
    "sku_id":       list(_SKU_ALIASES),
    "date":         ["date", "demand_date", "txn_date", "order_date",
                     "transaction_date", "day", "ds"],
    "y":            ["y", "demand_units", "units", "qty", "quantity",
                     "demand", "sales", "sales_units", "volume"],
}
OPTIONAL_DEMAND_COLS_ALIASES = {
    "channel":      ["channel", "chan", "sales_channel"],
    "availability": ["availability", "availability_score"],
}
REQUIRED_SKU_COLS_ALIASES = {
    "sku_id":   list(_SKU_ALIASES),
    "name":     ["name", "sku_name", "product_name", "description", "item_name"],
    "category": ["category", "cat", "product_category", "category_name"],
}
COL_ALIASES = {
    "warehouse_id": ["warehouse_id", "warehouse", "wh", "warehouse id",
                     "depot", "depot_id", "depot id", "dc", "dc_id"],
    "sku_id":       list(_SKU_ALIASES),
    "category":     ["category", "cat", "product category", "product_category"],
    "channel":      ["channel", "chan", "sales_channel"],
}


def _coerce_sku_id(s: pd.Series) -> pd.Series:
    """Normalise a SKU identifier column. Numeric codes become clean integers
    (so 1234.0 -> 1234); anything alphanumeric (e.g. 'ABC-12') is kept as a
    trimmed string. Returning a consistent rule everywhere keeps merge keys
    type-aligned between demand, SKU master, and combinations files."""
    s = s.astype(str).str.strip()
    num = pd.to_numeric(s, errors="coerce")
    if len(num) and num.notna().all() and (num.dropna() % 1 == 0).all():
        return num.astype("int64")
    return s


def _normalize_columns(df: pd.DataFrame, alias_map: dict) -> tuple[pd.DataFrame, dict]:
    """Lower/strip headers, then rename any recognised alias -> canonical name.
    Returns (renamed_df, {canonical: original_header_used_or_None})."""
    out = df.copy()
    # Drop stray index columns that pandas writes to CSVs ("Unnamed: 0", etc.)
    junk = [c for c in out.columns if str(c).strip().lower().startswith("unnamed:")
            or str(c).strip() == ""]
    if junk:
        out = out.drop(columns=junk)
    lower = {str(c).lower().strip(): c for c in out.columns}
    rename, found = {}, {}
    for canon, alts in alias_map.items():
        hit = next((lower[a] for a in alts if a in lower), None)
        found[canon] = hit
        if hit:
            rename[hit] = canon
    return out.rename(columns=rename), found


def validate_and_normalize_demand(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValidationError("The demand history file is empty.")
    df, found = _normalize_columns(df, {**REQUIRED_DEMAND_COLS_ALIASES, **OPTIONAL_DEMAND_COLS_ALIASES})
    missing = [c for c in REQUIRED_DEMAND_COLS_ALIASES if not found.get(c)]
    if missing:
        raise ValidationError(
            f"Demand history file is missing required column(s): {missing}. "
            f"Found columns: {list(df.columns)}. "
            f"Accepted header variants: { {k: v for k, v in REQUIRED_DEMAND_COLS_ALIASES.items() if k in missing} }"
        )
    if "availability" not in df.columns:
        df["availability"] = 1.0
    if "channel" not in df.columns:
        df["channel"] = "ALL"

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    n_bad_dates = df["date"].isna().sum()
    if n_bad_dates:
        warnings.warn(f"{n_bad_dates} row(s) had an unparseable date and were dropped.")
        df = df.dropna(subset=["date"])
    if df.empty:
        raise ValidationError("After parsing dates, no valid demand rows remained.")

    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    n_bad_y = df["y"].isna().sum()
    if n_bad_y:
        warnings.warn(f"{n_bad_y} row(s) had a non-numeric demand value and were set to 0.")
        df["y"] = df["y"].fillna(0.0)
    if (df["y"] < 0).any():
        warnings.warn("Negative demand values found — clipped to 0.")
        df["y"] = df["y"].clip(lower=0)

    df["sku_id"] = _coerce_sku_id(df["sku_id"])
    # drop rows with an empty/blank SKU identifier (string-coded SKUs only)
    if not pd.api.types.is_numeric_dtype(df["sku_id"]):
        blank = df["sku_id"].astype(str).str.strip().isin(["", "nan", "none", "None", "NaN", "<NA>"])
        if blank.any():
            warnings.warn(f"{int(blank.sum())} row(s) had a blank sku_id and were dropped.")
            df = df[~blank]
    df["warehouse_id"] = df["warehouse_id"].astype(str).str.strip()
    return df


def validate_and_normalize_sku_master(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValidationError("The SKU master file is empty.")
    df, found = _normalize_columns(df, REQUIRED_SKU_COLS_ALIASES)
    missing = [c for c in REQUIRED_SKU_COLS_ALIASES if not found.get(c)]
    if missing:
        raise ValidationError(
            f"SKU master file is missing required column(s): {missing}. "
            f"Found columns: {list(df.columns)}."
        )
    df["sku_id"] = _coerce_sku_id(df["sku_id"])
    dupes = df["sku_id"].duplicated().sum()
    if dupes:
        warnings.warn(f"{dupes} duplicate sku_id row(s) in SKU master — keeping the first.")
        df = df.drop_duplicates(subset="sku_id", keep="first")
    return df[["sku_id", "name", "category"]]


def merge_demand_with_sku(demand: pd.DataFrame, skum: pd.DataFrame) -> pd.DataFrame:
    # Guard against an int-vs-string key mismatch, which would silently match nothing.
    if demand["sku_id"].dtype != skum["sku_id"].dtype:
        demand = demand.copy(); skum = skum.copy()
        demand["sku_id"] = demand["sku_id"].astype(str)
        skum["sku_id"] = skum["sku_id"].astype(str)
    merged = demand.merge(skum, on="sku_id", how="left")
    unmatched = merged["category"].isna().sum()
    if unmatched:
        warnings.warn(
            f"{unmatched} demand row(s) reference a sku_id not present in the SKU master "
            f"and will show category/name as missing. Check for typos or a stale master file."
        )
        merged["category"] = merged["category"].fillna("UNMAPPED")
        merged["name"] = merged["name"].fillna("UNMAPPED")
    return merged


def validate_level_keys(level: str, level_keys: dict, demand: pd.DataFrame):
    if level not in level_keys:
        raise ValidationError(f"Unknown aggregation level '{level}'. Choose one of: {list(level_keys)}")
    keys = level_keys[level]
    missing = [k for k in keys if k not in demand.columns]
    if missing:
        raise ValidationError(
            f"Aggregation level '{level}' needs column(s) {keys}, but the demand data is "
            f"missing: {missing}. Available columns: {list(demand.columns)}"
        )


def validate_combinations_upload(combos_df: pd.DataFrame, level: str, level_keys: list,
                                  demand: pd.DataFrame, classification_df=None) -> pd.DataFrame:
    if combos_df is None or combos_df.empty:
        raise ValidationError("The uploaded combinations file is empty.")
    df, found = _normalize_columns(combos_df, COL_ALIASES)
    missing = [k for k in level_keys if not found.get(k)]
    if missing:
        raise ValidationError(
            f"Combinations file for level '{level}' is missing column(s): {missing}. "
            f"Required columns for this level: {level_keys}. Found: {list(combos_df.columns)}"
        )
    df = df[level_keys].dropna().drop_duplicates().reset_index(drop=True)
    if df.empty:
        raise ValidationError("Combinations file has the right columns but no usable rows after cleaning.")

    if "sku_id" in df.columns:
        df["sku_id"] = _coerce_sku_id(df["sku_id"])

    # cross-reference against the actual demand data so typos surface immediately
    valid_keys = demand[level_keys].drop_duplicates()
    # align key dtypes so the validation merge can't silently drop everything
    for k in level_keys:
        if df[k].dtype != valid_keys[k].dtype:
            df[k] = df[k].astype(str)
            valid_keys = valid_keys.copy()
            valid_keys[k] = valid_keys[k].astype(str)
    df = df.merge(valid_keys.assign(_found=True), on=level_keys, how="left")
    df["_found"] = df["_found"].fillna(False)

    if classification_df is not None and not classification_df.empty:
        meta_cols = [c for c in ["segment", "ABC", "lifecycle", "total_units"] if c in classification_df.columns]
        if meta_cols:
            df = df.merge(classification_df[level_keys + meta_cols], on=level_keys, how="left")
    return df


def validate_date_window(label: str, months: list, available_months: list):
    if not months:
        raise ValidationError(f"No {label} month(s) selected.")
    unknown = [m for m in months if m not in available_months]
    if unknown and label == "test":
        raise ValidationError(
            f"Test month(s) {unknown} are not present in the uploaded demand history "
            f"(available: {available_months[0]}..{available_months[-1]})."
        )


# ===================================================================== #
#  3. PHILIPPINE HOLIDAY CALENDAR  (unchanged from v3)
# ===================================================================== #
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
    out = {}
    for y in range(year_min, year_max + 1):
        for mmdd, nm in _FIXED_REGULAR.items():
            out[pd.Timestamp(f"{y}-{mmdd}")] = (nm, "regular")
        for mmdd, nm in _FIXED_SPECIAL.items():
            out[pd.Timestamp(f"{y}-{mmdd}")] = (nm, "special")
        out[_last_monday_august(y)] = ("National Heroes Day", "regular")
        for ds, nm in _MOVABLE.get(y, {}).items():
            if nm in ("Maundy Thursday", "Good Friday", "Eidul Fitr", "Eidul Adha"):
                kind = "regular"
            else:
                kind = "special"
            out[pd.Timestamp(ds)] = (nm, kind)
    return out


def trading_days_between(start, end, closed_dows=()):
    rng = pd.date_range(start, end, freq="D")
    return int(sum(d.dayofweek not in closed_dows for d in rng))


# ===================================================================== #
#  4. ADAPTIVE DEMAND CLASSIFICATION  (unchanged logic from v3)
# ===================================================================== #
DEFAULT_PARAMS = dict(
    abc_a=0.80, abc_b=0.95,
    adi_cut=1.32, cv2_cut=0.49,
    new_history_days=60,
    dormant_recency_days=28,
    closed_dows=(),
    seasonality_min_cycles=2,
)


def _series_metrics(daily_y, dates, ref_date, params):
    demand_days = int((daily_y > 0).sum())
    nonzero = daily_y[daily_y > 0]
    total = float(daily_y.sum())
    first_demand = dates[daily_y > 0].min() if demand_days > 0 else dates.min()
    history_td = trading_days_between(first_demand, ref_date, params["closed_dows"])
    adi = history_td / demand_days if demand_days > 0 else np.inf
    cv2 = float((nonzero.std(ddof=0) / nonzero.mean()) ** 2) if demand_days > 1 and nonzero.mean() > 0 else 0.0
    last_demand = dates[daily_y > 0].max() if demand_days > 0 else None
    recency_td = trading_days_between(last_demand, ref_date, params["closed_dows"]) - 1 if last_demand is not None else 9999
    coverage = demand_days / history_td if history_td > 0 else 0.0
    return dict(total_units=total, demand_days=demand_days, history_td=history_td,
                recency_td=max(recency_td, 0), coverage=coverage, adi=adi, cv2=cv2)


def _seasonality_trend(y, period):
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
    if adi <= p["adi_cut"] and cv2 > p["cv2_cut"]: return "Erratic"
    if adi > p["adi_cut"] and cv2 <= p["cv2_cut"]: return "Intermittent"
    return "Lumpy"


def _recommend_method(seg, has_yoy, s_strength):
    seasonal = (has_yoy and (s_strength or 0) >= 0.3)
    table = {
        "Smooth":       ("ETS/LightGBM + seasonal features" if seasonal else "ETS/LightGBM + day-of-week", "daily"),
        "Erratic":      ("Quantile GBM / ETS damped on custom weekly buckets", "weekly"),
        "Intermittent": ("Croston / SBA / TSB", "weekly"),
        "Lumpy":        ("TSB / empirical resampling", "weekly/monthly"),
        "Dormant":      ("Exclude from active forecasting", "—"),
        "New":          ("Category-proxy disaggregation", "daily proxy"),
    }
    return table.get(seg, ("ETS", "daily"))


LEVEL_KEYS = {
    "WH_SKU":              ["warehouse_id", "sku_id"],
    "WH_Category":         ["warehouse_id", "category"],
    "WH_SKU_Channel":      ["warehouse_id", "sku_id", "channel"],
    "WH_Category_Channel": ["warehouse_id", "category", "channel"],
}


def classify_level(demand, group_keys, ref_date, params=None, seasonal_period_daily=None):
    p = {**DEFAULT_PARAMS, **(params or {})}
    ref_date = pd.Timestamp(ref_date)
    span_start, span_end = demand["date"].min(), ref_date
    hist_days = (span_end - span_start).days
    if hist_days >= 540:
        p["new_history_days"] = max(p["new_history_days"], 90)
        p["dormant_recency_days"] = max(p["dormant_recency_days"], 42)
    has_yoy = hist_days >= 700
    sp = seasonal_period_daily or (7 - len(p["closed_dows"]))

    g = demand.groupby(group_keys + ["date"])["y"].sum().reset_index()
    rows = []
    for key, sub in g.groupby(group_keys):
        key = key if isinstance(key, tuple) else (key,)
        s = sub.set_index("date")["y"].sort_index()
        dates = s.index
        m = _series_metrics(s.values, dates, ref_date, p)
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
    if out.empty:
        return out
    out = out.sort_values("total_units", ascending=False).reset_index(drop=True)
    grand = out["total_units"].sum()
    out["units_share"] = out["total_units"] / grand if grand > 0 else 0
    out["cum_share"] = out["units_share"].cumsum()
    prev = out["cum_share"] - out["units_share"]
    out["ABC"] = np.where(prev < p["abc_a"], "A", np.where(prev < p["abc_b"], "B", "C"))
    out["coverage_pct"] = (out["coverage"] * 100).round(1)
    out["adi"] = out["adi"].round(3)
    out["cv2"] = out["cv2"].round(4)
    return out


def classify_all(demand, ref_date, params=None, levels=None):
    levels = levels or list(LEVEL_KEYS)
    return {lv: classify_level(demand, LEVEL_KEYS[lv], ref_date, params) for lv in levels}


def detect_closed_dows(demand_df, rel_threshold=0.02):
    by = demand_df.groupby(demand_df["date"].dt.dayofweek)["y"].sum().reindex(range(7), fill_value=0.0)
    open_med = by[by > 0].median() if (by > 0).any() else 0.0
    return tuple(int(d) for d in range(7) if by[d] <= rel_threshold * open_med)


MIN_OBS_ML = {"daily": 24, "weekly": 12, "monthly": 18}
GRAIN_ORDER = {"daily": 0, "weekly": 1, "monthly": 2}


# ===================================================================== #
#  5. CUSTOM WEEK CALENDAR  ⭐ (the requested change)
#  Each month -> 6 fixed buckets: [1-5][6-10][11-15][16-20][21-25][26-EOM]
#  A bucket is LABELLED by its last calendar day (5,10,15,20,25,EOM), e.g.
#  2026-03-05, 2026-03-10, ..., 2026-03-31, 2026-04-05, ...
# ===================================================================== #
_CUSTOM_WEEK_CUTS = (5, 10, 15, 20, 25)  # the 6th bucket always runs 26->month-end


def custom_week_end(d) -> pd.Timestamp:
    """Last calendar day of the 5-day bucket containing date `d`."""
    d = pd.Timestamp(d)
    day = d.day
    for cut in _CUSTOM_WEEK_CUTS:
        if day <= cut:
            return pd.Timestamp(year=d.year, month=d.month, day=cut)
    eom = (d + pd.offsets.MonthEnd(0)).day
    return pd.Timestamp(year=d.year, month=d.month, day=eom)


def custom_week_start(d) -> pd.Timestamp:
    """First calendar day of the 5-day bucket containing date `d`."""
    d = pd.Timestamp(d)
    day = d.day
    prev_cut = 0
    for cut in _CUSTOM_WEEK_CUTS:
        if day <= cut:
            return pd.Timestamp(year=d.year, month=d.month, day=prev_cut + 1)
        prev_cut = cut
    return pd.Timestamp(year=d.year, month=d.month, day=26)


def custom_week_series(ds: pd.Series) -> pd.Series:
    """Vectorised bucket-END label for a whole datetime Series."""
    return ds.map(custom_week_end)


def custom_week_range(start, end) -> pd.DatetimeIndex:
    """All custom-week END labels from `start`'s bucket through `end`'s bucket, inclusive."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    cur_month = pd.Timestamp(start.year, start.month, 1)
    end_month = pd.Timestamp(end.year, end.month, 1)
    ends = []
    while cur_month <= end_month:
        eom = (cur_month + pd.offsets.MonthEnd(0)).day
        month_cuts = list(_CUSTOM_WEEK_CUTS) + [eom]
        for cut in month_cuts:
            ends.append(pd.Timestamp(cur_month.year, cur_month.month, cut))
        cur_month = cur_month + pd.DateOffset(months=1)
    idx = pd.DatetimeIndex(sorted(set(ends)))
    return idx[(idx >= custom_week_end(start)) & (idx <= custom_week_end(end))]


# ===================================================================== #
#  6. SERIES BUILD + TEMPORAL RESAMPLE (weekly now = custom week)
# ===================================================================== #
def build_daily_series(sub, trading_dow):
    s = sub.groupby("date").agg(y=("y", "sum"), availability=("availability", "mean")).reset_index()
    full = pd.date_range(s["date"].min(), s["date"].max(), freq="D")
    full = full[[d.dayofweek in trading_dow for d in full]]
    out = pd.DataFrame({"ds": full}).merge(s.rename(columns={"date": "ds"}), on="ds", how="left")
    out["y"] = out["y"].fillna(0.0)
    out["availability"] = out["availability"].fillna(1.0)
    return out


def resample_grain(daily, grain, holidays):
    d = daily.copy()
    if grain == "daily":
        d["n_trading"] = 1
        d["n_holiday"] = d["ds"].isin(holidays).astype(int)
        return d
    if grain == "weekly":
        d["bucket"] = custom_week_series(d["ds"])          # <-- custom 5-day week
    else:  # monthly
        d["bucket"] = d["ds"].values.astype("datetime64[M]")
    agg = (d.groupby("bucket")
           .agg(y=("y", "sum"), availability=("availability", "mean"),
                n_trading=("y", "size"),
                n_holiday=("ds", lambda x: sum(t in holidays for t in x)))
           .reset_index().rename(columns={"bucket": "ds"}).sort_values("ds"))
    return agg


def grain_to_bucket(ds_series, to_grain):
    if to_grain == "daily":
        return ds_series
    if to_grain == "weekly":
        return custom_week_series(ds_series)                # <-- custom 5-day week
    return ds_series.values.astype("datetime64[M]")


def seasonal_period(grain, history_len, trading_dow):
    base = {"daily": len(trading_dow), "weekly": 6, "monthly": 12}[grain]  # 6 custom-weeks/month
    return base if history_len >= 2 * base + 1 else None


def validate_grains(temporal, forecast):
    if GRAIN_ORDER[forecast] < GRAIN_ORDER[temporal]:
        raise ValidationError(f"FORECAST_AGG='{forecast}' must be >= TEMPORAL_AGG='{temporal}' "
                               "(daily < weekly < monthly).")


def _future_periods(last_ds, grain, fc_end, trading_dow):
    start = last_ds + pd.Timedelta(days=1)
    fc_end = pd.Timestamp(fc_end)
    if grain == "daily":
        rng = pd.date_range(start, fc_end, freq="D")
        return pd.DatetimeIndex([d for d in rng if d.dayofweek in trading_dow])
    if grain == "weekly":
        return custom_week_range(start, fc_end)              # <-- custom 5-day week
    return pd.date_range(start.replace(day=1), fc_end, freq="MS")


def _bucket_end(period_start, fgrain):
    p = pd.Timestamp(period_start)
    if fgrain == "daily":
        return p
    if fgrain == "weekly":
        return custom_week_end(p)                            # already the end label
    return (p + pd.DateOffset(months=1)) - pd.Timedelta(days=1)


def _bucket_start(period_start, fgrain):
    p = pd.Timestamp(period_start)
    if fgrain == "daily":
        return p
    if fgrain == "weekly":
        return custom_week_start(p)
    return p.replace(day=1)


def _in_window(period_start, fgrain, win_start, win_end):
    ps = _bucket_start(period_start, fgrain)
    pe = _bucket_end(period_start, fgrain)
    return (pe >= pd.Timestamp(win_start)) and (ps <= pd.Timestamp(win_end))


def _drop_trailing_partial(g, grain):
    if grain == "weekly" and len(g) and g["n_trading"].iloc[-1] < 4:
        g = g.iloc[:-1]
    if grain == "monthly" and len(g) and g["n_trading"].iloc[-1] < 20:
        g = g.iloc[:-1]
    return g.reset_index(drop=True)


# ===================================================================== #
#  7. FEATURE ENGINEERING (grain-aware; weekly now uses custom week-of-month)
# ===================================================================== #
def add_features(df, grain, holidays, reg_dates, spc_dates):
    d = df.copy().reset_index(drop=True)
    ds = d["ds"]
    if grain == "daily":
        d["dow"] = ds.dt.dayofweek; d["day"] = ds.dt.day; d["month"] = ds.dt.month
        d["weekofmonth"] = (ds.dt.day - 1) // 7 + 1
        d["is_payday"] = ds.dt.day.isin(list(range(13, 17)) + list(range(29, 32)) + [1, 2]).astype(int)
        d["is_holiday"] = ds.isin(holidays).astype(int)
        d["is_reg_hol"] = ds.isin(reg_dates).astype(int)
        d["is_spc_hol"] = ds.isin(spc_dates).astype(int)
        hs = sorted(holidays)
        d["pre_hol"] = [1 if any(0 < (h - t).days <= 3 for h in hs) else 0 for t in ds]
        d["post_hol"] = [1 if any(0 < (t - h).days <= 2 for h in hs) else 0 for t in ds]
        lags, rolls = (1, 2, 3, 6, 12, 18), (3, 6, 12, 24)
        seas = "dow"
    elif grain == "weekly":
        # custom week: bucket index 1-6 within its month (1=days1-5 ... 6=26-EOM)
        d["weekofmonth"] = ds.map(lambda x: min(((x.day - 1) // 5) + 1, 6))
        d["month"] = ds.dt.month
        lags, rolls = (1, 2, 3, 6, 12), (2, 3, 6)
        seas = "weekofmonth"
    else:
        d["month"] = ds.dt.month; d["quarter"] = ds.dt.quarter
        lags, rolls = (1, 2, 3, 12), (2, 3, 6)
        seas = "month"
    for L in lags:
        d[f"lag_{L}"] = d["y"].shift(L)
    for R in rolls:
        d[f"rmean_{R}"] = d["y"].shift(1).rolling(R, min_periods=1).mean()
        d[f"rstd_{R}"] = d["y"].shift(1).rolling(R, min_periods=2).std()
    d["ewm"] = d["y"].shift(1).ewm(span=max(rolls[1], 3), min_periods=1).mean()
    d["seas_expmean"] = (d.groupby(seas)["y"]
                          .apply(lambda s: s.shift(1).expanding(min_periods=1).mean())
                          .reset_index(level=0, drop=True))
    d["t_index"] = np.arange(len(d))
    feat_cols = [c for c in d.columns if c not in ("ds", "y", "availability")]
    return d, feat_cols


# ===================================================================== #
#  8. MODEL LIBRARY
# ===================================================================== #
def _inv(x, use_log): return np.expm1(x) if use_log else x
def _fwd(x, use_log): return np.log1p(np.clip(x, 0, None)) if use_log else x


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
        return np.array([max(0.0, dm.get(dt.dayofweek, recent) * 0.7 + recent * 0.3) for dt in fdates])
    return np.repeat(max(0.0, train["y"].tail(4).mean()), n)


def m_ets(train, n, sp=None, use_log=False, **k):
    y = train["y"].astype(float).values
    yt = _fwd(y, use_log)
    if len(y) < 6 or (y > 0).sum() < 3:
        return np.repeat(max(0.0, np.mean(y[-3:]) if len(y) else 0), n)
    try:
        seasonal = "add" if (sp and len(y) >= 2 * sp + 1 and (y > 0).sum() >= sp) else None
        fit = ExponentialSmoothing(np.where(yt <= 0, 1e-6, yt) if not use_log else yt,
                                    trend="add", damped_trend=True,
                                    seasonal=seasonal, seasonal_periods=sp if seasonal else None,
                                    initialization_method="estimated").fit()
        fc = _inv(fit.forecast(n), use_log)
        return np.clip(np.nan_to_num(fc, nan=np.median(y)), 0, None)
    except Exception:
        return np.repeat(max(0.0, np.mean(y[-3:])), n)


def _croston_core(y, n, alpha=0.1, variant="classic"):
    y = np.asarray(y, float)
    nz = np.where(y > 0)[0]
    if len(nz) == 0:
        return np.zeros(n)
    z = y[nz[0]]; p = 1.0; q = 1
    for i in range(nz[0] + 1, len(y)):
        if y[i] > 0:
            z += alpha * (y[i] - z); p += alpha * (q - p); q = 1
        else:
            q += 1
    if variant == "sba":
        f = (1 - alpha / 2) * z / p
    elif variant == "tsb":
        prob = 0.0; lvl = z
        for i in range(len(y)):
            occ = 1.0 if y[i] > 0 else 0.0
            prob += alpha * (occ - prob)
            if y[i] > 0:
                lvl += alpha * (y[i] - lvl)
        f = prob * lvl
    else:
        f = z / p
    return np.repeat(max(0.0, f), n)


def m_croston(train, n, **k): return _croston_core(train["y"].values, n, variant="classic")
def m_sba(train, n, **k): return _croston_core(train["y"].values, n, variant="sba")
def m_tsb(train, n, **k): return _croston_core(train["y"].values, n, variant="tsb")


def _new_ml(kind):
    if kind == "xgb":
        return XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.9,
                             colsample_bytree=0.9, min_child_weight=3, random_state=42,
                             n_jobs=2, verbosity=0)
    return LGBMRegressor(n_estimators=300, max_depth=6, num_leaves=31, learning_rate=0.05,
                          subsample=0.9, colsample_bytree=0.9, min_child_samples=5,
                          random_state=42, n_jobs=2, verbose=-1)


def m_ml(train_hist, fut_index, kind, grain, holidays, reg, spc, use_log=False, avail_future=1.0):
    work = train_hist[["ds", "y", "availability"]].copy()
    df, fcols = add_features(work, grain, holidays, reg, spc)
    lag_cols = [c for c in fcols if c.startswith("lag_")]
    tr = df.dropna(subset=lag_cols[-1:]) if lag_cols else df
    if len(tr) < 15:
        tr = df.fillna(0)
    Xy = tr.copy(); Xy["y"] = _fwd(Xy["y"].values, use_log)
    mdl = _new_ml(kind); mdl.fit(Xy[fcols].fillna(0), Xy["y"])
    preds, cur = [], work.copy()
    for _, r in fut_index.iterrows():
        cur = pd.concat([cur, pd.DataFrame({"ds": [r["ds"]], "y": [np.nan],
                                             "availability": [avail_future]})], ignore_index=True)
        f, _ = add_features(cur, grain, holidays, reg, spc)
        p = _inv(float(mdl.predict(f.iloc[[-1]][fcols].fillna(0))[0]), use_log)
        p = max(0.0, p); preds.append(p)
        cur.iloc[-1, cur.columns.get_loc("y")] = p
    return np.array(preds)


STAT_MODELS = {"Naive": m_naive, "SeasonalNaive": m_seasonal_naive, "MovingAvg": m_moving_avg,
               "SeasonalRollAvg": m_seasonal_rollavg, "ETS": m_ets,
               "Croston": m_croston, "SBA": m_sba, "TSB": m_tsb}
ML_MODELS = {"XGBoost": "xgb", "LightGBM": "lgb"}


def compute_metrics(a, p):
    a = np.asarray(a, float); p = np.asarray(p, float)
    err = p - a; abserr = np.abs(err)
    sum_a = a.sum()
    wape = abserr.sum() / sum_a if sum_a > 0 else np.nan
    nz = a > 0
    mape = np.mean(abserr[nz] / a[nz]) if nz.any() else np.nan
    smape = np.mean(2 * abserr / (np.abs(a) + np.abs(p) + 1e-9))
    mm = [min(x, y) / max(x, y) for x, y in zip(a, p) if max(x, y) > 0]
    agg_mm = (min(sum_a, p.sum()) / max(sum_a, p.sum())) if max(sum_a, p.sum()) > 0 else np.nan
    return dict(weighted_accuracy=(1 - wape) if wape == wape else np.nan,
                minmax_bucket=np.mean(mm) if mm else np.nan, minmax_agg=agg_mm,
                mape=mape, smape=smape, mae=abserr.mean(),
                rmse=np.sqrt((err ** 2).mean()), bias=err.mean(), test_volume=sum_a)


SELECT_DIR = {"weighted_accuracy": "max", "minmax_agg": "max", "minmax_bucket": "max",
              "mape": "min", "smape": "min", "mae": "min", "rmse": "min"}


# ===================================================================== #
#  9. ORCHESTRATOR
# ===================================================================== #
def run_series(daily_df, label, meta, cfg, holidays, reg, spc, trading_dow):
    grain, fgrain = cfg["TEMPORAL_AGG"], cfg["FORECAST_AGG"]
    use_log = cfg.get("PRE_TRANSFORM", "none") == "log1p"
    g = _drop_trailing_partial(resample_grain(daily_df, grain, holidays), grain)
    if g["y"].sum() == 0 or len(g) < 8:
        return None
    sp = seasonal_period(grain, len(g), trading_dow)
    test_start, test_end = pd.Timestamp(cfg["TEST_START"]), pd.Timestamp(cfg["TEST_END"])
    tr = g[g["ds"] < test_start].copy()
    te = g[(g["ds"] >= test_start) & (g["ds"] <= test_end)].copy()
    if len(te) < 1 or len(tr) < 6:
        return None

    models = cfg["RUN_MODELS"]; rows = []
    for nm in models:
        if nm in STAT_MODELS:
            p = STAT_MODELS[nm](tr, len(te), sp=sp, grain=grain, use_log=use_log, fdates=list(te["ds"]))
            for dt, pv, av in zip(te["ds"], p, te["y"]):
                rows.append((dt, nm, av, pv))
    ml_ok = len(tr) >= MIN_OBS_ML[grain]
    for nm, kind in ML_MODELS.items():
        if nm in models and ml_ok:
            p = m_ml(tr, te[["ds", "availability"]].assign(availability=1.0), kind, grain,
                      holidays, reg, spc, use_log=use_log)
            for dt, pv, av in zip(te["ds"], p, te["y"]):
                rows.append((dt, nm, av, pv))
    if not rows:
        return None
    md = pd.DataFrame(rows, columns=["ds", "model", "actual", "pred"])
    if "Ensemble" in models:
        ml_present = [m for m in ML_MODELS if m in md["model"].unique()]
        base = md[md.model.isin(ml_present)] if ml_present else md
        ens = base.groupby("ds").agg(actual=("actual", "first"), pred=("pred", "mean")).reset_index()
        ens["model"] = "Ensemble"
        md = pd.concat([md, ens[["ds", "model", "actual", "pred"]]], ignore_index=True)

    md["fbucket"] = grain_to_bucket(md["ds"], fgrain)
    roll = md.groupby(["model", "fbucket"]).agg(actual=("actual", "sum"), pred=("pred", "sum")).reset_index()
    lb_rows = []
    for nm, sgrp in roll.groupby("model"):
        met = compute_metrics(sgrp["actual"].values, sgrp["pred"].values)
        lb_rows.append({"series": label, **meta, "model": nm, **met})
    leaderboard = pd.DataFrame(lb_rows)
    metric = cfg["SELECTION_METRIC"]; direction = SELECT_DIR[metric]
    valid = leaderboard.dropna(subset=[metric]); valid = valid if not valid.empty else leaderboard
    champ = valid.sort_values(metric, ascending=(direction == "min")).iloc[0]["model"]
    leaderboard["is_champion"] = leaderboard["model"] == champ
    leaderboard["champion_model"] = champ
    roll = roll.rename(columns={"fbucket": "period"}); roll.insert(0, "series", label)

    gfull = _drop_trailing_partial(resample_grain(daily_df, grain, holidays), grain)
    fc_start, fc_end = pd.Timestamp(cfg["FC_START"]), pd.Timestamp(cfg["FC_END"])
    fperiods = _future_periods(gfull["ds"].max(), grain, fc_end, trading_dow)
    fc = pd.DataFrame(columns=["series", "model", "period", "forecast", "is_champion"])
    if len(fperiods):
        fut_idx = pd.DataFrame({"ds": fperiods, "availability": 1.0})
        spf = seasonal_period(grain, len(gfull), trading_dow); frows = []
        for nm in models:
            if nm in STAT_MODELS:
                p = STAT_MODELS[nm](gfull, len(fperiods), sp=spf, grain=grain, use_log=use_log, fdates=list(fperiods))
                for dt, pv in zip(fperiods, p):
                    frows.append((dt, nm, pv))
        ml_ok2 = len(gfull) >= MIN_OBS_ML[grain]
        for nm, kind in ML_MODELS.items():
            if nm in models and ml_ok2:
                p = m_ml(gfull, fut_idx, kind, grain, holidays, reg, spc, use_log=use_log)
                for dt, pv in zip(fperiods, p):
                    frows.append((dt, nm, pv))
        fcr = pd.DataFrame(frows, columns=["ds", "model", "forecast"])
        if "Ensemble" in models and not fcr.empty:
            ml_present = [m for m in ML_MODELS if m in fcr["model"].unique()]
            base = fcr[fcr.model.isin(ml_present)] if ml_present else fcr
            ensf = base.groupby("ds")["forecast"].mean().reset_index(); ensf["model"] = "Ensemble"
            fcr = pd.concat([fcr, ensf[["ds", "model", "forecast"]]], ignore_index=True)
        fcr["period"] = grain_to_bucket(fcr["ds"], fgrain)
        fcr = fcr.groupby(["model", "period"])["forecast"].sum().reset_index()
        keep = fcr["period"].apply(lambda ps: _in_window(ps, fgrain, fc_start, fc_end))
        fcr = fcr[keep].copy()
        fcr.insert(0, "series", label); fcr["is_champion"] = fcr["model"] == champ
        for kk, vv in meta.items():
            fcr[kk] = vv
        fc = fcr
    return dict(backtest=roll, leaderboard=leaderboard, forecast=fc, champion=champ)


def attach_output_calendar_columns(df: pd.DataFrame, date_col="period") -> pd.DataFrame:
    """Adds depot/sku/model/date_type/date/month/year-style helper columns the
    client asked for, matching the example output (month label like 'Mar')."""
    out = df.copy()
    out["date"] = pd.to_datetime(out[date_col]).dt.date
    out["month"] = pd.to_datetime(out[date_col]).dt.strftime("%b")
    out["year"] = pd.to_datetime(out[date_col]).dt.year
    return out
