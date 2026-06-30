"""
GrowSari Demand Forecasting — Streamlit app (v4)
Run with:  streamlit run streamlit_app.py
"""
import io
import time

import numpy as np
import pandas as pd
import streamlit as st

import growsari_engine as eng
from growsari_engine import ValidationError

st.set_page_config(page_title="GrowSari Demand Forecasting", layout="wide")
st.title("📦 GrowSari — Demand Forecasting")
st.caption("Upload your data (local file or S3 URL), choose your run settings, and get a champion "
           "forecast per series — exported in the client's custom 5-day / 6-week-per-month format.")

# --------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------- #
for k in ["demand", "skum", "ref_date", "hol", "reg", "spc", "trading_dow", "classif", "results"]:
    st.session_state.setdefault(k, None)


def err_box(e: Exception):
    st.error(f"⚠️ {e}")


# --------------------------------------------------------------------- #
# 1. DATA SOURCE  (local upload OR s3:// URL)
# --------------------------------------------------------------------- #
st.header("1 · Load data")

with st.expander("AWS credentials (only needed for s3:// URLs)", expanded=False):
    st.caption("Credentials are kept only in this browser session's memory — never written to "
               "disk or logged. Prefer leaving these blank and using environment variables / "
               "an IAM role on the host instead.")
    c1, c2, c3 = st.columns(3)
    aws_key = c1.text_input("AWS_ACCESS_KEY_ID", type="password", value="")
    aws_secret = c2.text_input("AWS_SECRET_ACCESS_KEY", type="password", value="")
    aws_region = c3.text_input("AWS region", value="ap-south-1")

src_mode = st.radio("Demand history source", ["Upload file", "S3 URL"], horizontal=True)
if src_mode == "Upload file":
    demand_src = st.file_uploader("Demand history (.csv or .xlsx)", type=["csv", "xlsx"], key="demand_up")
else:
    demand_src = st.text_input(
        "Demand history S3 URL",
        placeholder="s3://ds-stocksense-dev/DEV/client_experiment/project_experiment/raw_input_files/demand_history_with_channel.csv",
    ) or None

sku_mode = st.radio("SKU master source", ["Upload file", "S3 URL"], horizontal=True, key="sku_mode")
if sku_mode == "Upload file":
    sku_src = st.file_uploader("SKU master (.csv or .xlsx)", type=["csv", "xlsx"], key="sku_up")
else:
    sku_src = st.text_input("SKU master S3 URL", key="sku_url") or None

if st.button("Load & validate data", type="primary"):
    try:
        with st.spinner("Reading demand history..."):
            raw_demand = eng.load_any_table(demand_src, "demand history",
                                             aws_key, aws_secret, aws_region)
        with st.spinner("Reading SKU master..."):
            raw_skum = eng.load_any_table(sku_src, "SKU master",
                                           aws_key, aws_secret, aws_region)

        demand = eng.validate_and_normalize_demand(raw_demand)
        skum = eng.validate_and_normalize_sku_master(raw_skum)
        demand = eng.merge_demand_with_sku(demand, skum)

        ref_date = demand["date"].max()
        hol = eng.build_ph_holidays(demand["date"].min().year, ref_date.year + 1)
        reg = {d for d, v in hol.items() if v[1] == "regular"}
        spc = {d for d, v in hol.items() if v[1] == "special"}
        closed_dows = eng.detect_closed_dows(demand)
        trading_dow = tuple(d for d in range(7) if d not in closed_dows)
        eng.DEFAULT_PARAMS["closed_dows"] = closed_dows
        classif = eng.classify_all(demand, ref_date)

        st.session_state.update(demand=demand, skum=skum, ref_date=ref_date, hol=hol,
                                 reg=reg, spc=spc, trading_dow=trading_dow, classif=classif)

        st.success(f"Loaded {len(demand):,} demand rows | "
                    f"range {demand.date.min().date()} → {ref_date.date()} | "
                    f"{len(skum):,} SKUs in master.")
        dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        st.caption(f"Trading days detected: {[dow_names[d] for d in trading_dow]} "
                   f"(closed: {[dow_names[d] for d in closed_dows] or 'none — open 7 days'})")
        for lv, df in classif.items():
            st.caption(f"  {lv}: {len(df)} series classified")
    except ValidationError as e:
        err_box(e)
    except Exception as e:
        err_box(f"Unexpected error while loading data: {e}")

if st.session_state["demand"] is None:
    st.info("Load your demand history and SKU master above to continue.")
    st.stop()

demand = st.session_state["demand"]
classif = st.session_state["classif"]
ref_date = st.session_state["ref_date"]

# --------------------------------------------------------------------- #
# 2. RUN SETTINGS
# --------------------------------------------------------------------- #
st.header("2 · Run settings")

GRAINS = ["daily", "weekly", "monthly"]
hist_months = sorted(demand["date"].dt.to_period("M").astype(str).unique())
future_months = [(ref_date.to_period("M") + i).strftime("%Y-%m") for i in range(0, 13)]
all_months = sorted(set(hist_months) | set(future_months))
last_full = (ref_date.replace(day=1) - pd.Timedelta(days=1)).strftime("%Y-%m")
next_month = (ref_date.to_period("M") + 1).strftime("%Y-%m")

c1, c2 = st.columns(2)
level = c1.selectbox("Aggregation level", list(eng.LEVEL_KEYS), index=0)
temporal = c2.selectbox("Temporal aggregation (modeling grain)", GRAINS, index=0)
fgrain_opts = [g for g in GRAINS if eng.GRAIN_ORDER[g] >= eng.GRAIN_ORDER[temporal]]
fgrain = st.selectbox("Forecast granularity (output grain — 'weekly' = custom 5-day week, "
                       "6 buckets/month)", fgrain_opts, index=len(fgrain_opts) - 1)

c3, c4, c5 = st.columns(3)
testmonths = c3.multiselect("Test month(s)", hist_months, default=[last_full])
fc_start = c4.selectbox("Forecast start month", all_months, index=all_months.index(next_month))
fc_end = c5.selectbox("Forecast end month", all_months, index=all_months.index(next_month))

c6, c7 = st.columns(2)
metric = c6.selectbox("Champion-selection metric",
                       ["weighted_accuracy", "minmax_agg", "mape", "smape", "mae", "rmse"])
transform = c7.selectbox("Pre-transform", ["log1p", "none"])

try:
    eng.validate_grains(temporal, fgrain)
except ValidationError as e:
    err_box(e); st.stop()

# --------------------------------------------------------------------- #
# 3. COMBINATIONS  (template download + upload)
# --------------------------------------------------------------------- #
st.header("3 · Combinations to forecast")
level_keys = eng.LEVEL_KEYS[level]
try:
    eng.validate_level_keys(level, eng.LEVEL_KEYS, demand)
except ValidationError as e:
    err_box(e); st.stop()

tmpl_cols = level_keys + [c for c in ["segment", "ABC", "lifecycle", "total_units"]
                          if c in classif[level].columns]
tmpl_df = classif[level].sort_values("total_units", ascending=False)[tmpl_cols].reset_index(drop=True)

buf = io.BytesIO()
tmpl_df.to_excel(buf, index=False)
st.download_button(f"⬇️ Download combinations template for '{level}' ({len(tmpl_df)} series)",
                    data=buf.getvalue(), file_name=f"combinations_template_{level}.xlsx")

combos_upload = st.file_uploader("Upload your edited combinations (.xlsx) — optional", type=["xlsx"])
topn_fallback = st.number_input("If nothing uploaded, run top-N series by volume", 1, 500, 10)

combos_df = None
if combos_upload is not None:
    try:
        raw_combo = pd.read_excel(combos_upload)
        combos_df = eng.validate_combinations_upload(raw_combo, level, level_keys, demand, classif[level])
        not_found = combos_df[~combos_df["_found"]] if "_found" in combos_df else combos_df.iloc[0:0]
        if len(not_found):
            st.warning(f"{len(not_found)} uploaded combination(s) not found in demand data — skipped:\n"
                       f"{not_found[level_keys].to_string(index=False)}")
        combos_df = combos_df[combos_df["_found"]].reset_index(drop=True)
        st.success(f"Loaded {len(combos_df)} valid combinations.")
    except ValidationError as e:
        err_box(e); st.stop()
else:
    combos_df = tmpl_df.head(int(topn_fallback))
    st.caption(f"No upload — defaulting to top {len(combos_df)} combinations by volume.")

st.dataframe(combos_df.head(10), use_container_width=True)

# --------------------------------------------------------------------- #
# 4. RUN
# --------------------------------------------------------------------- #
st.header("4 · Run forecast")

if st.button("🚀 Run forecast engine", type="primary"):
    try:
        if not testmonths:
            raise ValidationError("Select at least one test month.")
        eng.validate_date_window("test", testmonths, hist_months)

        test_start = pd.Timestamp(sorted(testmonths)[0] + "-01")
        test_end = pd.Timestamp(sorted(testmonths)[-1] + "-01") + pd.offsets.MonthEnd(0)
        fc_start_ts = pd.Timestamp(fc_start + "-01")
        fc_end_ts = pd.Timestamp(fc_end + "-01") + pd.offsets.MonthEnd(0)
        if fc_end_ts < fc_start_ts:
            raise ValidationError("Forecast end month must be on/after the forecast start month.")

        cfg = dict(TEMPORAL_AGG=temporal, FORECAST_AGG=fgrain,
                   TEST_START=test_start, TEST_END=test_end,
                   FC_START=fc_start_ts, FC_END=fc_end_ts,
                   PRE_TRANSFORM=transform, SELECTION_METRIC=metric,
                   RUN_MODELS=["Naive", "SeasonalNaive", "MovingAvg", "SeasonalRollAvg", "ETS",
                               "Croston", "SBA", "TSB", "XGBoost", "LightGBM", "Ensemble"])

        hol, reg, spc = st.session_state["hol"], st.session_state["reg"], st.session_state["spc"]
        trading_dow = st.session_state["trading_dow"]

        progress = st.progress(0.0, text="Starting...")
        BT, LB, FCAST = [], [], []
        n = len(combos_df)
        t0 = time.time()
        for i, (_, r) in enumerate(combos_df.iterrows(), 1):
            keyvals = [r[k] for k in level_keys]
            label = level + "|" + "|".join(str(v) for v in keyvals)
            meta = {"level": level, **{k: r[k] for k in level_keys},
                    "segment": r.get("segment"), "ABC": r.get("ABC")}
            mask = np.ones(len(demand), bool)
            for k, v in zip(level_keys, keyvals):
                mask &= (demand[k] == v)
            sub = demand[mask]
            daily = eng.build_daily_series(sub, trading_dow)
            res = eng.run_series(daily, label, meta, cfg, hol, reg, spc, trading_dow)
            if res:
                BT.append(res["backtest"]); LB.append(res["leaderboard"]); FCAST.append(res["forecast"])
            progress.progress(i / n, text=f"[{i}/{n}] {label}")
        progress.progress(1.0, text=f"Done in {time.time()-t0:.0f}s")

        BACKTEST = pd.concat(BT, ignore_index=True) if BT else pd.DataFrame()
        LEADER = pd.concat(LB, ignore_index=True) if LB else pd.DataFrame()
        FORECASTS = pd.concat([f for f in FCAST if len(f)], ignore_index=True) if any(len(f) for f in FCAST) else pd.DataFrame()

        CHAMPIONS = LEADER[LEADER.is_champion].copy().sort_values("test_volume", ascending=False) if not LEADER.empty else pd.DataFrame()
        FORECAST_CHAMPION = FORECASTS[FORECASTS.is_champion].copy() if not FORECASTS.empty else pd.DataFrame()

        # client-requested output shape: depot, sku, model, data_type, date, month, year
        export_rows = []
        if not FORECAST_CHAMPION.empty:
            fc_out = eng.attach_output_calendar_columns(FORECAST_CHAMPION, "period")
            fc_out["data_type"] = "forecast"
            export_rows.append(fc_out)
        if not BACKTEST.empty:
            champ_map = CHAMPIONS.set_index("series")["model"].to_dict() if not CHAMPIONS.empty else {}
            bt_champ = BACKTEST[BACKTEST.apply(lambda r: champ_map.get(r["series"]) == r["model"], axis=1)]
            bt_out = eng.attach_output_calendar_columns(bt_champ, "period")
            bt_out["data_type"] = "test"
            export_rows.append(bt_out)
        OUTPUT_CUSTOM_WEEK = pd.concat(export_rows, ignore_index=True) if export_rows else pd.DataFrame()

        st.session_state["results"] = dict(BACKTEST=BACKTEST, LEADER=LEADER, FORECASTS=FORECASTS,
                                            CHAMPIONS=CHAMPIONS, FORECAST_CHAMPION=FORECAST_CHAMPION,
                                            OUTPUT_CUSTOM_WEEK=OUTPUT_CUSTOM_WEEK,
                                            run_cfg=dict(level=level, temporal=temporal, fgrain=fgrain,
                                                         test_window=f"{test_start.date()}..{test_end.date()}",
                                                         forecast_window=f"{fc_start_ts.date()}..{fc_end_ts.date()}",
                                                         metric=metric, transform=transform,
                                                         n_combinations=len(combos_df)))
        st.success(f"Finished — {len(CHAMPIONS)} champion series, {len(FORECAST_CHAMPION)} forecast rows.")
    except ValidationError as e:
        err_box(e)
    except Exception as e:
        err_box(f"Run failed: {e}")

# --------------------------------------------------------------------- #
# 5. RESULTS
# --------------------------------------------------------------------- #
if st.session_state["results"]:
    res = st.session_state["results"]
    st.header("5 · Results")

    if not res["CHAMPIONS"].empty:
        st.subheader("Champion model mix")
        st.bar_chart(res["CHAMPIONS"]["model"].value_counts())
        vw_acc = np.average(res["CHAMPIONS"]["weighted_accuracy"].fillna(0),
                             weights=res["CHAMPIONS"]["test_volume"])
        st.metric("Volume-weighted champion accuracy", f"{vw_acc:.1%}")
        st.dataframe(res["CHAMPIONS"][["series", "segment", "ABC", "model", "weighted_accuracy",
                                        "minmax_agg", "mape", "mae"]].round(3), use_container_width=True)

    if not res["OUTPUT_CUSTOM_WEEK"].empty:
        st.subheader("Output — custom week format (depot · sku · model · test/forecast · date)")
        show_cols = [c for c in ["depot", "warehouse_id", "sku_id", "category", "channel", "model",
                                  "data_type", "date", "month", "year", "actual", "pred", "forecast"]
                     if c in res["OUTPUT_CUSTOM_WEEK"].columns]
        st.dataframe(res["OUTPUT_CUSTOM_WEEK"][show_cols].head(200), use_container_width=True)

    # Excel export
    xbuf = io.BytesIO()
    with pd.ExcelWriter(xbuf, engine="openpyxl") as xl:
        pd.DataFrame({"parameter": list(res["run_cfg"]), "value": list(res["run_cfg"].values())}) \
            .to_excel(xl, sheet_name="Run_Config", index=False)
        if not res["CHAMPIONS"].empty:
            res["CHAMPIONS"].round(4).to_excel(xl, sheet_name="Champions", index=False)
        if not res["LEADER"].empty:
            res["LEADER"].round(4).to_excel(xl, sheet_name="Leaderboard", index=False)
        if not res["FORECAST_CHAMPION"].empty:
            res["FORECAST_CHAMPION"].round(2).to_excel(xl, sheet_name="Forecast_Champion", index=False)
        if not res["FORECASTS"].empty:
            res["FORECASTS"].round(2).to_excel(xl, sheet_name="Forecast_AllModels", index=False)
        if not res["BACKTEST"].empty:
            res["BACKTEST"].round(2).to_excel(xl, sheet_name="Backtest_Detail", index=False)
        if not res["OUTPUT_CUSTOM_WEEK"].empty:
            res["OUTPUT_CUSTOM_WEEK"].round(2).to_excel(xl, sheet_name="Output_CustomWeek", index=False)

    st.download_button("⬇️ Download full results (.xlsx)", data=xbuf.getvalue(),
                        file_name="GrowSari_Forecast_v4_Output.xlsx", type="primary")
