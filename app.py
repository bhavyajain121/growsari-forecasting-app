from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

from forecasting_engine import (
    ALL_MODELS,
    GRAIN_ORDER,
    LEVEL_KEYS,
    build_context,
    combinations_template,
    export_results_excel,
    load_demand_from_sources,
    month_options,
    prepare_demand_data,
    read_csv_source,
    run_forecasting_job,
    summarize_context,
)

DEFAULT_S3_DEMAND = "s3://ds-stocksense-dev/DEV/client_experiment/project_experiment/raw_input_files/demand_history_merged.csv"
APP_DIR = Path(__file__).resolve().parent
BUNDLED_SKU_MASTER = APP_DIR / "data" / "sku_master.csv"

st.set_page_config(page_title="GrowSari Demand Forecasting", page_icon="📈", layout="wide")
st.title("📈 GrowSari Demand Forecasting")
st.caption("Input-driven demand forecasting with S3 demand history support, SKU master mapping, champion model selection, and Excel export.")


def _secret(path: str, default: str = "") -> str:
    """Safely read nested Streamlit secrets like aws.access_key_id."""
    try:
        cur: Any = st.secrets
        for part in path.split("."):
            cur = cur[part]
        return str(cur)
    except Exception:
        return default


@st.cache_data(show_spinner=False, ttl=60 * 60)
def _load_from_sources_cached(
    demand_source: str,
    sku_source: Optional[str],
    aws_access_key_id: str,
    aws_secret_access_key: str,
    aws_session_token: str,
    region_name: str,
    unsigned_s3: bool,
) -> pd.DataFrame:
    return load_demand_from_sources(
        demand_source,
        sku_source,
        aws_access_key_id=aws_access_key_id or None,
        aws_secret_access_key=aws_secret_access_key or None,
        aws_session_token=aws_session_token or None,
        region_name=region_name or None,
        unsigned_s3=unsigned_s3,
    )


@st.cache_data(show_spinner=False, ttl=60 * 60)
def _build_context_cached(demand: pd.DataFrame):
    return build_context(demand)


with st.sidebar:
    st.header("1) Data source")
    demand_mode = st.radio("Demand history", ["S3 / file path", "Upload CSV"], horizontal=False)
    demand_source = None
    demand_upload = None
    if demand_mode == "S3 / file path":
        demand_source = st.text_input("Demand CSV source", value=DEFAULT_S3_DEMAND)
    else:
        demand_upload = st.file_uploader("Upload demand_history CSV", type=["csv"], key="demand_upload")

    st.divider()
    st.header("2) SKU master")
    sku_mode = st.radio("SKU master source", ["Use bundled sku_master.csv", "Upload CSV", "S3 / file path", "No SKU master"], index=0)
    sku_source = None
    sku_upload = None
    if sku_mode == "Use bundled sku_master.csv":
        sku_source = str(BUNDLED_SKU_MASTER)
        st.caption(f"Using bundled file: `{BUNDLED_SKU_MASTER.name}`")
    elif sku_mode == "Upload CSV":
        sku_upload = st.file_uploader("Upload SKU master CSV", type=["csv"], key="sku_upload")
    elif sku_mode == "S3 / file path":
        sku_source = st.text_input("SKU master CSV source", value=str(BUNDLED_SKU_MASTER))

    st.divider()
    st.header("3) AWS / S3 access")
    with st.expander("Credentials / advanced", expanded=False):
        st.caption("For private buckets, prefer Streamlit secrets or IAM role. Manual inputs are optional.")
        region_name = st.text_input("AWS region", value=_secret("aws.region_name", ""))
        aws_access_key_id = st.text_input("AWS access key ID", value=_secret("aws.access_key_id", ""), type="password")
        aws_secret_access_key = st.text_input("AWS secret access key", value=_secret("aws.secret_access_key", ""), type="password")
        aws_session_token = st.text_input("AWS session token", value=_secret("aws.session_token", ""), type="password")
        unsigned_s3 = st.checkbox("Use unsigned S3 access for public bucket", value=False)

    load_clicked = st.button("Load data", type="primary", use_container_width=True)

if "ctx" not in st.session_state:
    st.info("Load the demand history from S3 or CSV to start.")

if load_clicked:
    try:
        with st.spinner("Loading demand history and building classification templates..."):
            if demand_mode == "Upload CSV":
                if demand_upload is None:
                    st.error("Please upload the demand history CSV.")
                    st.stop()
                demand_raw = pd.read_csv(demand_upload)
                sku_raw = None
                if sku_mode == "Upload CSV" and sku_upload is not None:
                    sku_raw = pd.read_csv(sku_upload)
                elif sku_source:
                    sku_raw = read_csv_source(
                        sku_source,
                        aws_access_key_id=aws_access_key_id or None,
                        aws_secret_access_key=aws_secret_access_key or None,
                        aws_session_token=aws_session_token or None,
                        region_name=region_name or None,
                        unsigned=unsigned_s3,
                    )
                demand = prepare_demand_data(demand_raw, sku_raw)
            else:
                if not demand_source:
                    st.error("Please enter a demand CSV source.")
                    st.stop()
                if sku_mode == "Upload CSV" and sku_upload is not None:
                    demand_raw = read_csv_source(
                        demand_source,
                        aws_access_key_id=aws_access_key_id or None,
                        aws_secret_access_key=aws_secret_access_key or None,
                        aws_session_token=aws_session_token or None,
                        region_name=region_name or None,
                        unsigned=unsigned_s3,
                    )
                    sku_raw = pd.read_csv(sku_upload)
                    demand = prepare_demand_data(demand_raw, sku_raw)
                else:
                    demand = _load_from_sources_cached(
                        demand_source,
                        sku_source,
                        aws_access_key_id,
                        aws_secret_access_key,
                        aws_session_token,
                        region_name,
                        unsigned_s3,
                    )
            ctx = _build_context_cached(demand)
            st.session_state.ctx = ctx
            st.session_state.summary = summarize_context(ctx)
        st.success("Data loaded successfully.")
    except Exception as exc:
        st.error(f"Could not load/build data: {exc}")
        st.stop()

ctx = st.session_state.get("ctx")
if ctx is not None:
    summary: Dict[str, Any] = st.session_state.get("summary") or summarize_context(ctx)
    st.subheader("Dataset summary")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rows", f"{summary['rows']:,}")
    c2.metric("Date range", f"{summary['date_min']} → {summary['date_max']}")
    c3.metric("Warehouses", f"{summary['warehouses']:,}")
    c4.metric("SKUs", f"{summary['skus']:,}")
    c5.metric("Channels", f"{summary['channels']:,}")
    st.caption(
        "Trading weekdays: " + ", ".join(summary["trading_weekdays"]) +
        (" | Closed: " + ", ".join(summary["closed_weekdays"]) if summary["closed_weekdays"] else " | Closed: none")
    )
    with st.expander("Series counts by level", expanded=False):
        st.dataframe(pd.DataFrame([summary["series_counts"]]).T.rename(columns={0: "series_count"}), use_container_width=True)

    st.divider()
    st.subheader("Forecast run setup")
    left, right = st.columns([1, 1])

    with left:
        level = st.selectbox("Aggregation level", list(LEVEL_KEYS.keys()), index=0)
        template_df = combinations_template(level, LEVEL_KEYS[level], ctx.classif[level])
        template_bytes = io.BytesIO()
        template_df.to_excel(template_bytes, index=False)
        st.download_button(
            "Download combinations template",
            data=template_bytes.getvalue(),
            file_name=f"combinations_template_{level}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
        combos_upload = st.file_uploader("Upload selected combinations .xlsx", type=["xlsx"], key=f"combo_{level}")
        fallback_top_n = st.number_input("Fallback top N if no combinations file is uploaded", min_value=1, max_value=500, value=10, step=1)

    hist_months, all_months, last_full, next_month = month_options(ctx)
    grains = ["daily", "weekly", "monthly"]

    with right:
        temporal_agg = st.selectbox("Temporal aggregation", grains, index=0)
        valid_fgrains = [g for g in grains if GRAIN_ORDER[g] >= GRAIN_ORDER[temporal_agg]]
        default_fgrain = "weekly" if "weekly" in valid_fgrains else valid_fgrains[0]
        forecast_agg = st.selectbox("Forecast granularity", valid_fgrains, index=valid_fgrains.index(default_fgrain))
        test_default = [last_full] if last_full in hist_months else hist_months[-1:]
        test_months = st.multiselect("Test month(s)", hist_months, default=test_default)
        fc_start = st.selectbox("Forecast start month", all_months, index=all_months.index(next_month) if next_month in all_months else len(all_months)-1)
        fc_end = st.selectbox("Forecast end month", all_months, index=all_months.index(next_month) if next_month in all_months else len(all_months)-1)

    with st.expander("Model and metric options", expanded=False):
        selected_models = st.multiselect("Models to run", ALL_MODELS, default=ALL_MODELS)
        selection_metric = st.selectbox("Champion selection metric", ["weighted_accuracy", "minmax_agg", "mape", "smape", "mae", "rmse"], index=0)
        pre_transform = st.selectbox("Pre-transform", ["log1p", "none"], index=0)

    run_clicked = st.button("Run forecast", type="primary", use_container_width=True)

    if run_clicked:
        if not selected_models:
            st.error("Select at least one model.")
            st.stop()
        if not test_months:
            st.error("Select at least one test month.")
            st.stop()
        try:
            combos_df = pd.read_excel(combos_upload) if combos_upload is not None else None
            progress = st.progress(0)
            status = st.empty()

            def _progress(i: int, n: int, label: str) -> None:
                progress.progress(min(1.0, i / max(n, 1)))
                status.write(f"Running {i}/{n}: `{label}`")

            with st.spinner("Running model backtests, champion selection, and forecast..."):
                results = run_forecasting_job(
                    ctx,
                    level=level,
                    combinations_df=combos_df,
                    temporal_agg=temporal_agg,
                    forecast_agg=forecast_agg,
                    test_months=test_months,
                    forecast_start_month=fc_start,
                    forecast_end_month=fc_end,
                    selection_metric=selection_metric,
                    pre_transform=pre_transform,
                    run_models=selected_models,
                    fallback_top_n=int(fallback_top_n),
                    progress_callback=_progress,
                )
            status.success("Forecast run completed.")
            st.session_state.results = results
        except Exception as exc:
            st.error(f"Forecast run failed: {exc}")
            st.stop()

    results = st.session_state.get("results")
    if results is not None:
        st.divider()
        st.subheader("Results")
        champions = results.get("Champions", pd.DataFrame())
        forecasts = results.get("Forecast_Champion", pd.DataFrame())
        if champions.empty:
            st.warning("No champion results were generated. Try a different test window or combinations file.")
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Champion series", f"{len(champions):,}")
            weighted_acc = None
            if "weighted_accuracy" in champions and "test_volume" in champions and champions["test_volume"].sum() > 0:
                weighted_acc = (champions["weighted_accuracy"].fillna(0) * champions["test_volume"].fillna(0)).sum() / champions["test_volume"].fillna(0).sum()
            m2.metric("Vol-weighted accuracy", f"{weighted_acc:.3f}" if weighted_acc is not None else "—")
            m3.metric("Forecast rows", f"{len(forecasts):,}")

            tab1, tab2, tab3, tab4 = st.tabs(["Champions", "Champion Forecast", "Leaderboard", "Backtest Detail"])
            with tab1:
                st.dataframe(champions, use_container_width=True, height=420)
            with tab2:
                st.dataframe(forecasts, use_container_width=True, height=420)
            with tab3:
                st.dataframe(results.get("Leaderboard", pd.DataFrame()), use_container_width=True, height=420)
            with tab4:
                st.dataframe(results.get("Backtest_Detail", pd.DataFrame()), use_container_width=True, height=420)

        excel_bytes = export_results_excel(results)
        st.download_button(
            "Download forecast output Excel",
            data=excel_bytes,
            file_name="GrowSari_Forecast_Streamlit_Output.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
