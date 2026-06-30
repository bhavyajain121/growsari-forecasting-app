# GrowSari Demand Forecasting — v4

## ⚠️ Security note (read first)
An AWS access key/secret was pasted into the chat that produced this codebase. Treat that pair as
**already compromised** — rotate or deactivate it in IAM and issue a new one. None of the files
below contain that key; they all read AWS credentials from environment variables, a Streamlit
sidebar field (session memory only), or the default boto3 credential chain. Put your **new**
credentials there, not in code.

## Files
- `growsari_engine.py` — shared core engine (holiday calendar, dynamic validation, S3 loader,
  the custom 5-day/6-week-per-month calendar, model library, backtest/forecast orchestration).
  Both the notebook and the Streamlit app import this, so behavior can't drift between them.
- `GrowSari_Forecasting_v4.ipynb` — updated Colab notebook. Upload `growsari_engine.py` alongside
  it in Colab's Files pane before running.
- `streamlit_app.py` — a working Streamlit app: upload files or paste S3 URLs, set run options,
  download the combinations template, run the engine, browse results, download the Excel output.
- `requirements.txt`

## Running the Streamlit app
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
This runs locally (or on any server/host you have — Streamlit Community Cloud, an EC2 box, etc.).
I can't deploy a live, internet-reachable URL for you from here, but this is the working,
functional app — point it at any deployment target and it runs as-is.

## What changed vs the original notebook
1. **Dynamic validation** — `growsari_engine.py` has `validate_and_normalize_demand`,
   `validate_and_normalize_sku_master`, `validate_combinations_upload`, `validate_level_keys`, and
   `validate_date_window`. Header aliases (e.g. `Warehouse`/`Depot`/`WH`) are auto-mapped; missing
   columns, bad dates, non-numeric SKU IDs, and unknown level/column references raise a clear
   `ValidationError` instead of failing deep inside the model loop.
2. **S3 support** — `load_any_table()` accepts a local path, an uploaded file, or an `s3://...`
   URL, and streams CSVs in chunks so large files don't need to fit in memory. Credentials are
   resolved from explicit args → env vars → boto3's default chain (IAM role, etc.) — never
   hardcoded.
3. **Custom week calendar** — `custom_week_end()` / `custom_week_start()` / `custom_week_range()`
   implement the requested 5-day, 6-buckets-per-month calendar (days 1–5, 6–10, 11–15, 16–20,
   21–25, 26–end-of-month), labelled by each bucket's last day. This replaces the old ISO
   `W-MON` weekly grain everywhere it mattered: resampling, backtest bucketing, the forecast
   horizon, and feature engineering (week-of-month is now 1–6, not 1–5). The exported
   `Output_CustomWeek` sheet matches the client's requested shape (depot · sku · model · data_type
   · date · month · year), verified end-to-end against the exact dates in the spec
   (2026-03-05 … 2026-03-31, 2026-04-05 … 2026-04-30, 2026-05-05 … 2026-05-31).
