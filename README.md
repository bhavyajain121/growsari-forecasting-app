# GrowSari Demand Forecasting Streamlit App

This package converts the Colab/widget notebook into a deployable Streamlit application and adds direct `s3://` CSV input support for large demand history files.

## What changed

- Demand history can be loaded directly from:
  - `s3://bucket/key.csv`
  - local CSV path
  - browser-uploaded CSV
- SKU master can be loaded from the bundled `data/sku_master.csv`, upload, local path, or S3 path.
- The notebook engine is wrapped into `forecasting_engine.py` for reusable Streamlit execution.
- Streamlit UI supports:
  - aggregation level selection
  - combinations template download
  - selected combinations upload
  - test month and forecast month selection
  - model selection
  - champion selection by metric
  - Excel output download
- The original Colab notebook is also updated as `GrowSari_Forecasting_v3_S3_READY.ipynb` for direct S3 loading.

## Files

```text
app.py                                  Streamlit application
forecasting_engine.py                   Forecasting engine extracted from notebook
requirements.txt                        Python dependencies
.streamlit/config.toml                  Streamlit config
.streamlit/secrets.toml.example         AWS secrets template; do not commit real secrets
data/sku_master.csv                     SKU master supplied with this request
GrowSari_Forecasting_v3_S3_READY.ipynb   Updated notebook with S3 loader
```

## Local run

```bash
cd growsari_streamlit_app
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## S3 credentials

For private S3 buckets, use one of these options:

### Option 1: Environment variables

```bash
export AWS_DEFAULT_REGION=ap-south-1
export AWS_ACCESS_KEY_ID=YOUR_KEY
export AWS_SECRET_ACCESS_KEY=YOUR_SECRET
# export AWS_SESSION_TOKEN=YOUR_SESSION_TOKEN   # only if using temporary credentials
streamlit run app.py
```

### Option 2: Streamlit secrets

Create `.streamlit/secrets.toml` locally, or add the same content in Streamlit Cloud app secrets:

```toml
[aws]
region_name = "ap-south-1"
access_key_id = "YOUR_AWS_ACCESS_KEY_ID"
secret_access_key = "YOUR_AWS_SECRET_ACCESS_KEY"
# session_token = "OPTIONAL_TEMP_SESSION_TOKEN"
```

## Default demand source

The app is prefilled with:

```text
s3://ds-stocksense-dev/DEV/client_experiment/project_experiment/raw_input_files/demand_history_merged.csv
```

## Deploy to Streamlit Cloud

1. Push this folder to a GitHub repository.
2. In Streamlit Cloud, create a new app from that repository.
3. Set the main file path to `app.py`.
4. Add AWS credentials under App settings > Secrets if the S3 bucket is private.
5. Deploy.

## Notes

- The app does not include AWS credentials.
- Very large S3 CSVs are read by the Streamlit server, not uploaded through the browser.
- For faster runs, upload a combinations file with only the required warehouse/SKU/category/channel combinations, or use a smaller fallback Top N.
- For production deployment, use IAM role-based access where possible instead of long-lived access keys.
