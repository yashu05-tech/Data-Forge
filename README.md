# DataForgeML — ML Data Cleaning Studio

Full-stack ML data cleaning application with a FastAPI backend and Vite + React frontend.

---

## Architecture

```
ml-cleaning-studio/
├── backend/
│   ├── app.py          # FastAPI app, session state, upload/download endpoints
│   ├── cleaners.py     # All ML/statistical cleaning pipelines (scikit-learn, scipy)
│   └── requirements.txt
└── frontend/
    ├── src/
    │   ├── api.js      # Axios API client
    │   ├── App.jsx     # Full UI — dropzone, stats, null-map, column config, clean/download
    │   └── index.css   # Industrial dark theme (Space Mono + DM Sans)
    ├── index.html
    ├── package.json
    └── vite.config.js
```

---

## Quick Start

### Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
# Opens at http://localhost:5173
```

---

## REST API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/upload` | Upload CSV/Parquet → returns `session_id` + full column stats |
| `GET`  | `/session/{id}/stats` | Fetch session stats |
| `POST` | `/session/{id}/clean` | Apply cleaning config (global + per-column) |
| `GET`  | `/session/{id}/download?fmt=csv\|parquet` | Download cleaned dataset |
| `DELETE` | `/session/{id}` | Release session memory |

### Cleaning config payload (`POST /session/{id}/clean`)

```json
{
  "global_config": { "method": "median", "params": {} },
  "column_configs": {
    "age":    { "method": "knn",              "params": { "n_neighbors": 5 } },
    "income": { "method": "isolation_forest", "params": { "contamination": 0.05, "action": "remove" } },
    "gender": { "method": "onehot",           "params": { "drop_first": false } }
  }
}
```

---

## Supported ML Methods

| Category | Method key | Description |
|----------|-----------|-------------|
| Imputation | `iterative` | IterativeImputer (MICE / linear regression) |
| Imputation | `knn` | KNNImputer |
| Imputation | `median` | Column median (categorical → mode) |
| Imputation | `mean` | Column mean |
| Imputation | `mode` | Most-frequent value |
| Outliers | `isolation_forest` | IsolationForest (remove or → NaN) |
| Outliers | `zscore` | Z-score threshold filter |
| Encoding | `onehot` | One-Hot Encoding |
| Encoding | `label` | Label / ordinal encoding |
| Encoding | `target` | Mean target encoding |
| Scaling | `standard` | StandardScaler (z-norm) |
| Scaling | `robust` | RobustScaler (IQR-based) |
| Scaling | `minmax` | MinMaxScaler [0, 1] |
