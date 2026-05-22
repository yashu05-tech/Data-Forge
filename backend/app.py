import uuid
import io
import json
import traceback
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import  StreamingResponse
from pydantic import BaseModel

from cleaners import DataCleaner

app = FastAPI(title="ML Data Cleaning Studio", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store: session_id -> {"df": DataFrame, "filename": str, "stats": dict}
sessions: Dict[str, Dict[str, Any]] = {}

CHUNK_SIZE = 50_000  # rows per chunk for large files


def compute_stats(df: pd.DataFrame) -> Dict[str, Any]:
    stats = {}
    total_rows = len(df)
    total_cells = df.size

    null_counts = df.isnull().sum().to_dict()
    null_pcts = {col: round(cnt / total_rows * 100, 2) for col, cnt in null_counts.items()}

    col_types = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            col_types[col] = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            col_types[col] = "datetime"
        else:
            col_types[col] = "categorical"

    col_stats = {}
    for col in df.columns:
        dtype = col_types[col]
        s = {
            "dtype": str(df[col].dtype),
            "col_type": dtype,
            "null_count": int(null_counts[col]),
            "null_pct": null_pcts[col],
            "unique_count": int(df[col].nunique()),
        }
        if dtype == "numeric":
            desc = df[col].describe()
            s.update({
                "mean": round(float(desc["mean"]), 4) if not np.isnan(desc["mean"]) else None,
                "std": round(float(desc["std"]), 4) if not np.isnan(desc.get("std", float("nan"))) else None,
                "min": round(float(desc["min"]), 4),
                "max": round(float(desc["max"]), 4),
                "median": round(float(df[col].median()), 4),
                "q25": round(float(desc["25%"]), 4),
                "q75": round(float(desc["75%"]), 4),
            })
            # Distribution histogram (20 bins)
            non_null = df[col].dropna()
            if len(non_null) > 0:
                counts, bin_edges = np.histogram(non_null, bins=min(20, len(non_null)))
                s["histogram"] = {
                    "counts": counts.tolist(),
                    "bins": [round(float(x), 4) for x in bin_edges.tolist()],
                }
        elif dtype == "categorical":
            top_vals = df[col].value_counts().head(10)
            s["top_values"] = {str(k): int(v) for k, v in top_vals.items()}

        col_stats[col] = s

    stats["total_rows"] = total_rows
    stats["total_cols"] = len(df.columns)
    stats["total_cells"] = int(total_cells)
    stats["total_nulls"] = int(df.isnull().sum().sum())
    stats["null_pct_overall"] = round(df.isnull().sum().sum() / total_cells * 100, 2)
    stats["columns"] = col_stats
    stats["column_names"] = list(df.columns)
    stats["col_types"] = col_types

    return stats


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    filename = file.filename or "data.csv"
    content = await file.read()

    try:
        if filename.endswith(".parquet"):
            df = pd.read_parquet(io.BytesIO(content))
        elif filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Only CSV and Parquet files are supported.")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to parse file: {str(e)}")

    if df.empty:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")

    session_id = str(uuid.uuid4())
    stats = compute_stats(df)
    sessions[session_id] = {"df": df, "filename": filename, "stats": stats}

    return {"session_id": session_id, "filename": filename, "stats": stats}


@app.get("/session/{session_id}/stats")
def get_stats(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    return sessions[session_id]["stats"]


class CleaningConfig(BaseModel):
    global_config: Optional[Dict[str, Any]] = None   # {method: str, params: dict}
    column_configs: Optional[Dict[str, Dict[str, Any]]] = None  # {col_name: {method, params}}


@app.post("/session/{session_id}/clean")
def clean_data(session_id: str, config: CleaningConfig):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")

    df_original = sessions[session_id]["df"].copy()
    cleaner = DataCleaner(df_original)

    try:
        result_df = cleaner.apply(
            global_config=config.global_config,
            column_configs=config.column_configs or {},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleaning failed: {traceback.format_exc()}")

    sessions[session_id]["cleaned_df"] = result_df
    sessions[session_id]["cleaned_stats"] = compute_stats(result_df)

    return {
        "status": "ok",
        "cleaned_stats": sessions[session_id]["cleaned_stats"],
        "rows_before": len(df_original),
        "rows_after": len(result_df),
        "nulls_before": int(df_original.isnull().sum().sum()),
        "nulls_after": int(result_df.isnull().sum().sum()),
    }


@app.get("/session/{session_id}/download")
def download_cleaned(session_id: str, fmt: str = "csv"):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    if "cleaned_df" not in sessions[session_id]:
        raise HTTPException(status_code=400, detail="No cleaned dataset available. Run /clean first.")

    result_df = sessions[session_id]["cleaned_df"]
    original_name = Path(sessions[session_id]["filename"]).stem

    if fmt == "parquet":
        buf = io.BytesIO()
        result_df.to_parquet(buf, index=False)
        buf.seek(0)
        media_type = "application/octet-stream"
        filename = f"{original_name}_cleaned.parquet"
    else:
        buf = io.StringIO()
        result_df.to_csv(buf, index=False)
        buf = io.BytesIO(buf.getvalue().encode())
        media_type = "text/csv"
        filename = f"{original_name}_cleaned.csv"

    return StreamingResponse(
        buf,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    sessions.pop(session_id, None)
    return {"status": "deleted"}


@app.get("/health")
def health():
    return {"status": "ok", "sessions": len(sessions)}
