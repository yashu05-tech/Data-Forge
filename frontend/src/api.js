import axios from "axios";

const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 120_000,
});

api.interceptors.response.use(
  (res) => res,
  (err) => {
    const msg =
      err?.response?.data?.detail ||
      err?.message ||
      "Unknown error occurred";
    return Promise.reject(new Error(typeof msg === "string" ? msg : JSON.stringify(msg)));
  }
);

// ── Upload ──────────────────────────────────────────────────────────────────

/**
 * Upload a CSV or Parquet file.
 * @returns {Promise<{session_id: string, filename: string, stats: object}>}
 */
export async function uploadFile(file, onProgress) {
  const formData = new FormData();
  formData.append("file", file);

  const res = await api.post("/upload", formData, {
    headers: { "Content-Type": "multipart/form-data" },
    onUploadProgress: (evt) => {
      if (onProgress && evt.total) {
        onProgress(Math.round((evt.loaded / evt.total) * 100));
      }
    },
  });
  return res.data;
}

// ── Stats ───────────────────────────────────────────────────────────────────

/**
 * Fetch session stats (re-fetch after cleaning).
 * @returns {Promise<object>}
 */
export async function getStats(sessionId) {
  const res = await api.get(`/session/${sessionId}/stats`);
  return res.data;
}

// ── Clean ───────────────────────────────────────────────────────────────────

/**
 * Apply cleaning configuration to the session dataset.
 *
 * @param {string} sessionId
 * @param {{ global_config?: {method: string, params: object}, column_configs?: Record<string, {method: string, params: object}> }} config
 * @returns {Promise<{status: string, cleaned_stats: object, rows_before: number, rows_after: number, nulls_before: number, nulls_after: number}>}
 */
export async function cleanData(sessionId, config) {
  const res = await api.post(`/session/${sessionId}/clean`, config);
  return res.data;
}

// ── Download ─────────────────────────────────────────────────────────────────

/**
 * Trigger cleaned dataset download.
 * @param {string} sessionId
 * @param {"csv"|"parquet"} fmt
 */
export async function downloadCleaned(sessionId, fmt = "csv") {
  const res = await api.get(`/session/${sessionId}/download`, {
    params: { fmt },
    responseType: "blob",
  });

  const contentDisp = res.headers["content-disposition"] || "";
  const match = contentDisp.match(/filename="?([^"]+)"?/);
  const filename = match ? match[1] : `cleaned_data.${fmt}`;

  const url = window.URL.createObjectURL(new Blob([res.data]));
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.URL.revokeObjectURL(url);
}

// ── Session cleanup ──────────────────────────────────────────────────────────

export async function deleteSession(sessionId) {
  await api.delete(`/session/${sessionId}`);
}

export default api;
