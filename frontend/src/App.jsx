import { useState, useCallback, useRef, useEffect } from "react";
import {
  Upload, Sparkles, Download, BarChart3, AlertTriangle,
  CheckCircle2, Loader2, ChevronDown, ChevronUp, X,
  Database, Zap, RefreshCw, FileText, Activity
} from "lucide-react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell
} from "recharts";
import { uploadFile, cleanData, downloadCleaned, deleteSession } from "./api";

// ── Constants ────────────────────────────────────────────────────────────────

const NUMERIC_METHODS = [
  { value: "", label: "— No operation —" },
  { value: "iterative", label: "Iterative Imputer (MICE)", category: "Imputation" },
  { value: "knn", label: "KNN Imputer", category: "Imputation" },
  { value: "median", label: "Median Imputation", category: "Imputation" },
  { value: "mean", label: "Mean Imputation", category: "Imputation" },
  { value: "isolation_forest", label: "Isolation Forest (Outliers)", category: "Outliers" },
  { value: "zscore", label: "Z-Score Filter (Outliers)", category: "Outliers" },
  { value: "standard", label: "Standard Scaler", category: "Scaling" },
  { value: "robust", label: "Robust Scaler", category: "Scaling" },
  { value: "minmax", label: "Min-Max Scaler", category: "Scaling" },
];

const CATEGORICAL_METHODS = [
  { value: "", label: "— No operation —" },
  { value: "mode", label: "Mode Imputation", category: "Imputation" },
  { value: "onehot", label: "One-Hot Encoding", category: "Encoding" },
  { value: "label", label: "Label Encoding", category: "Encoding" },
];

const METHOD_PARAMS = {
  iterative: { max_iter: { type: "number", label: "Max Iterations", default: 10, min: 1, max: 50 } },
  knn: { n_neighbors: { type: "number", label: "Neighbors (k)", default: 5, min: 1, max: 20 } },
  isolation_forest: {
    contamination: { type: "number", label: "Contamination", default: 0.05, min: 0.01, max: 0.5, step: 0.01 },
    action: { type: "select", label: "Action", default: "remove", options: ["remove", "nan"] },
  },
  zscore: {
    threshold: { type: "number", label: "Z-Score Threshold", default: 3.0, min: 1.0, max: 5.0, step: 0.1 },
    action: { type: "select", label: "Action", default: "remove", options: ["remove", "nan"] },
  },
  onehot: { drop_first: { type: "boolean", label: "Drop First Column", default: false } },
};

const NULL_COLORS = ["#10b981", "#f59e0b", "#ef4444"];
function nullColor(pct) {
  if (pct < 5) return NULL_COLORS[0];
  if (pct < 30) return NULL_COLORS[1];
  return NULL_COLORS[2];
}

// ── Sub-components ────────────────────────────────────────────────────────────

function StatCard({ label, value, sub, icon: Icon, accent }) {
  return (
    <div className="stat-card" style={{ "--accent": accent }}>
      <div className="stat-icon">
        <Icon size={18} />
      </div>
      <div>
        <div className="stat-value">{value}</div>
        <div className="stat-label">{label}</div>
        {sub && <div className="stat-sub">{sub}</div>}
      </div>
    </div>
  );
}

function ParamEditor({ method, params, onChange }) {
  const schema = METHOD_PARAMS[method];
  if (!schema) return null;
  return (
    <div className="param-grid">
      {Object.entries(schema).map(([key, def]) => (
        <div key={key} className="param-item">
          <label className="param-label">{def.label}</label>
          {def.type === "boolean" ? (
            <input
              type="checkbox"
              checked={params[key] ?? def.default}
              onChange={(e) => onChange({ ...params, [key]: e.target.checked })}
              className="param-check"
            />
          ) : def.type === "select" ? (
            <select
              value={params[key] ?? def.default}
              onChange={(e) => onChange({ ...params, [key]: e.target.value })}
              className="param-select"
            >
              {def.options.map((o) => <option key={o} value={o}>{o}</option>)}
            </select>
          ) : (
            <input
              type="number"
              value={params[key] ?? def.default}
              min={def.min}
              max={def.max}
              step={def.step ?? 1}
              onChange={(e) => onChange({ ...params, [key]: parseFloat(e.target.value) })}
              className="param-input"
            />
          )}
        </div>
      ))}
    </div>
  );
}

function ColumnCard({ col, colStats, config, onConfigChange }) {
  const [expanded, setExpanded] = useState(false);
  const isNumeric = colStats.col_type === "numeric";
  const methods = isNumeric ? NUMERIC_METHODS : CATEGORICAL_METHODS;
  const nullPct = colStats.null_pct;
  const color = nullColor(nullPct);

  const [randomMethod] = useState(() => {
    const choices = methods.filter((m) => m.value);
    return choices[Math.floor(Math.random() * choices.length)]?.value || "";
  });

  const method = config?.method ?? randomMethod;
  const params = config?.params || {};

  useEffect(() => {
    if (!config && method) {
      onConfigChange({ method, params: {} });
    }
  }, [config, method, onConfigChange]);

  const histData = colStats.histogram
    ? colStats.histogram.counts.map((count, i) => ({
        name: colStats.histogram.bins[i]?.toFixed(2),
        count,
      }))
    : null;

  return (
    <div className="col-card" style={{ "--null-color": color }}>
      <div className="col-header" onClick={() => setExpanded(!expanded)}>
        <div className="col-header-left">
          <span className="col-type-badge" data-type={colStats.col_type}>
            {colStats.col_type}
          </span>
          <span className="col-name">{col}</span>
        </div>
        <div className="col-header-right">
          <span className="null-badge" style={{ color }}>
            {nullPct}% null
          </span>
          <span className="null-bar-bg">
            <span className="null-bar-fill" style={{ width: `${Math.min(nullPct, 100)}%`, background: color }} />
          </span>
          {expanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
        </div>
      </div>

      {expanded && (
        <div className="col-body">
          {/* Stats row */}
          <div className="col-stats-row">
            <span>Nulls: <strong>{colStats.null_count}</strong></span>
            <span>Unique: <strong>{colStats.unique_count}</strong></span>
            {isNumeric && (
              <>
                <span>Mean: <strong>{colStats.mean ?? "—"}</strong></span>
                <span>Std: <strong>{colStats.std ?? "—"}</strong></span>
                <span>Min: <strong>{colStats.min}</strong></span>
                <span>Max: <strong>{colStats.max}</strong></span>
              </>
            )}
          </div>

          {/* Histogram */}
          {histData && histData.length > 0 && (
            <div className="mini-chart">
              <ResponsiveContainer width="100%" height={80}>
                <BarChart data={histData} margin={{ top: 0, right: 0, left: -30, bottom: 0 }}>
                  <XAxis dataKey="name" tick={{ fontSize: 9, fill: "#94a3b8" }} interval="preserveStartEnd" />
                  <YAxis tick={{ fontSize: 9, fill: "#94a3b8" }} />
                  <Tooltip
                    contentStyle={{ background: "#0f172a", border: "1px solid #334155", fontSize: 11 }}
                    cursor={{ fill: "rgba(99,102,241,0.1)" }}
                  />
                  <Bar dataKey="count" fill="#6366f1" radius={[2, 2, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Top values for categoricals */}
          {!isNumeric && colStats.top_values && (
            <div className="top-vals">
              {Object.entries(colStats.top_values).slice(0, 5).map(([k, v]) => (
                <div key={k} className="top-val-row">
                  <span className="top-val-key">{k}</span>
                  <span className="top-val-count">{v}</span>
                </div>
              ))}
            </div>
          )}

          {/* Method selector */}
          <div className="method-selector">
            <label className="method-label">Cleaning Method</label>
            <select
              value={method}
              onChange={(e) => onConfigChange(e.target.value ? { method: e.target.value, params: {} } : null)}
              className="method-select"
            >
              {methods.map((m) => (
                <option key={m.value} value={m.value}>{m.label}</option>
              ))}
            </select>
            {method && (
              <ParamEditor
                method={method}
                params={params}
                onChange={(p) => onConfigChange({ method, params: p })}
              />
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Main App ─────────────────────────────────────────────────────────────────

export default function App() {
  const [phase, setPhase] = useState("upload"); // upload | loading | explore | cleaning | done
  const [uploadPct, setUploadPct] = useState(0);
  const [session, setSession] = useState(null); // {session_id, filename, stats}
  const [cleanResult, setCleanResult] = useState(null);
  const [globalConfig, setGlobalConfig] = useState(null);
  const [globalMethod, setGlobalMethod] = useState("");
  const [globalParams, setGlobalParams] = useState({});
  const [colConfigs, setColConfigs] = useState({});
  const [error, setError] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [activeTab, setActiveTab] = useState("overview");
  const fileRef = useRef();

  const handleFile = useCallback(async (file) => {
    if (!file) return;
    if (!file.name.match(/\.(csv|parquet)$/i)) {
      setError("Only CSV and Parquet files are supported.");
      return;
    }
    setError(null);
    setPhase("loading");
    setUploadPct(0);
    try {
      const data = await uploadFile(file, setUploadPct);
      setSession(data);
      setColConfigs({});
      setGlobalMethod("");
      setGlobalParams({});
      setCleanResult(null);
      setPhase("explore");
    } catch (e) {
      setError(e.message);
      setPhase("upload");
    }
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setDragging(false);
    handleFile(e.dataTransfer.files[0]);
  }, [handleFile]);

  const handleClean = async () => {
    if (!session) return;
    setPhase("cleaning");
    setError(null);
    try {
      const payload = {
        global_config: globalMethod ? { method: globalMethod, params: globalParams } : null,
        column_configs: Object.fromEntries(
          Object.entries(colConfigs).filter(([, v]) => v?.method)
        ),
      };
      const result = await cleanData(session.session_id, payload);
      setCleanResult(result);
      setPhase("done");
    } catch (e) {
      setError(e.message);
      setPhase("explore");
    }
  };

  const handleReset = () => {
    if (session) deleteSession(session.session_id).catch(() => {});
    setSession(null);
    setCleanResult(null);
    setColConfigs({});
    setGlobalMethod("");
    setPhase("upload");
    setError(null);
  };

  const stats = session?.stats;
  const cols = stats?.columns || {};
  const colNames = stats?.column_names || [];
  const numericCols = colNames.filter((c) => cols[c]?.col_type === "numeric");
  const catCols = colNames.filter((c) => cols[c]?.col_type === "categorical");

  const nullMapData = colNames.map((c) => ({
    name: c.length > 14 ? c.slice(0, 13) + "…" : c,
    null_pct: cols[c]?.null_pct || 0,
  }));

  const configuredCount = Object.values(colConfigs).filter((v) => v?.method).length;

  return (
    <div className="app-root">
      {/* Header */}
      <header className="app-header">
        <div className="header-brand">
          <div className="brand-icon"><Sparkles size={18} /></div>
          <span className="brand-name">DataForge<span className="brand-accent">ML</span></span>
        </div>
        <div className="header-center">
          {session && (
            <div className="file-pill">
              <FileText size={12} />
              <span>{session.filename}</span>
              <span className="pill-sep">·</span>
              <span>{stats?.total_rows?.toLocaleString()} rows</span>
              <span className="pill-sep">·</span>
              <span>{stats?.total_cols} cols</span>
            </div>
          )}
        </div>
        <div className="header-actions">
          {phase === "done" && (
            <>
              <button className="btn-download" onClick={() => downloadCleaned(session.session_id, "csv")}>
                <Download size={14} /> CSV
              </button>
              <button className="btn-download secondary" onClick={() => downloadCleaned(session.session_id, "parquet")}>
                <Download size={14} /> Parquet
              </button>
            </>
          )}
          {session && (
            <button className="btn-reset" onClick={handleReset}>
              <RefreshCw size={13} /> New File
            </button>
          )}
        </div>
      </header>

      <main className="app-main">
        {/* Error toast */}
        {error && (
          <div className="error-toast">
            <AlertTriangle size={15} />
            <span>{error}</span>
            <button onClick={() => setError(null)}><X size={13} /></button>
          </div>
        )}

        {/* ── UPLOAD PHASE ── */}
        {phase === "upload" && (
          <div className="upload-scene">
            <div className="upload-hero">
              <div className="hero-glow" />
              <h1 className="hero-title">ML Data Cleaning Studio</h1>
              <p className="hero-sub">Upload a dataset. Inspect. Configure ML pipelines per column. Clean & export.</p>
            </div>
            <div
              className={`dropzone ${dragging ? "dragging" : ""}`}
              onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={handleDrop}
              onClick={() => fileRef.current.click()}
            >
              <input
                ref={fileRef}
                type="file"
                accept=".csv,.parquet"
                className="hidden"
                onChange={(e) => handleFile(e.target.files[0])}
              />
              <div className="dz-icon"><Upload size={32} /></div>
              <div className="dz-text">Drop CSV or Parquet here</div>
              <div className="dz-sub">or click to browse</div>
            </div>
            <div className="method-pills">
              {["MICE Imputation", "KNN Imputer", "Isolation Forest", "Z-Score", "One-Hot", "RobustScaler"].map((m) => (
                <span key={m} className="method-pill">{m}</span>
              ))}
            </div>
          </div>
        )}

        {/* ── LOADING ── */}
        {phase === "loading" && (
          <div className="loading-scene">
            <Loader2 className="spin" size={40} />
            <div className="loading-label">Parsing & profiling dataset…</div>
            <div className="progress-bar">
              <div className="progress-fill" style={{ width: `${uploadPct}%` }} />
            </div>
            <div className="progress-pct">{uploadPct}%</div>
          </div>
        )}

        {/* ── EXPLORE & CONFIGURE ── */}
        {(phase === "explore" || phase === "done") && stats && (
          <div className="explore-layout">
            {/* Stat cards */}
            <div className="stat-row">
              <StatCard label="Total Rows" value={stats.total_rows.toLocaleString()} icon={Database} accent="#6366f1" />
              <StatCard label="Columns" value={stats.total_cols} sub={`${numericCols.length} numeric · ${catCols.length} categorical`} icon={BarChart3} accent="#8b5cf6" />
              <StatCard label="Missing Cells" value={stats.total_nulls.toLocaleString()} sub={`${stats.null_pct_overall}% of data`} icon={AlertTriangle} accent="#f59e0b" />
              {cleanResult && (
                <StatCard label="Rows After Clean" value={cleanResult.rows_after.toLocaleString()} sub={`${cleanResult.rows_before - cleanResult.rows_after} removed`} icon={CheckCircle2} accent="#10b981" />
              )}
            </div>

            {/* Tabs */}
            <div className="tabs">
              {["overview", "columns", "configure"].map((t) => (
                <button key={t} className={`tab ${activeTab === t ? "active" : ""}`} onClick={() => setActiveTab(t)}>
                  {t === "overview" && <Activity size={13} />}
                  {t === "columns" && <Database size={13} />}
                  {t === "configure" && <Zap size={13} />}
                  {t.charAt(0).toUpperCase() + t.slice(1)}
                </button>
              ))}
            </div>

            {/* Overview */}
            {activeTab === "overview" && (
              <div className="panel">
                <h3 className="panel-title">Null Map — % Missing per Column</h3>
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={nullMapData} margin={{ top: 10, right: 10, left: -10, bottom: 60 }}>
                    <XAxis dataKey="name" tick={{ fontSize: 10, fill: "#94a3b8" }} angle={-40} textAnchor="end" interval={0} />
                    <YAxis tick={{ fontSize: 10, fill: "#94a3b8" }} unit="%" domain={[0, 100]} />
                    <Tooltip
                      formatter={(v) => [`${v}%`, "Null %"]}
                      contentStyle={{ background: "#0f172a", border: "1px solid #334155", fontSize: 11 }}
                    />
                    <Bar dataKey="null_pct" radius={[3, 3, 0, 0]}>
                      {nullMapData.map((entry, i) => (
                        <Cell key={i} fill={nullColor(entry.null_pct)} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
                <div className="legend-row">
                  <span className="legend-item" style={{ color: "#10b981" }}>■ &lt;5% (clean)</span>
                  <span className="legend-item" style={{ color: "#f59e0b" }}>■ 5–30% (moderate)</span>
                  <span className="legend-item" style={{ color: "#ef4444" }}>■ &gt;30% (high)</span>
                </div>
              </div>
            )}

            {/* Columns */}
            {activeTab === "columns" && (
              <div className="col-list">
                {colNames.map((col) => (
                  <ColumnCard
                    key={col}
                    col={col}
                    colStats={cols[col]}
                    config={colConfigs[col]}
                    onConfigChange={(cfg) => setColConfigs((prev) => ({ ...prev, [col]: cfg }))}
                  />
                ))}
              </div>
            )}

            {/* Configure */}
            {activeTab === "configure" && (
              <div className="panel">
                <h3 className="panel-title">Global Pipeline</h3>
                <p className="panel-desc">Applied to all columns not individually configured. Column-level settings take priority.</p>

                <div className="global-row">
                  <div className="method-selector" style={{ flex: 1 }}>
                    <label className="method-label">Global Method (Numeric)</label>
                    <select
                      value={globalMethod}
                      onChange={(e) => { setGlobalMethod(e.target.value); setGlobalParams({}); }}
                      className="method-select"
                    >
                      {NUMERIC_METHODS.map((m) => (
                        <option key={m.value} value={m.value}>{m.label}</option>
                      ))}
                    </select>
                    {globalMethod && (
                      <ParamEditor method={globalMethod} params={globalParams} onChange={setGlobalParams} />
                    )}
                  </div>
                </div>

                <h3 className="panel-title" style={{ marginTop: "1.5rem" }}>Column Summary</h3>
                <div className="config-summary">
                  {colNames.map((col) => {
                    const cfg = colConfigs[col];
                    return (
                      <div key={col} className={`summary-row ${cfg?.method ? "configured" : ""}`}>
                        <span className="summary-col">{col}</span>
                        <span className="summary-method">{cfg?.method || <em>none</em>}</span>
                        {cfg?.method && (
                          <button className="clear-btn" onClick={() => setColConfigs((p) => { const n = { ...p }; delete n[col]; return n; })}>
                            <X size={11} />
                          </button>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Action bar */}
            {phase === "explore" && (
              <div className="action-bar">
                <div className="action-summary">
                  {configuredCount > 0 && <span className="badge">{configuredCount} column{configuredCount > 1 ? "s" : ""} configured</span>}
                  {globalMethod && <span className="badge global">Global: {globalMethod}</span>}
                  {configuredCount === 0 && !globalMethod && <span className="muted">No operations configured yet</span>}
                </div>
                <button
                  className="btn-clean"
                  onClick={handleClean}
                  disabled={configuredCount === 0 && !globalMethod}
                >
                  <Zap size={16} /> Apply ML Cleaning
                </button>
              </div>
            )}

            {/* Done banner */}
            {phase === "done" && cleanResult && (
              <div className="done-banner">
                <CheckCircle2 size={20} className="done-icon" />
                <div className="done-text">
                  <strong>Cleaning complete.</strong>{" "}
                  {cleanResult.rows_before - cleanResult.rows_after > 0 &&
                    `${(cleanResult.rows_before - cleanResult.rows_after).toLocaleString()} rows removed. `}
                  {cleanResult.nulls_before - cleanResult.nulls_after > 0 &&
                    `${(cleanResult.nulls_before - cleanResult.nulls_after).toLocaleString()} nulls resolved.`}
                </div>
                <div className="done-actions">
                  <button className="btn-download" onClick={() => downloadCleaned(session.session_id, "csv")}>
                    <Download size={14} /> CSV
                  </button>
                  <button className="btn-download secondary" onClick={() => downloadCleaned(session.session_id, "parquet")}>
                    <Download size={14} /> Parquet
                  </button>
                  <button className="btn-reset" onClick={handleReset} style={{ marginLeft: 8 }}>
                    New File
                  </button>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ── CLEANING SPINNER ── */}
        {phase === "cleaning" && (
          <div className="loading-scene">
            <Loader2 className="spin" size={40} />
            <div className="loading-label">Running ML pipelines…</div>
            <div className="loading-sub">Imputing · Detecting outliers · Encoding · Scaling</div>
          </div>
        )}
      </main>
    </div>
  );
}
