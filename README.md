<div align="center">

# 🧹 ML Cleaning Studio

**A powerful, browser-based data cleaning tool built for ML workflows**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Visit%20App-6366f1?style=for-the-badge&logo=vercel)](https://yashu-s.vercel.app/)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi)
![React](https://img.shields.io/badge/React-Vite-61DAFB?style=for-the-badge&logo=react)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python)

Upload a CSV → clean it → export it. No notebooks. No code.

</div>

---

## ✨ Features

| Category | Operations |
|---|---|
| **Missing Values** | Mean, Median, Mode imputation · KNN imputation · Iterative (MICE) imputation |
| **Outlier Detection** | Z-Score · IQR · Isolation Forest |
| **Encoding** | One-Hot · Label · Target Encoding |
| **Scaling** | Standard (Z-score) · MinMax · Robust Scaler |
| **Insights** | Per-column stats · Null % · Data types · Unique counts |

- 📂 Upload any CSV file and get instant column-level statistics
- ⚙️ Apply multiple cleaning operations in sequence
- 📥 Export the cleaned dataset ready for model training
- ⚡ Session-based — your data stays in memory during the session

---

## 🚀 Live Demo

👉 **[https://yashu-s.vercel.app/](https://yashu-s.vercel.app/)**

---

## 🏗️ Tech Stack

```
Frontend  →  React + Vite + Tailwind CSS   (deployed on Vercel)
Backend   →  FastAPI + Uvicorn             (deployed on Render)
ML        →  scikit-learn · pandas · scipy · numpy
```

---

## 🛠️ Local Development

### Prerequisites
- Python 3.11+
- Node.js 18+

### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port $PORT

```

### Frontend

```bash
cd frontend
npm install
```

Create a `.env` file inside `frontend/`:

```env
VITE_API_URL=https://yashu-s.vercel.app/
```

Then run:

```bash
npm run dev
```

App will be live at `https://yashu-s.vercel.app/`

---

## ☁️ Deployment

| Service | Platform | Config |
|---|---|---|
| Frontend | Vercel | Root: `frontend` · Build: `npm run build` · Output: `dist` |
| Backend | Render | Root: `backend` · Start: `uvicorn app:app --host 0.0.0.0 --port $PORT` |

Set `VITE_API_URL` as an environment variable in Vercel pointing to your Render backend URL.

---

## 📁 Project Structure

```
ml-cleaning-studio/
├── backend/
│   ├── app.py           # FastAPI routes & session management
│   ├── cleaners.py      # All ML cleaning logic
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   └── api.js       # Axios calls to backend
│   ├── vite.config.js
│   └── package.json
└── runtime.txt          # Python 3.11.9 (for Render)
```

---

<div align="center">

Made by ME 😎· [Live App](https://yashu-s.vercel.app/)

</div>
