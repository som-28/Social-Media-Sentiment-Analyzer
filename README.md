# Social Media Sentiment Analyzer (Full Stack)

A complete full-stack starter project inspired by your synopsis.

## Stack

- Backend: Flask + scikit-learn
- Frontend: React (Vite) + Recharts
- Features: Single-text prediction, CSV batch prediction, sentiment distribution chart

## Project Structure

- backend/
  - app.py
  - requirements.txt
  - services/
    - preprocess.py
    - sentiment.py
- frontend/
  - src/
    - App.jsx
    - api.js
    - components/SentimentChart.jsx
    - styles.css

## Run Locally

### 1) Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Backend runs at http://localhost:5000

### 2) Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at http://localhost:5173

## API

### POST /api/predict

Body:

```json
{
  "text": "I love this product",
  "model": "distilbert"
}
```

Supported model values:
- tfidf_lr
- word2vec_svm
- distilbert

### POST /api/predict-csv

form-data:
- file: csv file (must contain a `text` column)
- model: tfidf_lr | word2vec_svm | distilbert

### GET /api/health

Returns status of backend service.

## Notes

- The DistilBERT path currently uses an ensemble proxy so the app works immediately without heavy model downloads.
- You can later replace it with a real fine-tuned transformer model in backend/services/sentiment.py.
