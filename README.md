# 🛡️ FactGuard – Fake News Detector

A production-ready ML web application that classifies news as REAL or FAKE using TF-IDF + sklearn.

---

## 📁 Project Structure

```
factguard/
├── app.py              # Flask application
├── run.py              # Launcher (auto-trains on first run)
├── train_model.py      # ML training pipeline
├── requirements.txt
├── Fake.csv            # Dataset (place here)
├── True.csv            # Dataset (place here)
├── model/              # Auto-created after training
│   ├── factguard_model.pkl
│   ├── tfidf_vectorizer.pkl
│   └── metadata.json
└── templates/
    └── index.html
```

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Make sure Fake.csv and True.csv are in the project folder

# 3. Run (auto-trains model on first launch)
python run.py

# 4. Open browser
http://localhost:5000
```

---

## 🔁 Retrain Model

```bash
python train_model.py
```

---

## 📊 Model Details

- **Algorithm**: Best selected from: Logistic Regression, Decision Tree, Random Forest, Naive Bayes
- **Selection**: By 5-fold cross-validated F1 (not raw test F1)
- **Features**: TF-IDF, 50k features, bigrams, sublinear TF scaling
- **Class imbalance**: Handled via `class_weight='balanced'`
- **Input**: title + article body combined, lowercased, stopwords removed

---

## 🛠️ Stack

| Layer     | Tech |
|-----------|------|
| Backend   | Flask 3.x |
| ML        | scikit-learn, NLTK |
| Features  | TF-IDF (50k, bigrams) |
| Serving   | joblib |
| Frontend  | HTML/CSS/JS (no framework) |

---

## ⚙️ API

### `POST /predict`
```json
{ "text": "Your article text here..." }
```
Response:
```json
{
  "result": "REAL",
  "label": 1,
  "confidence": 97.3,
  "prob_fake": 2.7,
  "prob_real": 97.3,
  "status": "real"
}
```

### `GET /recent`
Returns last 5 checks (in-memory).

### `GET /stats`
Returns model metadata JSON.
