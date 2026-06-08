"""
FactGuard – Flask Application
"""

import os, re, string, json, joblib
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from nltk.corpus import stopwords

BASE_DIR  = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

app = Flask(__name__)

# ── Load model ─────────────────────────────────────────────────────────────────
try:
    model      = joblib.load(MODEL_DIR / "factguard_model.pkl")
    vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")
    with open(MODEL_DIR / "metadata.json") as f:
        META = json.load(f)
    MODEL_READY = True
except FileNotFoundError:
    MODEL_READY = False
    META = {}

STOP_WORDS    = set(stopwords.words('english'))
recent_checks = []          # in-memory last-10

def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return ' '.join(t for t in text.split() if t not in STOP_WORDS and len(t) > 2)

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html', meta=META, model_ready=MODEL_READY)

@app.route('/predict', methods=['POST'])
def predict():
    if not MODEL_READY:
        return jsonify({"error": "Model not trained yet. Run python train_model.py first."}), 503

    data = request.get_json()
    text = (data.get('text') or '').strip()

    if len(text) < 20:
        return jsonify({"error": "Please enter at least 20 characters."}), 400
    if len(text) > 50_000:
        return jsonify({"error": "Text too long (max 50,000 characters)."}), 400

    clean  = preprocess(text)
    vec    = vectorizer.transform([clean])
    label  = int(model.predict(vec)[0])
    proba  = model.predict_proba(vec)[0]
    conf   = float(max(proba)) * 100

    result = "REAL" if label == 1 else "FAKE"
    status = "uncertain" if conf < 70 else ("real" if label == 1 else "fake")

    snippet = text[:80] + ("…" if len(text) > 80 else "")
    recent_checks.insert(0, {"text": snippet, "result": result, "status": status, "conf": round(conf,1)})
    if len(recent_checks) > 10:
        recent_checks.pop()

    return jsonify({
        "result":     result,
        "label":      label,
        "confidence": round(conf, 1),
        "prob_fake":  round(float(proba[0]) * 100, 1),
        "prob_real":  round(float(proba[1]) * 100, 1),
        "status":     status,
    })

@app.route('/recent')
def recent():
    return jsonify(recent_checks[:5])

@app.route('/stats')
def stats():
    return jsonify(META)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
