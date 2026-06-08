"""
FactGuard – ML Training Pipeline
Run this once before starting the Flask app.
Usage: python train_model.py
"""

import os, re, string, json, joblib, warnings, time
from pathlib import Path

import pandas as pd
import numpy as np
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings('ignore')

BASE_DIR  = Path(__file__).resolve().parent
FAKE_PATH = BASE_DIR / "Fake.csv"
TRUE_PATH = BASE_DIR / "True.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)

STOP_WORDS = set(stopwords.words('english'))

# ── 1. Load ────────────────────────────────────────────────────────────────────
print("\n── Loading datasets ──────────────────────────────────")
fake_df = pd.read_csv(FAKE_PATH); fake_df['label'] = 0
true_df = pd.read_csv(TRUE_PATH); true_df['label'] = 1
df = pd.concat([fake_df, true_df], ignore_index=True)
df.dropna(subset=['text', 'title'], inplace=True)
df.drop_duplicates(subset=['text'], inplace=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df['content'] = df['title'].fillna('') + " " + df['text'].fillna('')

fake_n, real_n = (df['label']==0).sum(), (df['label']==1).sum()
print(f"  Total: {len(df):,}  |  Fake: {fake_n:,}  |  Real: {real_n:,}")

# ── 2. Preprocess ──────────────────────────────────────────────────────────────
def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return ' '.join(t for t in text.split() if t not in STOP_WORDS and len(t) > 2)

print("  Preprocessing …")
df['clean_text'] = df['content'].apply(preprocess)

# ── 3. Vectorize ───────────────────────────────────────────────────────────────
X, y = df['clean_text'], df['label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

vectorizer = TfidfVectorizer(max_features=50_000, ngram_range=(1,2),
                             min_df=2, sublinear_tf=True)
X_tr = vectorizer.fit_transform(X_train)
X_te = vectorizer.transform(X_test)
print(f"  Vocab: {len(vectorizer.vocabulary_):,}  Train: {X_tr.shape}  Test: {X_te.shape}")

# ── 4. Train & evaluate ────────────────────────────────────────────────────────
classes = np.unique(y_train)
cw = dict(zip(classes, compute_class_weight('balanced', classes=classes, y=y_train)))

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, C=5, solver='lbfgs', class_weight=cw),
    "Decision Tree":       DecisionTreeClassifier(max_depth=20, min_samples_leaf=10, random_state=42, class_weight=cw),
    "Random Forest":       RandomForestClassifier(n_estimators=200, min_samples_leaf=5, random_state=42, n_jobs=-1, class_weight=cw),
    "Naive Bayes":         MultinomialNB(alpha=0.1),
}

print("\n── Training models ───────────────────────────────────")
results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    t0 = time.time()
    model.fit(X_tr, y_train)
    preds = model.predict(X_te)
    cv_f1 = cross_val_score(model, X_tr, y_train, cv=cv, scoring='f1', n_jobs=-1)
    results[name] = {
        "accuracy":   round(accuracy_score(y_test, preds)*100, 2),
        "precision":  round(precision_score(y_test, preds, average='weighted')*100, 2),
        "recall":     round(recall_score(y_test, preds, average='weighted')*100, 2),
        "f1":         round(f1_score(y_test, preds)*100, 2),
        "cv_f1_mean": round(cv_f1.mean()*100, 2),
        "cv_f1_std":  round(cv_f1.std()*100, 2),
        "cm":         confusion_matrix(y_test, preds).tolist(),
        "train_time": round(time.time()-t0, 1),
    }
    print(f"  {name:<22} acc={results[name]['accuracy']}%  "
          f"f1={results[name]['f1']}%  cv={results[name]['cv_f1_mean']}±{results[name]['cv_f1_std']}%  "
          f"[{results[name]['train_time']}s]")

best_name  = max(results, key=lambda n: results[n]['cv_f1_mean'])
best_model = models[best_name]
print(f"\n  ★ Best: {best_name}  CV-F1={results[best_name]['cv_f1_mean']}%")

# ── 5. Save ────────────────────────────────────────────────────────────────────
joblib.dump(best_model, MODEL_DIR / "factguard_model.pkl")
joblib.dump(vectorizer, MODEL_DIR / "tfidf_vectorizer.pkl")

metadata = {
    "best_model": best_name,
    "accuracy":   results[best_name]['accuracy'],
    "f1":         results[best_name]['f1'],
    "cv_f1_mean": results[best_name]['cv_f1_mean'],
    "cv_f1_std":  results[best_name]['cv_f1_std'],
    "precision":  results[best_name]['precision'],
    "recall":     results[best_name]['recall'],
    "dataset_size": int(len(df)),
    "fake_count": int(fake_n),
    "real_count": int(real_n),
    "all_results": {k: {x:v for x,v in r.items() if x!='cm'} for k,r in results.items()},
}
with open(MODEL_DIR / "metadata.json", 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\n  Saved → {MODEL_DIR}/")
print("  Done ✓")
