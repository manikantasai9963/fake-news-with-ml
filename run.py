"""
FactGuard Launcher
  • Downloads NLTK data if needed
  • Auto-trains model if model/ is empty
  • Starts Flask on http://localhost:5000
"""

import os, sys, subprocess, nltk
from pathlib import Path

BASE_DIR  = Path(__file__).resolve().parent
MODEL_PKL = BASE_DIR / "model" / "factguard_model.pkl"

# ── NLTK data ──────────────────────────────────────────────────────────────────
for pkg, kind in [('stopwords','corpora'), ('punkt','tokenizers'), ('punkt_tab','tokenizers')]:
    try:
        nltk.data.find(f'{kind}/{pkg}')
    except LookupError:
        nltk.download(pkg, quiet=True)

# ── Auto-train if model missing ────────────────────────────────────────────────
if not MODEL_PKL.exists():
    print("\n  Model not found — training now (this takes ~2-5 minutes)…\n")
    result = subprocess.run([sys.executable, str(BASE_DIR / "train_model.py")])
    if result.returncode != 0:
        print("\n  ✗ Training failed. Check that Fake.csv and True.csv are in the project folder.")
        sys.exit(1)

# ── Launch Flask ───────────────────────────────────────────────────────────────
from app import app
print("\n  🛡️  FactGuard running at  http://localhost:5000\n")
app.run(debug=False, host='0.0.0.0', port=5000)
