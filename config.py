from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CHART_DIR = DATA_DIR / "charts"
DOC_DIR = DATA_DIR / "docs"
MODEL_DIR = ROOT / "models"
RAG_DIR = MODEL_DIR / "faiss_index"
MODEL_PATH = MODEL_DIR / "chart_cnn.pt"
LABELS_PATH = DATA_DIR / "labels.csv"

IMG_SIZE = 224
CLASSES = ["SELL", "HOLD", "BUY"]

for p in [CHART_DIR, DOC_DIR, MODEL_DIR]:
    p.mkdir(parents=True, exist_ok=True)
