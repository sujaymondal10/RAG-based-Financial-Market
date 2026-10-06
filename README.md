# RAG-based Visual Financial Market Intelligence Platform

End-to-end project:
Yahoo Finance -> OHLCV -> chart generation -> OpenCV preprocessing ->
PyTorch CNN -> BUY/HOLD/SELL signal, plus LangChain/FAISS RAG over
financial TXT/MD/PDF documents -> Streamlit decision-support UI.

## Run

1. Create a virtual environment and install:
   `pip install -r requirements.txt`

2. Generate chart data:
   `python src/make_dataset.py --tickers AAPL MSFT NVDA --period 5y --interval 1d`

3. Train:
   `python src/train_cnn.py --epochs 15 --batch-size 32`

4. Put your own financial PDFs/TXT/MD files into `data/docs/`, then:
   `python src/ingest_rag.py`

5. Start:
   `streamlit run src/app.py`

6. Evaluate:
   `python src/evaluate.py`

7. Benchmark:
   `python src/benchmark.py --n 100`

Important: only claim the resume metrics after actually measuring them.
Model probability/confidence is not the same thing as trading accuracy.
This is decision-support/research software, not a guarantee of returns.
