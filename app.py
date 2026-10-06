import sys
from pathlib import Path
import streamlit as st
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent))
from config import MODEL_PATH, CHART_DIR
from market_data import download_ohlcv
from chart_generator import save_price_chart
from predict import ChartPredictor
from rag import FinancialRAG

st.set_page_config(page_title="Visual Financial Intelligence",layout="wide")
st.title("RAG-based Visual Financial Market Intelligence")
st.caption("PyTorch CNN + OpenCV-style preprocessing + LangChain/FAISS RAG")

if not MODEL_PATH.exists():
    st.error("Train the CNN first: python src/train_cnn.py")
    st.stop()

@st.cache_resource
def predictor(): return ChartPredictor()

@st.cache_resource
def rag(): return FinancialRAG()

ticker=st.text_input("Ticker","AAPL").upper().strip()
question=st.text_area(
    "Question",
    "What factors should I consider before acting on the current chart signal?"
)

if st.button("Analyze"):
    try:
        df=download_ohlcv(ticker,"1y","1d")
        chart_path=CHART_DIR/"runtime"/f"{ticker}.png"
        save_price_chart(df.tail(60),chart_path,f"{ticker} recent chart")
        image=Image.open(chart_path)

        result=predictor().predict(image)
        docs=rag().retrieve(f"{ticker}: {question}",4)

        c1,c2=st.columns(2)
        with c1:
            st.image(image,caption=f"{ticker} recent chart")
        with c2:
            st.metric("Signal",result["signal"],
                      f"{result['confidence']:.2%} model probability")
            st.write("Class probabilities:",result["probabilities"])
            st.write("Inference latency:",result["latency_seconds"],"seconds")

        st.subheader("Retrieved financial context")
        for d in docs:
            st.markdown(f"**{d['source']}**")
            st.write(d["content"][:1000])

        st.warning(
            "Research/decision support only. A CNN probability is not a "
            "guarantee of future returns."
        )
    except Exception as e:
        st.exception(e)
