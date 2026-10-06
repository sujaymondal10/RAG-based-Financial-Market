from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from config import RAG_DIR

class FinancialRAG:
    def __init__(self):
        if not RAG_DIR.exists():
            raise RuntimeError("Run: python src/ingest_rag.py")
        embeddings=HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.db=FAISS.load_local(str(RAG_DIR),embeddings,
                                 allow_dangerous_deserialization=True)

    def retrieve(self,query,k=4):
        docs=self.db.similarity_search(query,k=k)
        return [{"content":d.page_content,
                 "source":d.metadata.get("source_file","unknown")} for d in docs]
