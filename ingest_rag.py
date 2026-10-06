from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import DOC_DIR, RAG_DIR

def load_documents():
    docs=[]
    for path in DOC_DIR.rglob("*"):
        if not path.is_file(): continue
        try:
            if path.suffix.lower()==".pdf":
                loaded=PyPDFLoader(str(path)).load()
            elif path.suffix.lower() in {".txt",".md"}:
                loaded=TextLoader(str(path),encoding="utf-8").load()
            else:
                continue
            for d in loaded: d.metadata["source_file"]=path.name
            docs += loaded
        except Exception as e:
            print("Skipping",path,e)
    return docs

def main():
    docs=load_documents()
    if not docs: raise RuntimeError("No TXT/MD/PDF files found in data/docs")
    chunks=RecursiveCharacterTextSplitter(
        chunk_size=800,chunk_overlap=120).split_documents(docs)
    embeddings=HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2")
    db=FAISS.from_documents(chunks,embeddings)
    db.save_local(str(RAG_DIR))
    print("Indexed",len(chunks),"chunks into",RAG_DIR)

if __name__=="__main__": main()
