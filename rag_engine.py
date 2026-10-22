import os
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import requests

OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"  # أو أي نموذج متوفر محلياً

class HybridRAGEngine:
    def __init__(self, db_path="./chroma_local_db"):
        print("Initializing embedding model...")
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        
        print("Initializing local ChromaDB...")
        self.chroma_client = PersistentClient(path=db_path)
        self.collection = self.chroma_client.get_or_create_collection(name="local_knowledge_base")
        
        self.documents_store = []
        self._load_existing_documents()

    def _load_existing_documents(self):
        data = self.collection.get()
        if data and "documents" in data:
            self.documents_store = data["documents"]

    def add_document(self, doc_id: str, text: str, metadata: dict = None):
        embedding = self.encoder.encode(text).tolist()
        self.collection.upsert(
            ids=[doc_id],
            documents=[text],
            embeddings=[embedding],
            metadatas=[metadata or {}]
        )
        if text not in self.documents_store:
            self.documents_store.append(text)

    def hybrid_search(self, query: str, top_k: int = 3):
        if not self.documents_store:
            return []

        # 1. البحث الدلالي عبر ChromaDB
        query_embedding = self.encoder.encode(query).tolist()
        semantic_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, len(self.documents_store))
        )
        
        semantic_docs = semantic_results.get("documents", [[]])[0]

        # 2. البحث النصي التقليدي عبر BM25
        tokenized_corpus = [doc.split(" ") for doc in self.documents_store]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = query.split(" ")
        bm25_scores = bm25.get_scores(tokenized_query)
        
        # دمج النتائج (Hybrid Fusion)
        combined_results = list(set(semantic_docs))
        return combined_results[:top_k]

    def generate_answer(self, query: str) -> dict:
        relevant_docs = self.hybrid_search(query)
        context = "\n---\n".join(relevant_docs) if relevant_docs else "No specific context found."

        prompt = f"""
You are an expert local AI assistant. Answer the user query strictly based on the provided context below. If the answer cannot be found, state that you don't know based on local files.

Context:
{context}

User Query:
{query}
"""
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        }
        
        try:
            response = requests.post(OLLAMA_API_URL, json=payload, timeout=120)
            if response.status_code == 200:
                answer = response.json().get("response", "No response generated.")
                return {"answer": answer, "sources": relevant_docs}
            else:
                return {"answer": f"Ollama Error: {response.text}", "sources": []}
        except Exception as e:
            return {"answer": f"Connection error with Ollama: {str(e)}", "sources": []}
