import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any

class VectorStoreManager:
    def __init__(self, collection_name: str = "rag_documents"):
        self.client = chromadb.PersistentClient(path="./vector_db")
        # Lightweight & efficient embedding model
        self.emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.emb_fn
        )

    def add_documents(self, docs: List[Dict[str, Any]]):
        texts = [doc["text"] for doc in docs]
        metadatas = [doc["metadata"] for doc in docs]
        ids = [f"{doc['metadata']['source']}_p{doc['metadata']['page']}_{i}" for i, doc in enumerate(docs)]
        
        self.collection.upsert(
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )

    def similarity_search(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        return self.collection.query(
            query_texts=[query],
            n_results=top_k
        )