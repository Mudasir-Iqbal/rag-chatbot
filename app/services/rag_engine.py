import google.generativeai as genai
from app.services.vector_store import VectorStoreManager
from app.core.config import settings

class RAGEngine:
    def __init__(self, vector_store: VectorStoreManager):
        self.vector_store = vector_store
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model = genai.GenerativeModel('gemini-3.5-flash')

    def generate_response(self, query: str) -> Dict[str, Any]:
        # Step 1: Retrieval
        search_results = self.vector_store.similarity_search(query, top_k=3)
        retrieved_texts = search_results.get('documents', [[]])[0]
        retrieved_meta = search_results.get('metadatas', [[]])[0]

        if not retrieved_texts:
            return {
                "answer": "Mujhe provided documents mein is question ka relevant answer nahi mila.",
                "sources": []
            }

        # Context Formatting
        context_blocks = []
        sources = []
        for text, meta in zip(retrieved_texts, retrieved_meta):
            context_blocks.append(f"[Source: {meta['source']} - Page {meta['page']}]\n{text}")
            sources.append(f"{meta['source']} - Page {meta['page']}")

        formatted_context = "\n\n---\n\n".join(context_blocks)

        # Step 2: System Prompting with Strict Hallucination Control
        system_prompt = f"""
You are an enterprise AI assistant strictly operating under Retrieval-Augmented Generation (RAG) constraints.

Strict Rules:
1. Answer the question ONLY using the provided Context below.
2. Do NOT use outside knowledge or speculate.
3. If the answer is NOT explicitly present in the Context, respond EXACTLY with:
   "Mujhe provided documents mein is question ka relevant answer nahi mila."
4. Always maintain factual precision.

Context:
{formatted_context}

User Query: {query}
Answer:
"""

        response = self.model.generate_content(system_prompt)
        
        return {
            "answer": response.text.strip(),
            "sources": list(set(sources))
        }