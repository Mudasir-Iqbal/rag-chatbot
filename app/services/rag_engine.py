import tiktoken
from google import genai
from typing import Dict, Any
import traceback
from app.core.config import settings

class RAGEngine:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.max_context_tokens = 25000  # Context budget slightly expanded
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def generate_response(self, query: str) -> Dict[str, Any]:
        try:
            search_results = self.vector_store.similarity_search(query, top_k=5)
            
            retrieved_texts = search_results.get('documents', [[]])[0]
            retrieved_meta = search_results.get('metadatas', [[]])[0]
            distances = search_results.get('distances', [[]])[0]

            SIMILARITY_THRESHOLD = 1.25
            if not distances or distances[0] > SIMILARITY_THRESHOLD:
                return {
                    "answer": "Mujhe provided documents mein is question ka relevant answer nahi mila.",
                    "sources": []
                }

            current_tokens = 0
            context_blocks = []
            sources = []

            for text, meta in zip(retrieved_texts, retrieved_meta):
                text_tokens = self.count_tokens(text)
                if current_tokens + text_tokens > self.max_context_tokens:
                    break
                
                context_blocks.append(f"[{meta['source']} P.{meta['page']}]: {text}")
                sources.append(f"{meta['source']} - Page {meta['page']}")
                current_tokens += text_tokens

            formatted_context = "\n\n".join(context_blocks)

            system_prompt = f"""You are an accurate corporate AI assistant.
Provide a clear, direct, and concise summary or answer based strictly on the provided Context.
Do NOT use outside knowledge.
If the answer cannot be found in the context, state: "Mujhe provided documents mein is question ka relevant answer nahi mila."

Context:
{formatted_context}

User Query: {query}
Answer:"""

            # Increased output token budget to stop response trimming
            response = self.client.models.generate_content(
                model='gemini-3.5-flash',
                contents=system_prompt,
                config={
                    'max_output_tokens': 1000,
                    'temperature': 0.2
                }
            )
            
            answer_text = response.text.strip() if response.text else ""
            
            fallback_msg = "Mujhe provided documents mein is question ka relevant answer nahi mila."
            if fallback_msg in answer_text or not answer_text:
                return {"answer": fallback_msg, "sources": []}

            return {
                "answer": answer_text,
                "sources": list(set(sources))
            }

        except Exception as e:
            print("❌ ERROR IN RAG ENGINE:", str(e))
            traceback.print_exc()
            return {
                "answer": f"Internal Error: {str(e)}",
                "sources": []
            }