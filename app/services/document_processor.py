from typing import List, Dict, Any
import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentProcessor:

    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,      # Smaller chunks = Less redundant tokens
            chunk_overlap=50,     # Reduced overlap overhead
            separators=["\n\n", "\n", " ", ""]
        )
        # self.splitter = RecursiveCharacterTextSplitter(
        #     chunk_size=chunk_size,
        #     chunk_overlap=chunk_overlap,
        #     separators=["\n\n", "\n", " ", ""]
        # )

    def extract_text_with_metadata(self, file_path: str, filename: str) -> List[Dict[str, Any]]:
        doc = pymupdf.open(file_path)
        # doc = fitz.open(file_path)
        documents = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text").strip()
            if not text:
                continue
                
            chunks = self.splitter.split_text(text)
            for chunk in chunks:
                documents.append({
                    "text": chunk,
                    "metadata": {
                        "source": filename,
                        "page": page_num + 1
                    }
                })
        doc.close()
        return documents