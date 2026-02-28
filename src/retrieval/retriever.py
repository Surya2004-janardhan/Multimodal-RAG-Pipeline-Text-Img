from typing import List, Dict, Any
from src.embeddings.model_loader import MultimodalEmbedder
from src.vector_store.chroma_manager import ChromaManager

class MultimodalRetriever:
    def __init__(self, embedder: MultimodalEmbedder, vector_store: ChromaManager):
        """
        Initializes the retriever with an embedder and a vector store.
        """
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Performs text-to-multimodal retrieval using LangChain's similarity search.
        """
        print(f"[*] Retrieving context for query: '{query}'")
        
        # 1. Encode the text query into the CLIP shared space
        query_embedding = self.embedder.encode_text(query).cpu().detach().tolist()[0]
        
        # 2. Query LangChain Chroma
        # Using similarity_search_with_relevance_scores or similar
        docs_with_scores = self.vector_store.vectorstore.similarity_search_with_relevance_scores(
            query=query, # This will use the internal embedding_function if provided
            k=n_results
        )
        
        # 3. Format Results
        formatted_results = []
        for doc, score in docs_with_scores:
            formatted_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": score
            })
            
        print(f"[+] Retrieved {len(formatted_results)} relevant items.")
        return formatted_results

if __name__ == "__main__":
    print("MultimodalRetriever module loaded.")
