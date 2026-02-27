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
        Performs text-to-multimodal retrieval.
        Returns a ranked list of relevant context items (text, tables, images).
        """
        print(f"[*] Retrieving context for query: '{query}'")
        
        # 1. Encode the text query into the CLIP shared space
        query_embedding = self.embedder.encode_text(query).tolist()
        
        # 2. Query ChromaDB
        # We query for the top results across all modalities stored in the single collection.
        results = self.vector_store.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        
        # 3. Format and Fuse Results
        # ChromaDB results are returned as lists of lists. We flatten and format them.
        formatted_results = []
        
        if not results or not results.get("ids"):
            return []

        ids = results["ids"][0]
        metadatas = results["metadatas"][0]
        documents = results["documents"][0]
        distances = results["distances"][0]
        
        for i in range(len(ids)):
            formatted_results.append({
                "id": ids[i],
                "content": documents[i],
                "metadata": metadatas[i],
                "score": 1.0 - distances[i]  # Convert distance to similarity score
            })
            
        # Results are already ranked by ChromaDB based on similarity (distances)
        print(f"[+] Retrieved {len(formatted_results)} relevant items.")
        return formatted_results

if __name__ == "__main__":
    # Example initialization (mocked or dependent on existing DB)
    print("MultimodalRetriever module loaded.")
