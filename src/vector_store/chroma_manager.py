import os
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ChromaManager:
    def __init__(self, persist_directory: str = None, collection_name: str = "multimodal_rag"):
        """
        Initializes the ChromaDB manager.
        :param persist_directory: Path to the local directory for persistence.
        :param collection_name: Name of the collection to use.
        """
        base_persist = persist_directory or os.getenv("VECTOR_DB_PATH", "./data/chroma")
        self.persist_directory = Path(base_persist)
        
        # Initialize Persistent Client (ChromaDB >= 0.4.x)
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        
        # Get or create the collection
        # Note: CLIP embeddings are 512-dimensional for ViT-B-32
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"} # Using cosine similarity for CLIP
        )
        print(f"[+] ChromaDB initialized at {self.persist_directory}")
        print(f"[+] Using collection: {collection_name}")

    def add_embeddings(self, 
                       ids: List[str], 
                       embeddings: List[List[float]], 
                       metadatas: List[Dict[str, Any]], 
                       documents: List[str]):
        """
        Adds embeddings and metadata to the collection.
        """
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )
            print(f"[+] Successfully added {len(ids)} items to ChromaDB.")
        except Exception as e:
            print(f"[!] Error adding items to ChromaDB: {e}")

    def query(self, 
              query_embeddings: List[List[float]], 
              n_results: int = 5, 
              where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Queries the collection with given embeddings.
        """
        try:
            results = self.collection.query(
                query_embeddings=query_embeddings,
                n_results=n_results,
                where=where,
                include=["metadatas", "documents", "distances"]
            )
            return results
        except Exception as e:
            print(f"[!] Error querying ChromaDB: {e}")
            return {}

    def get_count(self) -> int:
        """
        Returns the number of items in the collection.
        """
        return self.collection.count()

if __name__ == "__main__":
    # Quick sanity check
    manager = ChromaManager()
    print(f"Collection count: {manager.get_count()}")
