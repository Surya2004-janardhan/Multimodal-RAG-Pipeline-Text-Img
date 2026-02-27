import torch
from PIL import Image
from typing import List, Union
from sentence_transformers import SentenceTransformer
from pathlib import Path

class MultimodalEmbedder:
    def __init__(self, model_name: str = "clip-ViT-B-32"):
        """
        Initializes the CLIP-based multimodal embedder.
        :param model_name: Name of the pre-trained CLIP model.
        """
        print(f"[*] Loading Multimodal Embedding Model: {model_name}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)
        print(f"[+] Model loaded successfully on {self.device}")

    def encode_text(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """
        Generates embeddings for text chunks.
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # SentenceTransformer handles the encoding to the shared CLIP space automatically
        embeddings = self.model.encode(texts, convert_to_tensor=True, show_progress_bar=False)
        return embeddings

    def encode_image(self, image_paths: Union[str, Path, List[Union[str, Path]]]) -> torch.Tensor:
        """
        Generates embeddings for image files.
        """
        if isinstance(image_paths, (str, Path)):
            image_paths = [image_paths]
            
        images = []
        for path in image_paths:
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
            except Exception as e:
                print(f"[!] Error loading image {path}: {e}")
                
        if not images:
            return torch.tensor([])

        # SentenceTransformer supports encoding PIL images directly for CLIP models
        embeddings = self.model.encode(images, convert_to_tensor=True, show_progress_bar=False)
        return embeddings

if __name__ == "__main__":
    # Quick sanity check
    embedder = MultimodalEmbedder()
    print("MultimodalEmbedder initialized and ready.")
