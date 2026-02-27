import os
import requests
import json
import base64
from typing import List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class MultimodalGenerator:
    def __init__(self, model_name: str = "llava"):
        """
        Initializes the generator using Ollama.
        """
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model_name = model_name
        print(f"[+] Generator initialized for model: {self.model_name} at {self.base_url}")

    def _encode_image_to_base64(self, image_path: str) -> str:
        """
        Converts an image file to a base64 string for Ollama.
        """
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except Exception as e:
            print(f"[!] Error encoding image {image_path}: {e}")
            return ""

    def generate_answer(self, query: str, context_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generates a visually grounded answer using LLaVA via Ollama.
        """
        print(f"[*] Generating answer for query: '{query}'")
        
        # 1. Separate context into text and images
        text_context = []
        images_base64 = []
        source_refs = []
        
        for item in context_items:
            metadata = item["metadata"]
            source_refs.append(metadata)
            
            if metadata["content_type"] in ["text", "table"]:
                text_context.append(f"Source ({metadata['source']}, Page {metadata['page_number']}):\n{item['content']}")
            elif metadata["content_type"] == "image":
                img_b64 = self._encode_image_to_base64(metadata["image_path"])
                if img_b64:
                    images_base64.append(img_b64)
                    text_context.append(f"Image Source ({metadata['source']}, Page {metadata['page_number']}) provided in visual context.")

        # 2. Construct the prompt
        context_str = "\n\n".join(text_context)
        prompt = f"""You are an advanced AI assistant that reasons over multimodal documents.
Below is relevant context (text, tables, and images) retrieved from various documents.

CONTEXT:
{context_str}

USER QUERY:
{query}

INSTRUCTIONS:
1. Answer the query accurately based ON THE PROVIDED CONTEXT AND IMAGES.
2. Be "visually grounded": If your answer is based on a chart, diagram, or image, explicitly say "As seen in the image on page X of document Y...".
3. Use a professional and clear tone.
4. If the context doesn't contain the answer, say you don't have enough information.

ANSWER:"""

        # 3. Call Ollama API
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "images": images_base64,
            "stream": False,
            "options": {
                "temperature": 0.2, # Lower temperature for factual accuracy
            }
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=30 # 30s timeout for local VLM
            )
            response.raise_for_status()
            result = response.json()
            
            return {
                "answer": result.get("response", "No response from model."),
                "sources": source_refs
            }
            
        except requests.exceptions.RequestException as e:
            print(f"[!] Error communicating with Ollama: {e}")
            return {
                "answer": f"Error: Could not connect to the Vision-LLM (Ollama). Ensure it is running at {self.base_url}.",
                "sources": source_refs
            }

if __name__ == "__main__":
    # Example usage
    generator = MultimodalGenerator()
    print("MultimodalGenerator module loaded.")
