import os
import base64
from typing import List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables
load_dotenv()

class MultimodalGenerator:
    def __init__(self, model_name: str = "meta-llama/llama-4-scout-17b-16e-instruct"):
        """
        Initializes the generator using Groq via LangChain.
        """
        self.api_key = os.getenv("GROQ_API_KEY")
        self.model_name = model_name
        
        if not self.api_key:
            print("[!] Warning: GROQ_API_KEY not found in environment.")
            
        self.llm = ChatGroq(
            model=self.model_name,
            groq_api_key=self.api_key,
            temperature=0.1
        )
        print(f"[+] Generator initialized for Groq: {self.model_name}")

    def _encode_image_to_base64(self, image_path: str) -> str:
        """
        Converts an image file to a base64 string.
        """
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except Exception as e:
            print(f"[!] Error encoding image {image_path}: {e}")
            return ""

    def generate_answer(self, query: str, context_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generates a grounded answer using Groq.
        Note: If the model doesn't support vision, it uses the OCR text in the prompt.
        """
        print(f"[*] Generating answer with Groq for query: '{query}'")
        
        text_context = []
        image_contents = []
        source_refs = []
        
        for item in context_items:
            metadata = item["metadata"]
            source_refs.append(metadata)
            
            if metadata["content_type"] in ["text", "table"]:
                text_context.append(f"Source ({metadata['source']}, Page {metadata['page_number']}):\n{item['content']}")
            elif metadata["content_type"] == "image":
                # For Groq Llama models (non-vision), we rely heavily on the OCR text we extracted
                ocr_text = metadata.get("ocr_text", "No text in image")
                text_context.append(f"Image Content (Source: {metadata['source']}, Page: {metadata['page_number']}):\n{ocr_text}")
                
                # We still try to encode the image in case it's a vision model
                img_b64 = self._encode_image_to_base64(metadata["image_path"])
                if img_b64:
                    image_contents.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                    })

        context_str = "\n\n".join(text_context)
        
        prompt_text = f"""You are an extremely advanced Multimodal AI Researcher.
Below is a mix of Text, Tables, and IMAGE OCR DATA retrieved from papers.

CONTEXT:
{context_str}

USER QUERY:
{query}

INSTRUCTIONS:
1. FOCUS: Pay ultra-close attention to the numbers and technical terms in the Image OCR data.
2. CITATION: Cite specific pages and sources.
3. LOGIC: If the query is about a diagram, use the OCR text provided in the context to reconstruct the logic.
4. TONE: Professional and highly technical.

Final Response:"""

        # Build message elements
        message_elements = [{"type": "text", "text": prompt_text}]
        
        # Only add images if the model supports it. Many Llama models on Groq are text-only.
        # However, we'll add them and let the API decide or fallback to OCR.
        if "vision" in self.model_name.lower():
            message_elements.extend(image_contents)
        
        try:
            message = HumanMessage(content=message_elements)
            response = self.llm.invoke([message])
            
            return {
                "answer": response.content,
                "sources": source_refs
            }
        except Exception as e:
            print(f"[!] Error generating with Groq: {e}")
            return {
                "answer": f"Error: Groq failed (Model: {self.model_name}). Check API Key and Model ID. Details: {str(e)}",
                "sources": source_refs
            }

if __name__ == "__main__":
    # Example usage
    generator = MultimodalGenerator()
    print("MultimodalGenerator module loaded.")
