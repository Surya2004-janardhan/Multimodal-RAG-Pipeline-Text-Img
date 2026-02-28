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
        
        # --- ULTIMATE RESEARCH PROMPT & GUARDRAILS ---
        system_prompt = SystemMessage(content="""You are a world-class AI Research Assistant. Your mission is to provide expert-level technical analysis of seminal ML research papers.

CORE OPERATIONAL RULES:
1. FOCUS ON SEMINAL PAPERS: You specialize in papers like Attention Is All You Need, Adam, ImageNet (Russakovsky), ResNet, Dropout, Word2Vec, etc.
2. TECHNICAL DEPTH: If a user asks for weight updates, experimental results, or architecture components, provide the mathematical or structural details.
3. ADAPTIVE CONTEXT: Use the provided context as your ground truth. If the context mentions a specific concept (like 'Adam weight update' or 'Transformer multi-head attention') but doesn't show the full equation, you MAY use your expert internal knowledge of those specific papers to provide the complete technical explanation, as long as it aligns perfectly with the paper's original work.
4. GUARDRAILS: If the query is completely unrelated to AI/ML research (e.g., general life advice, non-AI coding, recipes), politely decline.
5. CITATION: Always cite the paper and page number from the context.
""")

        human_prompt = f"""CONTEXT FROM RESEARCH PAPERS (Text & Image OCR):
{context_str}

USER QUERY:
{query}

TECHNICAL INSTRUCTIONS:
- Analyze the Image OCR carefully for specialized symbols, variables, and diagram components.
- Compare findings across multiple sources if relevant.
- Provide a structured, expert-level response. Use LaTeX for math if necessary.

Final Response:"""

        # Build message elements for the HumanMessage
        human_message_elements = [{"type": "text", "text": human_prompt}]
        
        # Only add images if the model supports it. 
        if "vision" in self.model_name.lower():
            human_message_elements.extend(image_contents)
        
        try:
            messages = [
                system_prompt,
                HumanMessage(content=human_message_elements)
            ]
            response = self.llm.invoke(messages)
            
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
