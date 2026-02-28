import os
import base64
from typing import List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables
load_dotenv()

class MultimodalGenerator:
    def __init__(self, model_name: str = "gemini-1.5-flash"):
        """
        Initializes the generator using Gemini via LangChain.
        """
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.model_name = model_name
        
        if not self.api_key:
            print("[!] Warning: GOOGLE_API_KEY not found in environment.")
            
        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=self.api_key,
            temperature=0.1
        )
        print(f"[+] Generator initialized for Gemini: {self.model_name}")

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
        Generates a grounded answer using Gemini via LangChain.
        """
        print(f"[*] Generating answer with Gemini for query: '{query}'")
        
        text_context = []
        image_contents = []
        source_refs = []
        
        for item in context_items:
            metadata = item["metadata"]
            source_refs.append(metadata)
            
            if metadata["content_type"] in ["text", "table"]:
                text_context.append(f"Source ({metadata['source']}, Page {metadata['page_number']}):\n{item['content']}")
            elif metadata["content_type"] == "image":
                img_b64 = self._encode_image_to_base64(metadata["image_path"])
                if img_b64:
                    # LangChain Google GenAI expects image content in a specific format
                    image_contents.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                    })
                    text_context.append(f"Image Source ({metadata['source']}, Page {metadata['page_number']}) provided in visual context.")

        context_str = "\n\n".join(text_context)
        
        # Build the message content for Gemini
        message_elements = [
            {
                "type": "text",
                "text": f"""You are an advanced Multimodal Financial Analyst.
Below is a mix of Text, Structured Tables, and Images retrieved from relevant documents.

CONTEXT:
{context_str}

USER QUERY:
{query}

CRITICAL INSTRUCTIONS:
1. PRE-ANALYSIS: Analyze provided images/tables for relevant numbers or trends.
2. VISUAL GROUNDING: Cite specific images or tables explicitly (e.g., "According to the chart on page 4...").
3. ACCURACY: If information is missing, say you don't have enough detail.
4. FORMAT: Use structured bullet points for data summaries."""
            }
        ]
        # Append all identified images
        message_elements.extend(image_contents)
        
        try:
            message = HumanMessage(content=message_elements)
            response = self.llm.invoke([message])
            
            return {
                "answer": response.content,
                "sources": source_refs
            }
        except Exception as e:
            print(f"[!] Error generating with Gemini: {e}")
            return {
                "answer": f"Error: Gemini failed to generate a response. {str(e)}",
                "sources": source_refs
            }

if __name__ == "__main__":
    # Example usage
    generator = MultimodalGenerator()
    print("MultimodalGenerator module loaded.")
