import os
import easyocr
import hashlib
from typing import Dict, Any, List
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ImageProcessor:
    def __init__(self, output_dir: str = None):
        """
        Initializes the ImageProcessor with EasyOCR.
        """
        base_output = output_dir or os.getenv("PROCESSED_DATA_PATH", "./data/processed")
        self.output_dir = Path(base_output)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize EasyOCR reader for English
        # Note: This will download models on the first run (~100MB)
        print("[*] Initializing EasyOCR Reader...")
        self.reader = easyocr.Reader(['en']) 

    def process_image(self, image_path: str) -> Dict[str, Any]:
        """
        Processes a standalone image: runs OCR and returns structured content.
        """
        doc_id = os.path.basename(image_path)
        
        try:
            print(f"[*] Running OCR on: {doc_id}")
            # Run OCR on the image
            result = self.reader.readtext(image_path, detail=0)
            ocr_text = " ".join(result)
            
            return {
                "doc_id": doc_id,
                "page": 1,
                "type": "image",
                "content": str(image_path),
                "ocr_text": ocr_text,
                "metadata": {
                    "source": image_path,
                    "page_number": 1,
                    "content_type": "image",
                    "image_path": str(image_path),
                    "ocr_text": ocr_text
                }
            }
        except Exception as e:
            print(f"[!] Error processing image {image_path}: {e}")
            return {}

    def ocr_only(self, image_path: str) -> str:
        """
        Runs OCR and returns only the text. Useful for images extracted from PDFs.
        """
        try:
            result = self.reader.readtext(str(image_path), detail=0)
            return " ".join(result)
        except Exception as e:
            print(f"[!] Error in ocr_only for {image_path}: {e}")
            return ""

if __name__ == "__main__":
    # Quick sanity check
    processor = ImageProcessor()
    print("ImageProcessor initialized with EasyOCR.")
