import os
import fitz  # PyMuPDF
import hashlib
from typing import List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv
from unstructured.partition.pdf import partition_pdf

# Load environment variables
load_dotenv()

class PDFParser:
    def __init__(self, output_dir: str = None):
        """
        Initializes the PDFParser.
        :param output_dir: Directory where processed assets (like images) will be stored.
        """
        base_output = output_dir or os.getenv("PROCESSED_DATA_PATH", "./data/processed")
        self.output_dir = Path(base_output)
        self.image_dir = self.output_dir / "images"
        
        # Ensure directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.image_dir.mkdir(parents=True, exist_ok=True)

    def extract_content(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Parses a PDF file and extracts text, structured tables, and images.
        Uses unstructured for layout/tables and PyMuPDF for images.
        """
        doc_id = os.path.basename(pdf_path)
        chunks = []
        
        print(f"[*] Processing document: {doc_id}")
        
        try:
            # 1. Layout-aware partitioning (Text and Tables)
            # strategy="auto" will use "fast" (no OCR) if text is detectable, 
            # or "hi_res" (OCR) only if needed.
            elements = partition_pdf(
                filename=pdf_path,
                strategy="auto",
                infer_table_structure=True,
                extract_images_in_pdf=False,
            )

            for i, element in enumerate(elements):
                element_type = element.category.lower()
                page_number = element.metadata.page_number if element.metadata.page_number else 1
                content_type = "table" if element_type == "table" else "text"
                
                chunks.append({
                    "doc_id": doc_id,
                    "page": page_number,
                    "type": content_type,
                    "content": str(element),
                    "metadata": {
                        "source": pdf_path,
                        "page_number": page_number,
                        "content_type": content_type,
                        "element_id": f"{doc_id}_el_{i}"
                    }
                })
        except Exception as e:
            print(f"[!] Unstructured parsing failed for {doc_id} (Text/Tables might be missing): {e}")

        try:
            # 2. Extract raw images using PyMuPDF (No Tesseract needed)
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc[page_num]
                image_list = page.get_images(full=True)
                
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    img_hash = hashlib.md5(image_bytes).hexdigest()
                    img_filename = f"{doc_id.replace('.', '_')}_p{page_num+1}_img{img_index}_{img_hash[:8]}.{image_ext}"
                    img_path = self.image_dir / img_filename
                    
                    if not img_path.exists():
                        with open(img_path, "wb") as f:
                            f.write(image_bytes)
                    
                    chunks.append({
                        "doc_id": doc_id,
                        "page": page_num + 1,
                        "type": "image",
                        "content": str(img_path),
                        "metadata": {
                            "source": pdf_path,
                            "page_number": page_num + 1,
                            "content_type": "image",
                            "image_path": str(img_path)
                        }
                    })
            doc.close()
        except Exception as e:
            print(f"[!] Image extraction failed for {doc_id}: {e}")

        if chunks:
            print(f"[+] Successfully extracted {len(chunks)} elements from {doc_id}")
            return chunks
        else:
            return []

if __name__ == "__main__":
    # Quick sanity check logic
    parser = PDFParser()
    print("PDFParser initialized and ready.")
