import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import re
import nltk
nltk.download('punkt', quiet=True)

# Imports for semantic chunking
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings


# Local application imports
from backend.app.core.config import settings
from backend.app.services.vstore_svc import VectorStoreService
from backend.app.core.exceptions import DocumentProcessingError, EmbeddingError, RetrievalError, LLMError

class DocParserFastService:
    """
    Service to parse documents (PDFs, images), extract text,
    perform STRUCTURE-AWARE semantic chunking, and add chunks to the vector store.
    """
    def __init__(self, vector_store_service: VectorStoreService):
        self.vector_store_service = vector_store_service
        self.ocr_processor = None  # Will be initialized on first use

        # Use Instructor embeddings model for more citation-aware and instruction-aligned chunks
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="hkunlp/instructor-xl",  # Free and powerful model from HuggingFace
                model_kwargs={'device': 'cpu'},
                encode_kwargs={"normalize_embeddings": True}
            )

            # NOTE: Using semantic chunking with INSTRUCTOR may require a custom splitter; fallback used here
            self.semantic_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,  # smaller for citation-aware precision
                chunk_overlap=200,
                length_function=len,
                is_separator_regex=False,
                separators=["\n\n", "\n", ". ", "? ", "! ", ",", " ", ""]
            )
        except Exception as e:
            print(f"Error initializing embedding model: {e}. Using fallback splitter.")
            self.embeddings = None
            self.semantic_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                is_separator_regex=False,
                separators=["\n\n", "\n", ". ", "? ", "! ", ",", " ", ""]
            )

    def _is_heading(self, text: str) -> bool:
        """
        Improved heading detection:
        - Short lines (<= 10 words)
        - Mostly capitalized or all uppercase
        - Ends with colon or is bold-like
        - Matches common heading patterns (e.g., numbered, roman numerals)
        """
        text = text.strip()
        if not text:
            return False
        words = text.split()
        if len(words) > 10:
            return False
        # Regex for numbered/roman headings
        if re.match(r'^(\d+\.|[IVXLC]+\.|[A-Z]\.)', text):
            return True
        # All uppercase or mostly capitalized
        if text.isupper() or sum(1 for w in words if w.istitle()) >= len(words) * 0.7:
            return True
        # Ends with colon
        if text.endswith(":"):
            return True
        # Bold-like (heuristic: no punctuation, short, capitalized)
        if text == text.title() and not any(c in text for c in '.!?'):
            return True
        return False

    def _extract_sections_from_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Improved: Extracts text from PDF and groups paragraphs into sections by detecting headings.
        Uses improved heading detection and sentence tokenization for robustness.
        """
        sections = []
        doc = fitz.open(file_path)
        section_counter = 0
        for page_index in range(len(doc)):
            page = doc.load_page(page_index)
            blocks = page.get_text("blocks", sort=True)
            current_section = {"title": f"page_{page_index+1}_untitled", "page_number": page_index+1, "paragraphs": []}
            for block in blocks:
                if block[6] != 0:
                    continue
                text = block[4].replace('\r', ' ').replace('\n', ' ').strip()
                if not text:
                    continue
                # Split block into sentences for better granularity
                sentences = nltk.sent_tokenize(text)
                for sent in sentences:
                    if self._is_heading(sent):
                        if current_section["paragraphs"]:
                            sections.append(current_section)
                        section_counter += 1
                        current_section = {"title": sent, "page_number": page_index+1, "paragraphs": [], "section_index": section_counter}
                    else:
                        current_section["paragraphs"].append(sent)
            if current_section["paragraphs"]:
                if "section_index" not in current_section:
                    section_counter += 1
                    current_section["section_index"] = section_counter
                sections.append(current_section)
        doc.close()
        return sections

    def _extract_text_from_image(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Extracts text from an image using OCR.
        Treats the whole image as one section.
        """
        sections = []
        try:
            img = Image.open(file_path)
            text_from_ocr = pytesseract.image_to_string(img)
            
            if text_from_ocr.strip():
                # Split text into paragraphs
                paragraphs = [p.strip() for p in text_from_ocr.split('\n\n') if p.strip()]
                if not paragraphs:  # If no double newlines, try single newlines
                    paragraphs = [p.strip() for p in text_from_ocr.split('\n') if p.strip()]
                
                if paragraphs:
                    sections.append({
                        "title": "ocr_image_section",
                        "page_number": 1,
                        "paragraphs": paragraphs,
                        "section_index": 1
                    })
            print(f"Extracted text using OCR from image: {os.path.basename(file_path)}")
        except pytesseract.TesseractNotFoundError:
            raise DocumentProcessingError("Tesseract is not installed or not in your PATH.")
        except Exception as e:
            raise DocumentProcessingError(f"Error extracting text from image {os.path.basename(file_path)} using OCR: {e}")
        return sections

    def _perform_semantic_chunking(self, text: str) -> List[str]:
        """
        Perform semantic chunking on text using embeddings, fallback if needed.
        Here we use RecursiveCharacterTextSplitter for INSTRUCTOR-compatible chunking.
        """
        try:
            return self.semantic_splitter.split_text(text)
        except Exception as e:
            raise DocumentProcessingError(f"Chunking failed: {e}")

    def process_document(self, file_path: str, source_doc_id: str, collection_name: Optional[str] = None) -> bool:
        """Process a document using fast rule-based chunking."""
        try:
            # Extract text based on file type
            file_extension = Path(file_path).suffix.lower()
            if file_extension in ['.pdf']:
                sections = self._extract_sections_from_pdf(file_path)
            elif file_extension in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif']:
                sections = self._extract_text_from_image(file_path)
            else:
                print(f"Unsupported file type: {file_extension}")
                return False

            if not sections:
                print(f"No text extracted from {file_path}")
                return False

            # Process the text
            all_chunks, all_metadatas, all_ids = [], [], []
            global_chunk_counter = 1
            for sec in sections:
                section_text = "\n\n".join(sec["paragraphs"])
                chunks = self._perform_semantic_chunking(section_text)
                for idx, chunk in enumerate(chunks):
                    sec_idx = sec.get("section_index", sec["page_number"])
                    chunk_id = f"{source_doc_id}_sec{sec_idx}_chunk{global_chunk_counter}"
                    metadata = {
                        "source_doc_id": source_doc_id,
                        "file_name": Path(file_path).name,
                        "section_title": sec["title"],
                        "page_number": sec["page_number"],
                        "section_index": sec_idx,
                        "chunk_index": global_chunk_counter,
                        "paragraph_number_in_page": idx + 1
                    }
                    all_chunks.append(chunk)
                    all_metadatas.append(metadata)
                    all_ids.append(chunk_id)
                    global_chunk_counter += 1

            # Add documents to vector store
            success = self.vector_store_service.add_documents(
                chunks=all_chunks,
                metadatas=all_metadatas,
                doc_ids=all_ids,
                collection_name=collection_name
            )

            return success
        except Exception as e:
            raise DocumentProcessingError(f"Error processing document {file_path}: {e}")

    def extract_pdf_metadata(self, file_path: str) -> dict:
        """
        Extracts metadata (title, author, creation/publication date) from a PDF using fitz (PyMuPDF).
        For arXiv PDFs, tries to extract from the first page or arXiv header if available.
        """
        doc = fitz.open(file_path)
        meta = doc.metadata or {}
        # Try to extract from PDF metadata
        title = meta.get('title') or ''
        author = meta.get('author') or ''
        creation_date = meta.get('creationDate') or meta.get('modDate') or ''
        # Try to extract from first page text (arXiv style)
        first_page = doc[0].get_text("text") if len(doc) > 0 else ''
        # Use regex to find arXiv-style author and date
        arxiv_author = ''
        arxiv_date = ''
        arxiv_title = ''
        # Title: often first non-empty line
        for line in first_page.splitlines():
            if not arxiv_title and line.strip() and len(line.strip()) > 5 and len(line.strip().split()) > 2:
                arxiv_title = line.strip()
            # Author: look for 'by' or typical author line
            if not arxiv_author:
                m = re.search(r'by\s+([A-Za-z,\-\s]+)', line, re.IGNORECASE)
                if m:
                    arxiv_author = m.group(1).strip()
            # Date: look for year or arXiv submission line
            if not arxiv_date:
                m = re.search(r'(\d{4})', line)
                if m:
                    arxiv_date = m.group(1)
            if arxiv_title and arxiv_author and arxiv_date:
                break
        doc.close()
        return {
            'title': arxiv_title or title,
            'author': arxiv_author or author,
            'publication_date': arxiv_date or creation_date
        }

    def extract_headings_from_pdf(self, file_path: str) -> List[str]:
        """
        Extracts all headings from a PDF using heuristics and regex (arXiv-style and general headings).
        Returns a list of heading strings.
        """
        doc = fitz.open(file_path)
        headings = []
        heading_pattern = re.compile(r'^(\d+\.?)+\s+.+|^[A-Z][A-Z\s\-:]{4,}$')  # e.g. 1. Introduction, II. METHODS, etc.
        for page in doc:
            blocks = page.get_text("blocks", sort=True)
            for block in blocks:
                text = block[4].replace('\r', ' ').replace('\n', ' ').strip()
                if not text:
                    continue
                # Split into lines for finer granularity
                for line in text.split('. '):
                    line = line.strip()
                    if not line:
                        continue
                    if self._is_heading(line) or heading_pattern.match(line):
                        headings.append(line)
        doc.close()
        return headings
