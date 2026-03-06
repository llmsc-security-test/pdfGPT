"""PDF document processing: loading, parsing, and chunking."""

import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import fitz


@dataclass
class DocumentChunk:
    """A chunk of text extracted from a document with metadata."""
    text: str
    page_number: int
    chunk_index: int
    source: str = ""

    @property
    def citation(self):
        return f"[Page {self.page_number}]"

    def __str__(self):
        return f"{self.citation} {self.text}"


class PDFProcessor:
    """Handles PDF loading, text extraction, and chunking with overlap."""

    def __init__(self, chunk_size: int = 200, chunk_overlap: int = 50):
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    @staticmethod
    def download_pdf(url: str, output_path: str) -> str:
        urllib.request.urlretrieve(url, output_path)
        return output_path

    @staticmethod
    def _clean_text(text: str) -> str:
        text = text.replace('\n', ' ')
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def extract_pages(self, pdf_path: str,
                      start_page: int = 1,
                      end_page: Optional[int] = None) -> List[dict]:
        """Extract text from each page of a PDF file."""
        doc = fitz.open(pdf_path)
        total_pages = doc.page_count
        if end_page is None:
            end_page = total_pages

        pages = []
        for i in range(start_page - 1, min(end_page, total_pages)):
            raw_text = doc.load_page(i).get_text("text")
            cleaned = self._clean_text(raw_text)
            if cleaned:
                pages.append({
                    "page_number": i + 1,
                    "text": cleaned,
                })
        doc.close()
        return pages

    def chunk_document(self, pdf_path: str,
                       start_page: int = 1,
                       end_page: Optional[int] = None) -> List[DocumentChunk]:
        """Extract and chunk a PDF into DocumentChunk objects with overlap."""
        pages = self.extract_pages(pdf_path, start_page, end_page)
        chunks = []
        chunk_index = 0
        source = Path(pdf_path).name

        for page in pages:
            words = page["text"].split()
            page_num = page["page_number"]
            step = max(1, self._chunk_size - self._chunk_overlap)

            for i in range(0, len(words), step):
                chunk_words = words[i:i + self._chunk_size]
                if len(chunk_words) < 20 and i > 0:
                    continue
                text = " ".join(chunk_words)
                chunks.append(DocumentChunk(
                    text=text,
                    page_number=page_num,
                    chunk_index=chunk_index,
                    source=source,
                ))
                chunk_index += 1

        return chunks

    def get_page_text(self, pdf_path: str, page_number: int) -> str:
        """Get the full text of a specific page."""
        doc = fitz.open(pdf_path)
        if page_number < 1 or page_number > doc.page_count:
            doc.close()
            return f"Page {page_number} does not exist. Document has {doc.page_count} pages."
        text = doc.load_page(page_number - 1).get_text("text")
        doc.close()
        return self._clean_text(text)

    def get_page_count(self, pdf_path: str) -> int:
        doc = fitz.open(pdf_path)
        count = doc.page_count
        doc.close()
        return count
