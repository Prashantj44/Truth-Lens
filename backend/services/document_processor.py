import re
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
import PyPDF2
from backend.config import CHUNK_SIZE, CHUNK_OVERLAP

def clean_text(text: str) -> str:
    """Sanitizes text by removing abnormal control chars and normalizing whitespace."""
    if not text:
        return ""
    # Normalize unicode spaces and hyphens
    text = text.replace("\u00a0", " ").replace("\u2013", "-").replace("\u2014", "-")
    # Replace multiple newlines or tabs with standard whitespace
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def detect_source_type(filename: str, source_hint: str = "") -> str:
    """Infers credibility tier from source name or filename."""
    combined = f"{filename} {source_hint}".lower()
    if any(k in combined for k in ["gov", "ministry", "official", "imf", "world bank", "who", "cdc", "un", "nasa", "rbi", "federal"]):
        return "Government / Official"
    elif any(k in combined for k in ["nature", "science", "lancet", "ieee", "arxiv", "journal", "academic", "university", "paper", "research", "pubmed"]):
        return "Peer-Reviewed Academic"
    elif any(k in combined for k in ["reuters", "ap", "bloomberg", "bbc", "hindu", "times", "wsj", "nyt", "guardian", "afp", "news"]):
        return "Established News"
    else:
        return "General Report"

def extract_text_from_file(file_path: Path) -> List[Dict[str, Any]]:
    """
    Extracts text and page numbers from supported file types.
    Returns a list of dicts: [{"page": int, "text": str}]
    """
    ext = file_path.suffix.lower()
    pages_data = []

    if ext == ".pdf":
        try:
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for idx, page in enumerate(reader.pages):
                    extracted = page.extract_text() or ""
                    cleaned = clean_text(extracted)
                    if cleaned:
                        pages_data.append({"page": idx + 1, "text": cleaned})
        except Exception as e:
            print(f"Error reading PDF {file_path}: {e}")
    else:
        # Default text, md, json
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = clean_text(f.read())
                if content:
                    pages_data.append({"page": 1, "text": content})
        except Exception as e:
            print(f"Error reading text file {file_path}: {e}")

    return pages_data

def recursive_chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> List[str]:
    """
    Recursively splits text on natural boundaries (paragraphs -> sentences -> words)
    ensuring each chunk does not cut statements mid-sentence where possible.
    """
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    # First split by double newline (paragraphs)
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current_chunk) + len(para) + 2 <= chunk_size:
            current_chunk = f"{current_chunk}\n\n{para}" if current_chunk else para
        else:
            if current_chunk:
                chunks.append(current_chunk)
                # Keep overlap from the tail of current chunk
                overlap_text = current_chunk[-chunk_overlap:] if len(current_chunk) > chunk_overlap else ""
                current_chunk = overlap_text

            if len(para) > chunk_size:
                # Split large paragraph by sentence
                sentences = re.split(r'(?<=[.?!])\s+', para)
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                        current_chunk = f"{current_chunk} {sentence}" if current_chunk else sentence
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                            overlap_text = current_chunk[-chunk_overlap:] if len(current_chunk) > chunk_overlap else ""
                            current_chunk = overlap_text
                        if len(sentence) > chunk_size:
                            # Hard split word by word
                            words = sentence.split(" ")
                            for w in words:
                                if len(current_chunk) + len(w) + 1 <= chunk_size:
                                    current_chunk = f"{current_chunk} {w}" if current_chunk else w
                                else:
                                    if current_chunk:
                                        chunks.append(current_chunk)
                                    current_chunk = w
                        else:
                            current_chunk = f"{current_chunk} {sentence}" if current_chunk else sentence
            else:
                current_chunk = f"{current_chunk}\n\n{para}" if current_chunk else para

    if current_chunk and current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks

def process_document(
    file_path: Path,
    source_name: Optional[str] = None,
    source_type: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Full document ingestion pipeline:
    Extracts, cleans, chunks, and attaches comprehensive metadata.
    """
    file_path = Path(file_path)
    filename = file_path.name
    source = source_name or filename
    source_cat = source_type or detect_source_type(filename, source)

    pages = extract_text_from_file(file_path)
    all_chunks = []
    chunk_counter = 0

    for p_info in pages:
        page_num = p_info["page"]
        text = p_info["text"]
        raw_chunks = recursive_chunk_text(text)

        for chunk_text in raw_chunks:
            if len(chunk_text.strip()) < 25:
                # Skip meaningless tiny snippets
                continue

            chunk_counter += 1
            chunk_id = f"{file_path.stem}_p{page_num}_c{chunk_counter}_{uuid.uuid4().hex[:6]}"
            all_chunks.append({
                "chunk_id": chunk_id,
                "document_name": filename,
                "source": source,
                "source_type": source_cat,
                "page_number": page_num,
                "text": chunk_text,
                "char_count": len(chunk_text)
            })

    return all_chunks
