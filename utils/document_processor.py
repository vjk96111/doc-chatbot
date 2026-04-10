"""
document_processor.py
Extracts text chunks and images from PDF and DOCX files.
"""

import io
import re
import base64
import zipfile
from typing import Dict, List


# ─────────────────────────────────── helpers ─────────────────────────────────

def _chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start += chunk_size - overlap
    return chunks


def _is_useful_image(img_bytes: bytes, ext: str) -> bool:
    """Filter out tiny icons / decorative images."""
    return len(img_bytes) >= 5_000 and ext.lower() in {
        "png", "jpeg", "jpg", "bmp", "tiff", "gif"
    }


# ─────────────────────────────────── PDF ─────────────────────────────────────

def extract_from_pdf(file_bytes: bytes, doc_name: str = "") -> Dict:
    """
    Extract text chunks and images from a PDF.

    Returns:
        {
          "chunks": [{"text", "page", "source", "doc_name"}, ...],
          "images": {page_num: [{"data": b64_str, "ext": str, "page": int, "doc_name": str}, ...]}
        }
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise ImportError("PyMuPDF is required. Run: pip install PyMuPDF") from exc

    try:
        pdf = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:
        raise ValueError(f"Could not open PDF '{doc_name}': {exc}") from exc

    all_chunks: List[Dict] = []
    images_by_page: Dict[int, List[Dict]] = {}

    for page_idx in range(len(pdf)):
        page = pdf[page_idx]
        page_num = page_idx + 1

        # ── text ──────────────────────────────────────────────────────────────
        raw_text = page.get_text("text").strip()
        if raw_text:
            for sub in _chunk_text(raw_text):
                all_chunks.append(
                    {
                        "text": sub,
                        "page": page_num,
                        "source": f"Page {page_num}",
                        "doc_name": doc_name,
                    }
                )

        # ── images ────────────────────────────────────────────────────────────
        page_imgs: List[Dict] = []
        for img_info in page.get_images(full=True):
            xref = img_info[0]
            try:
                base_img = pdf.extract_image(xref)
                img_bytes = base_img["image"]
                ext = base_img.get("ext", "png").lower().replace("jpg", "jpeg")
                if _is_useful_image(img_bytes, ext):
                    page_imgs.append(
                        {
                            "data": base64.b64encode(img_bytes).decode("utf-8"),
                            "ext": ext,
                            "page": page_num,
                            "doc_name": doc_name,
                        }
                    )
            except Exception:
                continue

        if page_imgs:
            images_by_page[page_num] = page_imgs

    pdf.close()
    return {"chunks": all_chunks, "images": images_by_page}


# ─────────────────────────────────── DOCX ────────────────────────────────────

def _docx_rel_map(z: zipfile.ZipFile) -> Dict[str, str]:
    """Map relationship IDs to media paths inside the zip."""
    rel_file = "word/_rels/document.xml.rels"
    if rel_file not in z.namelist():
        return {}
    xml_text = z.read(rel_file).decode("utf-8", errors="replace")
    rel_map: Dict[str, str] = {}
    for m in re.finditer(r'Id="([^"]+)"[^>]*Target="([^"]+)"', xml_text):
        rid, target = m.group(1), m.group(2)
        if "media/" in target:
            # Normalise path: ../media/img.png → word/media/img.png
            clean = re.sub(r"^\.\./", "word/", target)
            rel_map[rid] = clean
    return rel_map


def _docx_image_order(z: zipfile.ZipFile) -> List[str]:
    """Return rIds of images in document order (order they appear in XML)."""
    doc_file = "word/document.xml"
    if doc_file not in z.namelist():
        return []
    xml_text = z.read(doc_file).decode("utf-8", errors="replace")
    return re.findall(r'r:embed="(rId\d+)"', xml_text)


def extract_from_docx(file_bytes: bytes, doc_name: str = "") -> Dict:
    """
    Extract text chunks and images from a DOCX file.

    Returns same structure as extract_from_pdf with "sections" instead of pages.
    """
    try:
        from docx import Document
    except ImportError as exc:
        raise ImportError("python-docx is required. Run: pip install python-docx") from exc

    doc = Document(io.BytesIO(file_bytes))

    # ── Build sections from headings ──────────────────────────────────────────
    sections: List[Dict] = []       # [{"title", "text", "section_num"}]
    current_title = "Introduction"
    current_lines: List[str] = []
    section_num = 1

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        is_heading = para.style.name.startswith("Heading")
        if is_heading and current_lines:
            sections.append(
                {
                    "title": current_title,
                    "text": "\n".join(current_lines),
                    "section_num": section_num,
                }
            )
            section_num += 1
            current_title = text
            current_lines = []
        else:
            current_lines.append(text)

    if current_lines:
        sections.append(
            {
                "title": current_title,
                "text": "\n".join(current_lines),
                "section_num": section_num,
            }
        )

    # Fallback: no sections found
    if not sections:
        full = "\n".join(p.text.strip() for p in doc.paragraphs if p.text.strip())
        sections = [{"title": "Document", "text": full, "section_num": 1}]

    total_sections = len(sections)

    # ── Build text chunks ─────────────────────────────────────────────────────
    all_chunks: List[Dict] = []
    for sec in sections:
        for sub in _chunk_text(sec["text"]):
            all_chunks.append(
                {
                    "text": sub,
                    "page": sec["section_num"],
                    "source": f"Section: {sec['title']}",
                    "doc_name": doc_name,
                }
            )

    # ── Extract images from the zip ───────────────────────────────────────────
    images_by_section: Dict[int, List[Dict]] = {}

    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
            rel_map = _docx_rel_map(z)
            ordered_rids = _docx_image_order(z)

            # Filter to image rIds only (in document order)
            img_rids_in_order = [
                rid for rid in ordered_rids if rid in rel_map
            ]

            total_imgs = len(img_rids_in_order)
            for idx, rid in enumerate(img_rids_in_order):
                media_path = rel_map[rid]
                if media_path not in z.namelist():
                    continue
                try:
                    img_bytes_raw = z.read(media_path)
                    ext = media_path.rsplit(".", 1)[-1].lower().replace("jpg", "jpeg")
                    if not _is_useful_image(img_bytes_raw, ext):
                        continue
                    b64 = base64.b64encode(img_bytes_raw).decode("utf-8")
                    # Distribute images proportionally across sections
                    sec_num = min(
                        int(idx * total_sections / max(total_imgs, 1)) + 1,
                        total_sections,
                    )
                    images_by_section.setdefault(sec_num, []).append(
                        {
                            "data": b64,
                            "ext": ext,
                            "page": sec_num,
                            "doc_name": doc_name,
                        }
                    )
                except Exception:
                    continue
    except Exception:
        pass  # Image extraction is best-effort

    return {"chunks": all_chunks, "images": images_by_section}
