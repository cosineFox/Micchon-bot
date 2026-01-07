"""
OCR utilities for text extraction from images.

Uses SmolVLM as the primary method (via LlamaClient.describe_image).
Falls back to dedicated OCR (EasyOCR/Tesseract) for text-heavy images.
"""
import asyncio
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy imports for heavy OCR libraries
_easyocr_reader = None
_tesseract_available = None


async def extract_text_ocr(
    image_path: Path,
    method: str = "easyocr",
    languages: list[str] = ["en"]
) -> str:
    """
    Extract text from image using dedicated OCR

    Args:
        image_path: Path to image file
        method: "easyocr" (default, GPU) or "tesseract" (CPU)
        languages: Language codes (e.g., ["en", "ja"])

    Returns:
        Extracted text
    """
    if method == "easyocr":
        return await _ocr_easyocr(image_path, languages)
    elif method == "tesseract":
        return await _ocr_tesseract(image_path, languages)
    else:
        raise ValueError(f"Unknown OCR method: {method}")


async def _ocr_easyocr(image_path: Path, languages: list[str]) -> str:
    """Extract text using EasyOCR (GPU-accelerated)"""
    global _easyocr_reader

    loop = asyncio.get_event_loop()

    def _extract():
        global _easyocr_reader

        try:
            import easyocr

            # Lazy init reader (downloads models on first use)
            if _easyocr_reader is None:
                logger.info(f"Initializing EasyOCR with languages: {languages}")
                _easyocr_reader = easyocr.Reader(languages, gpu=True)

            # Run OCR
            results = _easyocr_reader.readtext(str(image_path))

            # Combine all detected text
            texts = [result[1] for result in results]
            return "\n".join(texts)

        except ImportError:
            logger.warning("EasyOCR not installed. Install with: pip install easyocr")
            return ""
        except Exception as e:
            logger.error(f"EasyOCR failed: {e}")
            return ""

    return await loop.run_in_executor(None, _extract)


async def _ocr_tesseract(image_path: Path, languages: list[str]) -> str:
    """Extract text using Tesseract (CPU)"""
    loop = asyncio.get_event_loop()

    def _extract():
        try:
            import pytesseract
            from PIL import Image

            # Check if tesseract is available
            global _tesseract_available
            if _tesseract_available is None:
                try:
                    pytesseract.get_tesseract_version()
                    _tesseract_available = True
                except pytesseract.TesseractNotFoundError:
                    logger.warning(
                        "Tesseract not found. Install with: "
                        "brew install tesseract (macOS) or apt install tesseract-ocr (Linux)"
                    )
                    _tesseract_available = False

            if not _tesseract_available:
                return ""

            # Run OCR
            img = Image.open(image_path)
            lang = "+".join(languages)  # Tesseract format: "eng+jpn"

            # Map common codes to tesseract codes
            lang_map = {"en": "eng", "ja": "jpn", "zh": "chi_sim", "ko": "kor"}
            tesseract_langs = [lang_map.get(l, l) for l in languages]
            lang = "+".join(tesseract_langs)

            text = pytesseract.image_to_string(img, lang=lang)
            return text.strip()

        except ImportError:
            logger.warning("pytesseract not installed. Install with: pip install pytesseract")
            return ""
        except Exception as e:
            logger.error(f"Tesseract OCR failed: {e}")
            return ""

    return await loop.run_in_executor(None, _extract)


async def is_text_heavy_image(image_path: Path, threshold: float = 0.3) -> bool:
    """
    Heuristic to detect if image is text-heavy (document, screenshot with text)

    Args:
        image_path: Path to image
        threshold: Text area ratio threshold

    Returns:
        True if image appears to be text-heavy
    """
    # Simple heuristic based on file extension and aspect ratio
    try:
        from PIL import Image
        img = Image.open(image_path)

        # Document-like aspect ratios (portrait A4, letter, etc.)
        width, height = img.size
        aspect = height / width if width > 0 else 0

        # Portrait documents typically have aspect > 1.2
        if aspect > 1.2:
            return True

        # Very wide images (screenshots) might have text
        if aspect < 0.6 and width > 1000:
            return True

        return False

    except Exception:
        return False


def cleanup_ocr():
    """Free OCR resources"""
    global _easyocr_reader
    if _easyocr_reader is not None:
        del _easyocr_reader
        _easyocr_reader = None
        logger.info("EasyOCR reader unloaded")
