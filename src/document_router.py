import pandas as pd
from src.pdf_processor import PDFProcessor
from src.logger import get_logger

logger = get_logger(__name__)


class DocumentRouter:
    def __init__(self, force: str | None = None):
        """
        DocumentRouter selects the appropriate extraction engine.
        :param force: "pdf", "ocr", or None for auto-detection
        """
        self.force = force
        self.pdf_processor = PDFProcessor()
        self._ocr_processor = None  # Lazy initialize

    @property
    def ocr_processor(self):
        if self._ocr_processor is None:
            from src.ocr_processor import OCRProcessor
            self._ocr_processor = OCRProcessor()
        return self._ocr_processor

    def extract(self, pdf_path: str) -> tuple[pd.DataFrame, str]:
        """
        Extracts payments from the PDF.
        Returns: (DataFrame, engine_name)
        """
        if self.force == "ocr":
            logger.info("Forcing OCR extraction.")
            return self.ocr_processor.extract_payments(pdf_path), "LandingAI OCR"

        if self.force == "pdf":
            logger.info("Forcing PDF coordinate extraction.")
            return self.pdf_processor.extract_payments(pdf_path), "Coordinates"

        # Auto-detection
        logger.info(f"Auto-detecting extraction engine for: {pdf_path}")
        try:
            df = self.pdf_processor.extract_payments(pdf_path)
            if not df.empty:
                logger.info("PDF coordinate extraction successful.")
                return df, "Coordinates"
            else:
                logger.info("PDF coordinate extraction returned empty. Falling back to OCR.")
        except Exception as e:
            logger.warning(f"PDF coordinate extraction failed: {e}. Falling back to OCR.")

        # Fallback to OCR
        return self.ocr_processor.extract_payments(pdf_path), "LandingAI OCR"
