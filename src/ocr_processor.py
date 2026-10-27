import os
import json
from pathlib import Path
import pandas as pd
from src.pdf_processor import PDFProcessor
from src.config import VISION_AGENT_API_KEY
from src.logger import get_logger

logger = get_logger(__name__)


class OCRProcessor:
    def __init__(self):
        try:
            from landingai_ade import LandingAIADE
        except ImportError:
            logger.error("LandingAI ADE library not found. Please install it with 'pip install landingai-ade'.")
            raise ImportError("LandingAI ADE library not found. Please install it with 'pip install landingai-ade'.")

        self.api_key = VISION_AGENT_API_KEY
        if not self.api_key:
            logger.error("VISION_AGENT_API_KEY not set in configuration.")
            raise ValueError("VISION_AGENT_API_KEY not set in configuration.")

        self.client = LandingAIADE()
        self._pdf_processor = PDFProcessor()

    def extract_payments(self, pdf_path: str) -> pd.DataFrame:
        """Extracts payments from scanned PDF using LandingAI OCR"""
        logger.info(f"Extracting payments (OCR) from: {pdf_path}")
        document_path = Path(pdf_path)

        try:
            parse_result = self.client.parse(
                document=document_path,
                model="dpt-2-latest"
            )

            schema_dict = {
                "type": "object",
                "properties": {
                    "osobe": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "ime_i_prezime": {
                                    "type": "string",
                                    "description": "Puno ime i prezime osobe iz kolone 'Naziv i ime' ili 'Ime i prezime'"
                                },
                                "rata": {
                                    "type": "number",
                                    "description": "Iznos SAMO iz kolone koja se zove tacno 'Rata' ili 'Iznos rate'. NE uzimaj 'Ukupno', 'Kamata', 'Ostatak duga' ili bilo koju drugu kolonu."
                                }
                            }
                        }
                    }
                }
            }

            extraction_result = self.client.extract(
                schema=json.dumps(schema_dict, ensure_ascii=False),
                markdown=parse_result.markdown,
                model="extract-latest"
            )

            osobe = extraction_result.extraction.get('osobe', [])
            rows = []
            for osoba in osobe:
                ime = self._obradi_ime(osoba.get('ime_i_prezime', ''))
                # LandingAI returns number, format it to string with dot
                rata_val = osoba.get('rata', '')
                rata_str = str(rata_val).replace(",", ".")

                rows.append({
                    'ime_prezime': ime,
                    'rata': rata_str
                })

            df = pd.DataFrame(rows)
            logger.info(f"Extracted {len(df)} payments using OCR")
            return df

        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            raise

    def _obradi_ime(self, ime: str) -> str:
        """Transliterates and flips the name using PDFProcessor logic"""
        latin_name = self._pdf_processor.transliterate(ime)
        return self._pdf_processor.flip_name(latin_name)