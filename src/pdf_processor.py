import pdfplumber
import pandas as pd
from src.config import PDF_COORDS
from src.logger import get_logger

logger = get_logger(__name__)


class PDFProcessor:
    def __init__(self):
        # Mapping for Cyrillic to Latin transliteration
        self.CYR_TO_LAT = {
            'А': 'A', 'Б': 'B', 'В': 'V', 'Г': 'G', 'Д': 'D',
            'Ђ': 'Đ', 'Е': 'E', 'Ж': 'Ž', 'З': 'Z', 'И': 'I',
            'Ј': 'J', 'К': 'K', 'Л': 'L', 'Љ': 'Lj', 'М': 'M',
            'Н': 'N', 'Њ': 'Nj', 'О': 'O', 'П': 'P', 'Р': 'R',
            'С': 'S', 'Т': 'T', 'Ћ': 'Ć', 'У': 'U', 'Ф': 'F',
            'Х': 'H', 'Ц': 'C', 'Ч': 'Č', 'Џ': 'Dž', 'Ш': 'Š',
            'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd',
            'ђ': 'đ', 'е': 'e', 'ж': 'ž', 'з': 'z', 'и': 'i',
            'ј': 'j', 'к': 'k', 'л': 'l', 'љ': 'lj', 'м': 'm',
            'н': 'n', 'њ': 'nj', 'о': 'o', 'п': 'p', 'р': 'r',
            'с': 's', 'т': 't', 'ћ': 'ć', 'у': 'u', 'ф': 'f',
            'х': 'h', 'ц': 'c', 'ч': 'č', 'џ': 'dž', 'ш': 'š',
        }

        self.BRANCH_NAMES = [
            "TREBINJE",
            "DOBOJ",
            "BIJELJINA",
            "ISTOCNO SARAJEVO",
            "PRIJEDOR",
            "BANJA LUKA",
        ]

        # Coordinates from config
        self.NAME_MIN_X = PDF_COORDS["NAME_MIN_X"]
        self.NAME_MAX_X = PDF_COORDS["NAME_MAX_X"]
        self.RATA_MIN_X = PDF_COORDS["RATA_MIN_X"]
        self.RATA_MAX_X = PDF_COORDS["RATA_MAX_X"]
        self.ROW_TOL = PDF_COORDS["ROW_TOL"]

        self.BRANCH_MIN_X = PDF_COORDS["BRANCH_MIN_X"]
        self.BRANCH_MAX_X = PDF_COORDS["BRANCH_MAX_X"]
        self.BRANCH_MIN_Y = PDF_COORDS["BRANCH_MIN_Y"]
        self.BRANCH_MAX_Y = PDF_COORDS["BRANCH_MAX_Y"]

    def transliterate(self, text):
        """Converts Cyrillic to Latin"""
        if not text:
            return ""
        return ''.join(self.CYR_TO_LAT.get(c, c) for c in text)

    def flip_name(self, ime_prezime: str) -> str:
        """
        'Pera Peric' -> 'Peric Pera'
        """
        parts = ime_prezime.split()
        if len(parts) < 2:
            return ime_prezime
        first = " ".join(parts[:-1])
        last = parts[-1]
        return f"{last} {first}"

    def extract_payments(self, pdf_file_path: str, start_page: int = 1) -> pd.DataFrame:
        """
        Extracts payments from PDF using coordinate filtering.

        start_page: 1-based index of the first page to include (1 = first page).
        """
        logger.info(f"Extracting payments from: {pdf_file_path} (start_page={start_page})")
        rows = self._extract_all_pages(pdf_file_path, start_page=start_page)

        # Create DataFrame
        df = pd.DataFrame(rows)

        # Remove header rows
        df = self._remove_header_rows(df)

        logger.info(f"Extracted {len(df)} payments")
        return df

    def extract_payments_from_page(self, pdf_file_path: str, start_page: int = 1) -> pd.DataFrame:
        """
        Ekstraktuje redove plaćanja počevši od zadate stranice PDF-a.

        start_page: 1 = prva stranica; stranice ispod ove se preskaču.
        """
        return self.extract_payments(pdf_file_path, start_page=start_page)

    def _extract_all_pages(self, pdf_file_path: str, start_page: int = 1) -> list:
        """Helper to iterate over pages and collect rows from start_page onward (1-based)."""
        rows = []
        sp = max(1, int(start_page))
        try:
            with pdfplumber.open(pdf_file_path) as pdf:
                total = len(pdf.pages)
                if sp > total:
                    logger.warning(
                        "start_page=%s is greater than PDF page count (%s); no rows extracted.",
                        sp,
                        total,
                    )
                for page_num, page in enumerate(pdf.pages, 1):
                    if page_num < sp:
                        continue
                    logger.debug(f"Processing page {page_num}...")
                    page_rows = self._extract_page_rows(page)
                    rows.extend(page_rows)
        except Exception as e:
            logger.error(f"Error opening PDF {pdf_file_path}: {e}")
            raise

        return rows

    def _extract_page_rows(self, page) -> list:
        """Helper to extract data from a single page using coordinates"""
        page_rows = []
        words = page.extract_words()

        # Group words into lines by `top` coordinate
        lines = {}
        for w in words:
            top = round(w["top"] / self.ROW_TOL)
            lines.setdefault(top, []).append(w)

        for _, line_words in lines.items():
            name_parts = []
            rata_parts = []

            for w in line_words:
                x = w["x0"]
                text = w["text"]

                if self.NAME_MIN_X <= x <= self.NAME_MAX_X:
                    name_parts.append(text)
                elif self.RATA_MIN_X <= x <= self.RATA_MAX_X:
                    rata_parts.append(text)

            if name_parts and rata_parts:
                name = " ".join(name_parts)
                # Clean amount from spaces and replace comma with dot
                rata = "".join(rata_parts).replace(",", ".")

                # Format name: transliterate and flip
                formatted_name = self._format_name(name)

                page_rows.append({
                    "ime_prezime": formatted_name,
                    "rata": rata
                })
        return page_rows

    def _format_name(self, name: str) -> str:
        """Transliterates and flips the name"""
        latin_name = self.transliterate(name)
        return self.flip_name(latin_name)

    def _extract_branch(self, pdf_file_path: str) -> str:
        """
        Extracts place from pdf header using coordinates
        """
        try:
            with pdfplumber.open(pdf_file_path) as pdf:
                page = pdf.pages[0]
                words = page.extract_words()

                # takes words from header
                branch_words = []
                for w in words:
                    x = w["x0"]
                    y = w["top"]
                    if (self.BRANCH_MIN_X <= x <= self.BRANCH_MAX_X and
                            self.BRANCH_MIN_Y <= y <= self.BRANCH_MAX_Y):
                        branch_words.append(w["text"])

                if not branch_words:
                    return ""

                raw_branch = " ".join(branch_words)
                # transliterate
                branch_lat = self.transliterate(raw_branch).upper()

                # mapping to standard places in db
                for name in ["BANJA LUKA", "TREBINJE", "DOBOJ", "BIJELJINA", "ISTOCNO SARAJEVO", "PRIJEDOR"]:
                    if name in branch_lat:
                        return name

                return branch_lat  # fallback: return raw text, then do the mapping
        except Exception as e:
            logger.error(f"Error extracting branch: {e}")
            return ""

    def _remove_header_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filters out unwanted rows like headers"""
        if df.empty:
            return df

        mask = (
                df["ime_prezime"].str.contains("SANO PLUS", case=False, na=False) |
                df["ime_prezime"].str.contains("Naziv / ime", case=False, na=False) |
                df["ime_prezime"].str.contains("ime Naziv /", case=False, na=False) |
                df["ime_prezime"].str.contains("PLUS SANO", case=False, na=False)
        )
        return df[~mask].reset_index(drop=True)

    def save_to_csv(self, df: pd.DataFrame, output_path: str = "extracted_payments.csv"):
        """Saves DataFrame to CSV with logging"""
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df)} rows to {output_path}")

