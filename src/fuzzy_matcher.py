from rapidfuzz import fuzz
from src.database import Database
import unicodedata
import re

# Filijala iz zaglavlja PDF-a se uzima u obzir samo kad nema (skoro) tačnog
# poklapanja: ako je najbolji WRatio ispod BRANCH_THRESHOLD, kandidati iz iste
# filijale dobijaju BRANCH_BONUS bodova pri rangiranju. f_wratio ostaje
# nepromijenjen (ML featurei se ne mijenjaju).
BRANCH_THRESHOLD = 97
BRANCH_BONUS = 5


def normalize_name(text: str) -> str:
    """
    Normalize a name for comparison:
    1. Remove diacritics (ć→c, š→s, nj→n, etc.)
    2. Remove prefixes (UD, D.O.O., etc.)
    """
    if not text:
        return ""

    # 1. Convert to uppercase for consistency
    text = text.upper()

    # 2. Remove diacritics (Ć→C, Č→C, Š→S, Đ→D, Ž→Z)
    # using NFKD normalization
    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ASCII', 'ignore').decode('utf-8')

    # 4. Remove special characters and extra spaces
    text = re.sub(r'[^\w\s]', ' ', text)  # replace punctuation with spaces
    text = re.sub(r'\s+', ' ', text)  # collapse multiple spaces into one
    text = text.strip()

    # 5. Remove common prefixes
    prefixes_to_remove = ["UD", "D.O.O.", "DOO", "AD", "UDOVA"]
    for prefix in prefixes_to_remove:
        if text.startswith(prefix + " "):
            text = text[len(prefix) + 1:]

    return text


def get_candidates(pdf_name: str, filijala: str, top_n: int = 3):
    db = Database()
    with db.connection() as conn:
        # Filtriranje po filijali je namjerno iskljuceno:
        # "SELECT * FROM customers WHERE UPPER(filijala) = UPPER(?)", (filijala,)
        rows = conn.execute("SELECT * FROM customers").fetchall()
        customers = [dict(r) for r in rows]

    # Normalize the PDF name once
    pdf_name_norm = normalize_name(pdf_name)

    scored = []
    for c in customers:
        cand_name = c["name"] or ""
        cand_name_norm = normalize_name(cand_name)

        # Compute similarity metrics on normalized names
        r = fuzz.ratio(pdf_name_norm, cand_name_norm)
        pr = fuzz.partial_ratio(pdf_name_norm, cand_name_norm)
        tsr = fuzz.token_sort_ratio(pdf_name_norm, cand_name_norm)
        tset = fuzz.token_set_ratio(pdf_name_norm, cand_name_norm)
        wratio = fuzz.WRatio(pdf_name_norm, cand_name_norm)

        scored.append({
            "customer_id": c["customer_id"],
            "name": cand_name,  # original name from database
            "name_norm": cand_name_norm,  # normalized version (for debugging)
            "filijala": c["filijala"],
            "f_ratio": r,
            "f_partial": pr,
            "f_token_sort": tsr,
            "f_token_set": tset,
            "f_wratio": wratio,
        })

    # Filijala: bonus za kandidate iz iste filijale, samo ako nema skoro tačnog poklapanja
    best = max((s["f_wratio"] for s in scored), default=0)
    use_branch = bool(filijala) and best < BRANCH_THRESHOLD
    branch_upper = (filijala or "").strip().upper()
    for s in scored:
        same = use_branch and (s["filijala"] or "").strip().upper() == branch_upper
        s["branch_bonus"] = BRANCH_BONUS if same else 0

    # Rang: WRatio + bonus filijale; tie-break token_set_ratio, pa ratio
    scored.sort(
        key=lambda x: (x["f_wratio"] + x["branch_bonus"], x["f_token_set"], x["f_ratio"]),
        reverse=True,
    )
    return scored[:top_n]

