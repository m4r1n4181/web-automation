"""
explore_customers.py

Dodaje filijalu svakom kupcu na osnovu mjesta (kolona municipality).

Kako radi:
  1. normalizuje mjesto (velika slova, bez dijakritika, bez tačaka/crtica)
  2. traži poznato mjesto kao cijelu riječ u tekstu (najduži naziv prvo),
     npr. "KOZARSKA DUBICA-KNEŽICA" -> DUBICA -> Prijedor
  3. ako ništa ne nađe, pokušava fuzzy za tipfelere (npr. "BQANJA LUKA")
  4. ako ni to ne uspije -> "Nepoznato" (ispisuje se na kraju, da se doda ručno)

Pokretanje (iz root foldera):
    python scripts/explore_customers.py
"""
import re
import unicodedata
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process

SCRIPTS_DIR = Path(__file__).resolve().parent
INPUT_CSV = SCRIPTS_DIR / "customers_all_pages.csv"
OUTPUT_CSV = SCRIPTS_DIR / "customers_with_filijala.csv"

FUZZY_THRESHOLD = 80
UNKNOWN = "Nepoznato"

# Mjesto (normalizovano) -> filijala.
# Mjesta van RS dodijeljena su geografski najbližoj filijali.
PLACES = {
    "Banja Luka": [
        "BANJA LUKA", "LAKTASI", "GRADISKA", "CELINAC", "MRKONJIC GRAD",
        "KOTOR VAROS", "SRBAC", "PRNJAVOR", "KNEZEVO", "SIPOVO", "JEZERO",
        "KRUPA NA VRBASU", "TOPOLA", "BRONZANI MAJDAN", "SLATINA",
        "ALEKSANDROVAC", "DOBRETICI", "78400",
        # centralna Bosna
        "JAJCE", "TRAVNIK", "TURBE", "VAKUF", "BUGOJNO", "VITEZ",
        "BUSOVACA",
    ],
    "Prijedor": [
        "PRIJEDOR", "NOVI GRAD", "BOSANSKI NOVI", "DUBICA", "KOSTAJNICA",
        "OMARSKA", "KOZARAC", "OSTRA LUKA", "RIBNIK", "DRVAR", "PETROVAC",
        "SANSKI MOST", "BOSANSKA KRUPA", "BOS KRUPA", "OTOKA", "KLJUC",
        "STARI MAJDAN", "GRAHOVO", "KULEN VAKUF", "DRINIC", "KOZ DOBICA",
        # Krajina (FBiH)
        "BIHAC", "CAZIN", "KLADUSA", "BUZIM", "VRNOGORAC",
    ],
    "Doboj": [
        "DOBOJ", "TESLIC", "DERVENTA", "STANARI", "PETROVO", "SOCKOVAC",
        "MODRICA", "VUKOSAVLJE", "SAMAC", "BROD", "PELAGICEVO", "ZABAR",
        "ZABARI", "OBUDOVAC", "SKUGRIC", "VUCKOVCI", "DRAGALOVCI",
        # FBiH
        "MAGLAJ", "TESANJ", "ZEPCE", "ZAVIDOVICI", "ODZAK", "DOMALJEVAC",
        "GRADACAC", "GRACANICA", "ZENICA",
    ],
    "Bijeljina": [
        "BIJELJINA", "BIJELJINSKA", "JANJA", "DVOROVI", "BRCKO",
        "BREZOVO POLJE", "LONCARI", "ZVORNIK", "OSMACI", "CAPARDE",
        "UGLJEVIK", "LOPARE", "PRIBOJ", "BRATUNAC", "SREBRENICA", "KRAVICA",
        "BRANJEVO", "MILICI", "VLASENICA", "SEKOVICI", "76300",
        # Tuzlanski kraj (FBiH)
        "TUZLA", "ZIVINICE", "LUKAVAC", "KALESIJA", "SREBRENIK", "TINJA",
        "BANOVICI", "KLADANJ", "STUPARI", "SAPNA", "TEOCAK", "CELIC",
        "KORAJ", "SIBOSNICA", "HUSINO", "MRAMOR", "DJURDJEVIK", "LJUBACE",
        "PRILUK",
        # Srbija
        "BEOGRAD",
    ],
    "Istocno Sarajevo": [
        "SARAJEVO", "ILIDZA", "PALE", "MOKRO", "SOKOLAC", "ROGATICA",
        "VISEGRAD", "FOCA", "RUDO", "CAJNICE", "GORAZDE", "TRNOVO",
        "KALINOVIK", "HAN PIJESAK", "VUCIJA LUKA", "STARI GRAD",
        # FBiH
        "BREZA", "KISELJAK", "VRELO BOSNE", "KAKANJ",
    ],
    "Trebinje": [
        "TREBINJE", "BILECA", "LJUBINJE", "NEVESINJE", "GACKO", "BERKOVICI",
        "KAZANCI",
        # Hercegovina (FBiH)
        "MOSTAR", "BLAGAJ", "CAPLJINA", "STOLAC", "JABLANICA", "PROZOR",
    ],
}

PLACE_TO_FILIJALA = {p: f for f, places in PLACES.items() for p in places}
# najduži naziv prvo: "KRUPA NA VRBASU" prije "KRUPA", "KULEN VAKUF" prije "VAKUF"
PLACES_SORTED = sorted(PLACE_TO_FILIJALA, key=len, reverse=True)


def norm_place(text) -> str:
    """'Kozarska Dubica-Knežica' -> 'KOZARSKA DUBICA KNEZICA'"""
    if not isinstance(text, str):
        return ""
    t = text.upper().replace("Đ", "DJ").replace("đ", "DJ")
    t = unicodedata.normalize("NFKD", t).encode("ASCII", "ignore").decode()
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def to_filijala(place_norm: str) -> tuple[str, str]:
    """Vraća (filijala, način) — način: 'tacno' | 'fuzzy:<mjesto>' | 'nepoznato'"""
    if not place_norm:
        return UNKNOWN, "nepoznato"

    # 1) poznato mjesto kao cijela riječ
    for place in PLACES_SORTED:
        if re.search(rf"\b{place}\b", place_norm):
            return PLACE_TO_FILIJALA[place], "tacno"

    # 2) tipfeleri
    match = process.extractOne(place_norm, PLACES_SORTED, scorer=fuzz.ratio)
    if match and match[1] >= FUZZY_THRESHOLD:
        return PLACE_TO_FILIJALA[match[0]], f"fuzzy:{match[0]}"

    return UNKNOWN, "nepoznato"


if __name__ == "__main__":
    df = pd.read_csv(INPUT_CSV, dtype=str).fillna("")
    df["place_norm"] = df["municipality"].apply(norm_place)
    result = df["place_norm"].apply(to_filijala)
    df["filijala"] = [r[0] for r in result]
    method = pd.Series([r[1] for r in result], index=df.index)

    df.drop(columns=[c for c in df.columns if c.startswith("col")]).to_csv(
        OUTPUT_CSV, index=False, encoding="utf-8"
    )

    print(f"Kupaca: {len(df)}")
    print("\nPo filijali:")
    print(df["filijala"].value_counts().to_string())

    fuzzy = df[method.str.startswith("fuzzy")]
    if not fuzzy.empty:
        print("\nIspravljeni tipfeleri (provjeri):")
        pairs = pd.DataFrame({"mjesto": fuzzy["municipality"], "kao": method[fuzzy.index]})
        print(pairs.value_counts().to_string())

    unknown = df[df["filijala"] == UNKNOWN]
    if not unknown.empty:
        print(f"\nNepoznato ({len(unknown)}) — dodaj u PLACES ako treba:")
        print(unknown["municipality"].replace("", "(prazno)").value_counts().to_string())

    print(f"\nSačuvano: {OUTPUT_CSV}")
