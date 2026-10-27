"""
feature_extractor.py

Prima pdf_name i jednog kandidata (dict koji dolazi iz fuzzy_matcher.get_candidates()).
Vraća dictionary sa svim featurima spremnim za ML model.

Kandidat već sadrži fuzzy scoreove iz get_candidates():
    f_ratio, f_partial, f_token_sort, f_token_set, f_wratio, name_norm

Grupe featura koje ovaj fajl dodaje:
    1. Fuzzy featurei         — preuzeti iz get_candidates(), normalizovani na [0,1]
    2. Strukturni featurei    — dužine stringova, broj tokena
    3. Token overlap          — jaccard, first_token_sim, shared_fuzzy_tokens
    4. Jaro-Winkler           — edit distance varijanta dobra za tipfelere
    5. Cosine sim (opciono)   — sentence transformer embedding, kontroliše use_embeddings
    6. Kontekstualni          — rank kandidata u fuzzy listi

Parametar use_embeddings=False omogućava treniranje i poređenje modela
sa i bez sentence transformers — kao eksperiment koji bilježimo u MLflow.
Ako embeddinzi ne donose mjerljivu korist, izbacujemo ih iz produkcije.
"""

import numpy as np
import jellyfish
from rapidfuzz import fuzz

# Sentence transformers se importuju samo ako su potrebni (lazy load)
_embedding_model = None


def _get_embedding_model():
    """
    Lazy load sentence transformer modela.
    Model se učitava samo jednom — prvi put kada zatreba.
    Svaki sljedeći poziv vraća već učitani model iz memorije.
    """
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embedding_model = SentenceTransformer(
                "paraphrase-multilingual-MiniLM-L12-v2"
            )
        except ImportError:
            raise ImportError(
                "sentence-transformers nije instaliran. "
                "Pokreni: pip install sentence-transformers\n"
                "Ili koristi extract_features(..., use_embeddings=False)"
            )
    return _embedding_model


# ------------------------------------------------------------------ #
# Pomoćne funkcije
# ------------------------------------------------------------------ #

def _tokenize(text: str) -> list[str]:
    """Dijeli normalizovani string na tokene (riječi)."""
    return text.split() if text else []


def _jaccard(tokens_a: list[str], tokens_b: list[str]) -> float:
    """
    Jaccard sličnost između dva seta tokena.
    jaccard = |presjek| / |unija|

    Primjer:
        {"PERIC", "PERA"} vs {"PERIC", "PERO"}
        presjek = {"PERIC"} → 1
        unija   = {"PERIC", "PERA", "PERO"} → 3
        jaccard = 1/3 = 0.33

    Za razliku od fuzzy metrika, jaccard ne hvata djelimičnu sličnost
    između tokena — "PERA" i "PERO" se ne računaju kao match.
    Zato postoji shared_fuzzy_tokens ispod.
    """
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _first_token_similarity(tokens_a: list[str], tokens_b: list[str]) -> float:
    """
    Fuzzy sličnost prvog tokena (prezimena) između pdf i kandidata,
    normalizovana na [0, 1].

    Prezime je najstabilniji dio imena i jak signal za tačan match.

    Primjer:
        ["PERIC", "PERA"] vs ["PERIC", "PERO"] → ratio("PERIC","PERIC")/100 = 1.0
        ["PERIC", "PERA"] vs ["KOVAC", "PERA"] → ratio("PERIC","KOVAC")/100 = 0.18
    """
    if not tokens_a or not tokens_b:
        return 0.0
    return fuzz.ratio(tokens_a[0], tokens_b[0]) / 100.0


def _shared_fuzzy_tokens(
    tokens_a: list[str],
    tokens_b: list[str],
    threshold: int = 80
) -> int:
    """
    Broj tokena iz pdf_name koji imaju "par" u kandidatu
    sa fuzzy sličnošću >= threshold.

    Za razliku od jaccard-a, ovo hvata djelimična poklapanja:
        "PERA" ↔ "PERO"  → ratio=80 ✓  (jaccard bi ovo propustio)

    Primjer:
        pdf:  ["PERIC", "PERA"]
        cand: ["PERIC", "PERO"]
        "PERIC" ↔ "PERIC" → ratio=100 ✓
        "PERA"  ↔ "PERO"  → ratio=80  ✓
        → shared_fuzzy_tokens = 2
    """
    if not tokens_a or not tokens_b:
        return 0
    count = 0
    for ta in tokens_a:
        for tb in tokens_b:
            if fuzz.ratio(ta, tb) >= threshold:
                count += 1
                break
    return count


def _compute_cosine_similarity(pdf_name: str, cand_name: str) -> float:
    """
    Semantička sličnost između dva imena koristeći sentence transformer.
    Vraća cosine similarity u opsegu [0, 1].

    Koristimo originalna (nenormalizovana) imena jer multilingual model
    razumije srpska slova i dijakritičke znakove bolje od ASCII verzija.
    """
    from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
    model = _get_embedding_model()
    embeddings = model.encode([pdf_name, cand_name])
    score = sk_cosine([embeddings[0]], [embeddings[1]])[0][0]
    return float(np.clip(score, 0.0, 1.0))


# ------------------------------------------------------------------ #
# Glavna funkcija
# ------------------------------------------------------------------ #

def extract_features(
    pdf_name: str,
    pdf_name_norm: str,
    candidate: dict,
    candidate_rank: int = 1,
    use_embeddings: bool = False,
) -> dict:
    """
    Izvlači sve feateure za jedan par (pdf_name, kandidat).

    Parametri
    ----------
    pdf_name : str
        Originalno ime iz PDF-a, npr. "Peric Pera".
        Koristi se za cosine similarity (model razumije dijakritike).

    pdf_name_norm : str
        Normalizovano pdf ime iz fuzzy_matcher.normalize_name().
        Koristi se za sve string metrike.

    candidate : dict
        Dict iz get_candidates(). Sadrži:
            name, name_norm,
            f_ratio, f_partial, f_token_sort, f_token_set, f_wratio,
            customer_id, filijala

    candidate_rank : int
        Pozicija kandidata u fuzzy listi (1 = najbolji po wratio).
        Niži rank → veća vjerovatnoća tačnog matcha.

    use_embeddings : bool
        False (default) → brži, jednostavniji, bez sentence-transformers.
                          Koristi se za Eksperiment 1 i 2.
        True            → dodaje cosine_sim feature.
                          Koristi se za Eksperiment 3.
        Poređenje modela sa i bez ovog featura bilježimo u MLflow.

    Vraća
    ------
    dict sa svim featurima — jedan red u trening datasetu.
    Ključ "cosine_sim" postoji samo ako use_embeddings=True.
    """
    cand_name_norm = candidate.get("name_norm", "")
    cand_name      = candidate.get("name", "")

    # ------------------------------------------------------------------ #
    # 1. Fuzzy featurei
    #    get_candidates() već računa ove vrijednosti (0-100).
    #    Normalizujemo na [0, 1] da svi featurei budu u istom opsegu.
    # ------------------------------------------------------------------ #
    f_ratio      = candidate["f_ratio"]      / 100.0
    f_partial    = candidate["f_partial"]    / 100.0
    f_token_sort = candidate["f_token_sort"] / 100.0
    f_token_set  = candidate["f_token_set"]  / 100.0
    f_wratio     = candidate["f_wratio"]     / 100.0

    # ------------------------------------------------------------------ #
    # 2. Strukturni featurei
    # ------------------------------------------------------------------ #
    len_pdf  = len(pdf_name_norm)
    len_cand = len(cand_name_norm)

    # Apsolutna razlika u dužini stringa
    len_diff = abs(len_pdf - len_cand)

    # Odnos dužina — uvijek između 0 i 1
    # Primjer: "PERA" (4) vs "PERICA" (6) → 4/6 = 0.67
    len_ratio = (
        min(len_pdf, len_cand) / max(len_pdf, len_cand)
        if max(len_pdf, len_cand) > 0
        else 1.0
    )

    tokens_pdf  = _tokenize(pdf_name_norm)
    tokens_cand = _tokenize(cand_name_norm)

    # Razlika u broju tokena
    # Primjer: "PERIC PERA" (2) vs "PERIC PERA MILAN" (3) → diff=1
    token_count_diff = abs(len(tokens_pdf) - len(tokens_cand))

    # ------------------------------------------------------------------ #
    # 3. Token overlap featurei
    # ------------------------------------------------------------------ #
    jaccard              = _jaccard(tokens_pdf, tokens_cand)
    first_token_sim      = _first_token_similarity(tokens_pdf, tokens_cand)
    shared_fuzzy_tokens  = _shared_fuzzy_tokens(tokens_pdf, tokens_cand, threshold=80)

    # ------------------------------------------------------------------ #
    # 4. Jaro-Winkler
    #    Dobar za tipfelere, posebno na početku stringa.
    #    Primjer: "PERICH" vs "PERIC" → 0.97  (samo jedan karakter razlike)
    # ------------------------------------------------------------------ #
    jaro_winkler = jellyfish.jaro_winkler_similarity(pdf_name_norm, cand_name_norm)

    # ------------------------------------------------------------------ #
    # 5. Kontekstualni featurei
    # ------------------------------------------------------------------ #
    features = {
        # Fuzzy (normalizovani na [0, 1])
        "f_ratio":             f_ratio,
        "f_partial":           f_partial,
        "f_token_sort":        f_token_sort,
        "f_token_set":         f_token_set,
        "f_wratio":            f_wratio,

        # Strukturni
        "len_diff":            len_diff,
        "len_ratio":           len_ratio,
        "token_count_diff":    token_count_diff,

        # Token overlap
        "jaccard":             jaccard,
        "first_token_sim":     first_token_sim,
        "shared_fuzzy_tokens": shared_fuzzy_tokens,

        # Edit distance
        "jaro_winkler":        jaro_winkler,

        # Kontekstualni — rank=1 je jak signal, rank=3 slab
        "candidate_rank":      candidate_rank,
    }

    # ------------------------------------------------------------------ #
    # 6. Cosine similarity (opciono, Eksperiment 3)
    #    Dodaje se samo kada use_embeddings=True.
    #    Ako razlika u F1/precision/recall nije značajna,
    #    ovaj feature izbacujemo iz produkcije radi jednostavnosti deploya.
    # ------------------------------------------------------------------ #
    if use_embeddings:
        features["cosine_sim"] = _compute_cosine_similarity(pdf_name, cand_name)

    return features