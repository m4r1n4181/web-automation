"""
dataset_builder.py

Gradi trening dataset iz ručno unijetih parova u name_aliases tabeli.

Četiri vrste primjera
----------------------
1. Pozitivni       — tačan par iz name_aliases                      label=1
2. Easy negatives  — fuzzy top 3 kandidati koji nisu tačni          label=0
3. Hard negatives  — rank 4-10, wratio > HARD_NEG_THRESHOLD         label=0
                     Jako slični ali pogrešni — najtežiji za model
4. Random negative — 1 nasumičan korisnik iz iste filijale          label=0
                     Jasno pogrešan, ali drži model "prizemljenim"

Rezultat
---------
Bez embeddings: ~620 redova  →  data/dataset.csv
Sa embeddings:  ~620 redova  →  data/dataset_with_embeddings.csv

Pokretanje
----------
    python -m src.ml.dataset_builder
    python -m src.ml.dataset_builder --embeddings
"""

import argparse
import random
import pandas as pd
from pathlib import Path

try:
    from src.database import Database
    from src.fuzzy_matcher import get_candidates, normalize_name
    from src.ml.feature_extractor import extract_features
    from src.logger import get_logger
except ImportError:
    from src.database import Database
    from src.fuzzy_matcher import get_candidates, normalize_name
    from src.ml.feature_extractor import extract_features
    from src.logger import get_logger

logger = get_logger(__name__)

# ------------------------------------------------------------------ #
# Konfiguracija
# ------------------------------------------------------------------ #

# Koliko fuzzy kandidata uzimamo ukupno
# Top 3  → easy negatives (rank 1-3, minus tačan)
# Top 10 → hard negatives (rank 4-10, wratio > praga)
TOTAL_CANDIDATES   = 10
EASY_NEG_TOP_N     = 3

# Hard negative: kandidat mora biti dovoljno sličan da bude "težak"
HARD_NEG_THRESHOLD = 75   # wratio > 75 znači jako sličan ali pogrešan

# Koliko random negativnih po pozitivnom primjeru
RANDOM_NEG_PER_POS = 1

# Seed za reproduktibilnost eksperimenata
RANDOM_SEED = 42

DATASET_PATH       = Path("data/dataset.csv")
DATASET_EMBED_PATH = Path("data/dataset_with_embeddings.csv")


# ------------------------------------------------------------------ #
# Pomoćne funkcije
# ------------------------------------------------------------------ #

def _load_confirmed_aliases(db: Database) -> list[dict]:
    """Učitava potvrđene parove iz name_aliases (confirmed=1)."""
    with db.connection() as conn:
        rows = conn.execute("""
            SELECT pdf_name, web_name
            FROM name_aliases
            WHERE confirmed = 1
        """).fetchall()
    aliases = [dict(r) for r in rows]
    logger.info(f"Učitano {len(aliases)} potvrđenih aliasa.")
    return aliases


def _get_filijala_for_web_name(db: Database, web_name: str) -> str | None:
    """
    Traži filijalu tačnog kandidata u customers tabeli.
    name_aliases ne čuva filijalu direktno, pa je tražimo po web_name.
    """
    with db.connection() as conn:
        row = conn.execute(
            "SELECT filijala FROM customers WHERE name = ? LIMIT 1",
            (web_name,)
        ).fetchone()
    return row["filijala"] if row else None


def _get_all_customers_in_filijala(db: Database, filijala: str) -> list[dict]:
    """
    Vraća sve korisnike iz date filijale.
    Koristi se za nasumični odabir random negativnih primjera.
    """
    with db.connection() as conn:
        rows = conn.execute(
            "SELECT * FROM customers WHERE UPPER(filijala) = UPPER(?)",
            (filijala,)
        ).fetchall()
    return [dict(r) for r in rows]


def _make_record(
    pdf_name: str,
    pdf_name_norm: str,
    candidate: dict,
    rank: int,
    label: int,
    neg_type: str,
    use_embeddings: bool,
) -> dict:
    """
    Pravi jedan red u datasetu za par (pdf_name, kandidat).

    neg_type opisuje vrstu negativnog primjera — koristimo ga
    za analizu u notebooku, ne ide u model.
    """
    features = extract_features(
        pdf_name=pdf_name,
        pdf_name_norm=pdf_name_norm,
        candidate=candidate,
        candidate_rank=rank,
        use_embeddings=use_embeddings,
    )
    return {
        "pdf_name":       pdf_name,
        "candidate_name": candidate["name"],
        "filijala":       candidate.get("filijala", ""),
        "label":          label,
        "neg_type":       neg_type,   # "positive" | "easy" | "hard" | "random"
        **features,
    }


# ------------------------------------------------------------------ #
# Glavna funkcija
# ------------------------------------------------------------------ #

def build_dataset(use_embeddings: bool = False) -> pd.DataFrame:
    """
    Gradi kompletan trening dataset sa sve četiri vrste primjera.

    Parametri
    ----------
    use_embeddings : bool
        Ako True, dodaje cosine_sim feature (Eksperiment 3).

    Vraća
    ------
    pd.DataFrame spreman za treniranje.
    """
    random.seed(RANDOM_SEED)
    db      = Database()
    aliases = _load_confirmed_aliases(db)

    records = []
    stats   = {"positive": 0, "easy": 0, "hard": 0, "random": 0, "skipped": 0}

    for alias in aliases:
        pdf_name = alias["pdf_name"]
        web_name = alias["web_name"]

        # 1) Pronađi filijalu
        filijala = _get_filijala_for_web_name(db, web_name)
        if not filijala:
            logger.warning(f"Filijala nije pronađena za '{web_name}' — preskačem.")
            stats["skipped"] += 1
            continue

        pdf_name_norm    = normalize_name(pdf_name)
        web_name_norm    = normalize_name(web_name)

        # 2) Uzmi top TOTAL_CANDIDATES kandidata (za hard negative pool)
        # filijala=None: filijala ovdje dolazi od TAČNOG kupca, pa bi bonus filijale
        # tačnom kandidatu odao labelu (data leakage). Rangiranje bez filijale.
        all_candidates   = get_candidates(pdf_name, filijala=None, top_n=TOTAL_CANDIDATES)
        easy_candidates  = all_candidates[:EASY_NEG_TOP_N]

        if not all_candidates:
            logger.warning(f"Nema kandidata za '{pdf_name}' — preskačem.")
            stats["skipped"] += 1
            continue

        # 3) Provjeri da li je tačan kandidat unutar easy poola (top 3)
        #    Ako nije, fuzzy ga uopće nije vratio — bez pozitivnog primjera nema smisla
        found_in_easy = any(
            normalize_name(c["name"]) == web_name_norm
            for c in easy_candidates
        )
        if not found_in_easy:
            logger.warning(
                f"'{web_name}' nije u top {EASY_NEG_TOP_N} za '{pdf_name}' — preskačem."
            )
            stats["skipped"] += 1
            continue

        # ------------------------------------------------------------------ #
        # A) Pozitivni primjer + easy negatives  (rank 1-3)
        # ------------------------------------------------------------------ #
        for rank, candidate in enumerate(easy_candidates, start=1):
            cand_norm = normalize_name(candidate["name"])
            is_match  = (cand_norm == web_name_norm)
            label     = 1 if is_match else 0
            neg_type  = "positive" if is_match else "easy"

            records.append(_make_record(
                pdf_name, pdf_name_norm, candidate,
                rank, label, neg_type, use_embeddings
            ))
            stats[neg_type] += 1

        # ------------------------------------------------------------------ #
        # B) Hard negatives  (rank 4-10, wratio > HARD_NEG_THRESHOLD)
        #    Kandidati koji su jako slični ali sigurno nisu tačni.
        #    Ovo je najvažnija dopuna — model mora naučiti da razlikuje
        #    visoke wratio score-ove koji su ipak pogrešni.
        # ------------------------------------------------------------------ #
        hard_pool = all_candidates[EASY_NEG_TOP_N:]   # rank 4-10
        for rank, candidate in enumerate(hard_pool, start=EASY_NEG_TOP_N + 1):
            cand_norm = normalize_name(candidate["name"])

            # Preskoči ako je slučajno tačan (ne bi trebalo biti u hard poolu)
            if cand_norm == web_name_norm:
                continue

            # Samo ako je dovoljno sličan da bude "težak"
            if candidate["f_wratio"] > HARD_NEG_THRESHOLD:
                records.append(_make_record(
                    pdf_name, pdf_name_norm, candidate,
                    rank, 0, "hard", use_embeddings
                ))
                stats["hard"] += 1

        # ------------------------------------------------------------------ #
        # C) Random negative  (1 nasumičan korisnik iz iste filijale)
        #    Jasno pogrešan primjer — drži model "prizemljenim" na
        #    niskim score-ovima. Bez ovih, model nikad ne vidi lak slučaj.
        # ------------------------------------------------------------------ #
        all_in_filijala = _get_all_customers_in_filijala(db, filijala)

        # Filtriraj tačnog kandidata i sve koji su već u top 10
        top10_names = {normalize_name(c["name"]) for c in all_candidates}
        random_pool = [
            c for c in all_in_filijala
            if normalize_name(c.get("name", "")) not in top10_names
            and normalize_name(c.get("name", "")) != web_name_norm
        ]

        if random_pool:
            random_cand_raw = random.choice(random_pool)

            # random_pool dolazi direktno iz customers tabele (ne iz get_candidates)
            # pa nema fuzzy scoreove — dodajemo ih ručno da extract_features radi
            random_cand_norm = normalize_name(random_cand_raw.get("name", ""))
            from rapidfuzz import fuzz as _fuzz
            random_candidate = {
                **random_cand_raw,
                "name_norm":    random_cand_norm,
                "f_ratio":      _fuzz.ratio(pdf_name_norm, random_cand_norm),
                "f_partial":    _fuzz.partial_ratio(pdf_name_norm, random_cand_norm),
                "f_token_sort": _fuzz.token_sort_ratio(pdf_name_norm, random_cand_norm),
                "f_token_set":  _fuzz.token_set_ratio(pdf_name_norm, random_cand_norm),
                "f_wratio":     _fuzz.WRatio(pdf_name_norm, random_cand_norm),
            }

            records.append(_make_record(
                pdf_name, pdf_name_norm, random_candidate,
                TOTAL_CANDIDATES + 1,   # rank van normalnog opsega
                0, "random", use_embeddings
            ))
            stats["random"] += 1

    df = pd.DataFrame(records)

    logger.info(
        f"\nDataset izgrađen:\n"
        f"  Pozitivni      : {stats['positive']}\n"
        f"  Easy negatives : {stats['easy']}\n"
        f"  Hard negatives : {stats['hard']}\n"
        f"  Random neg.    : {stats['random']}\n"
        f"  Preskočeno     : {stats['skipped']}\n"
        f"  UKUPNO         : {len(df)}"
    )

    return df


def save_dataset(df: pd.DataFrame, path: Path) -> None:
    """Čuva dataset na disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    logger.info(f"Dataset sačuvan: {path}  ({len(df)} redova)")


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gradi ML trening dataset.")
    parser.add_argument(
        "--embeddings",
        action="store_true",
        help="Dodaj cosine_sim feature (sporije, za Eksperiment 3)"
    )
    args = parser.parse_args()

    output_path = DATASET_EMBED_PATH if args.embeddings else DATASET_PATH

    logger.info(f"Gradim dataset — embeddinzi: {'DA' if args.embeddings else 'NE'}")
    df = build_dataset(use_embeddings=args.embeddings)

    if not df.empty:
        save_dataset(df, output_path)

        print("\n=== Pregled dataseta ===")
        print(f"Ukupno redova  : {len(df)}")
        print(f"Pozitivnih     : {(df['label'] == 1).sum()}")
        print(f"Negativnih     : {(df['label'] == 0).sum()}")
        print(f"\nRaspodjela po tipu:")
        print(df["neg_type"].value_counts())
        print(f"\nKolone: {list(df.columns)}")
        print(f"\nPrvih 5 redova:")
        print(df.head())
        print(f"\nSačuvano u: {output_path}")